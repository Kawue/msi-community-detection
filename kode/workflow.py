from sys import argv
import os
import pandas as pd
import simplejson as json
import argparse
from scipy.stats.mstats import winsorize
from kode.similarity_measures import *
from kode.edge_reduction import *
from kode.community_detection import *
from kode.json_factory import *
from kode.mmm_own import *

def workflow(h5_data, ds_idx, similarity_measure, community_method, transform, transform_params, savepath, hdf5_name):
	# Winsorize data
	winsorize(h5_data, limits=(0, 0.01), axis=0, inplace=True)

	# Convert hdf data in numpy array; desired structure Samples = m/z-images, Features = Pixel
	data = h5_data.as_matrix().transpose()

	# Calculate similarity matrix
	similarity_matrix = similarity_measure(data)
	print("Similarity Matrix Calculation Done!")

	# Transform similarity matrix to adjacency matrix

	if transform == "pca":
		if transform_params is None:
			adjacency_matrix, edge_reduction_threshold = transform_by_pca(similarity_matrix, [-1, 1], 200)
		else:
			if len(transform_params) != 3:
				raise ValueError("Wrong parameter for Transformation!")
			lower = transform_params[0]
			upper = transform_params[1]
			step = transform_params[2]
			adjacency_matrix, edge_reduction_threshold = transform_by_pca(similarity_matrix, [lower, upper], step)
	if transform == "statistics":
		center = np.mean(similarity_matrix)
		dev =  np.std(similarity_matrix)
		if transform_params is None:
			C = 1
			adjacency_matrix = transform_by_global_statistics(similarity_matrix, center, dev, C)
			edge_reduction_threshold = center + dev*C
		else:
			if len(transform_params) != 1:
				raise ValueError("Wrong parameter for Transformation!")
			C = transform_params[0]
			adjacency_matrix = transform_by_global_statistics(similarity_matrix, center, dev, C)
			edge_reduction_threshold = center + dev*C
	print("Adjecency Matrix Calculation Done!")

	# Transform weighted adjacency matrix to unweighted
	adjacency_matrix_binary = adjacency_matrix.astype(bool).astype(float)
	adjacency_matrix = adjacency_matrix.astype(float)

	np.save(os.path.join(savepath, "adjacency-matrix-%s"%(hdf5_name)), adjacency_matrix)

	hierarchy_dict = {}
	# Index if multiple graphs will be saved later on.
	hierarchy_dict["graph_idx"] = ds_idx
	lvl = -1
	while True:
		try:
			lvl += 1
			# Calculate communities
			if community_method == "louvain":
				community_list, _, dendro, inv_dendro = calc_louvain(adjacency_matrix_binary, level=lvl,return_c_graph=True)
			elif community_method == "eigenvector":
				if lvl > 0:
					raise ValueError("Non hierarchical method, only one hierarchy is computed")
				community_list = leading_eigenvector_community(adjacency_matrix, None, False, False, None)
				dendro, inv_dendro = calc_dendro_for_ig(community_list)


			# Sort communities by length
			#sorted_community_list = sorted(community_list, key=len)

			# Calculate membership list
			membership_list = []
			for vertex in range(len(adjacency_matrix_binary)):
				for membership_id, community in enumerate(community_list):
					if vertex in community:
						membership_list.append(membership_id)

			# Calculate graph
			graph = base_graph_structure(h5_data, adjacency_matrix_binary)

			# Calculate communities as clustering object
			communities = ig.VertexClustering(graph, membership_list)

			print("")
			print("Number of Communities: " + str(len(list(communities))))

			# Add community membership as attribute
			for v in graph.vs:
				v["membership"] = communities.membership[v.index]

			# Calculate unweighted modularity
			modularity = communities.modularity
			print("")
			print("Modularity: " + str(modularity))
			print("Community Calculation Done!")

			# Calculate community-graph
			c_graph = community_graph(communities, graph, "mean")

			hierarchy_dict[lvl] = {"communities": communities, "graph": graph, "cgraph": c_graph}

		except Exception as e:
			hierarchy_dict["dendro"] = dendro
			hierarchy_dict["inv_dendro"] = inv_dendro
			print(e)
			break

	return h5_data, hierarchy_dict, adjacency_matrix, edge_reduction_threshold

def workflow_exec():
	parser = argparse.ArgumentParser(description="Create an MSI Image Graph and calculate MSI Communities. Also, produce a JSON for GRINE.")
	parser.add_argument("-d", "--datapath", action="store", dest="datapath", type=str, required=True, help="Path to HDF5 folder or file.")
	parser.add_argument("-p", "--savepath", action='store', dest='savepath', type=str, required=True, help="Path to save the resulting JSON.")
	parser.add_argument("-sm", "--similarity", action="store", dest="similarity", type=str, choices=["pearson", "cosine", "euclidean", "euclidean2"], required=True, help="Similarity method to use.")
	parser.add_argument("-cm", "--community", action='store', dest='community', type=str, choices=["eigenvector", "louvain"], required=True, help="Community detection method to use.")
	parser.add_argument("-tm", "--transformation", action="store", dest="transformation", type=str, choices=["pca", "statistics"], required=True, help="Transformation method to use.")
	parser.add_argument("-tp", "--transformationparams", default=None, action="store", dest="transformationparams", type=float, nargs="+", required=False, help="Transformation parameters to use (optional, otherwise default is applied).")
	args = parser.parse_args()
		
	similarity_measures_dict = {
		"pearson": calc_pearson_correlation,
		"consine": calc_cosine_similarity,
		"euclidean": calc_euclidean_distance,
		"euclidean2": calc_normalized_euclidean_distance
	}
	

	h5_files = []
	fnames = []
	if os.path.isfile(args.datapath):
		fnames.append(os.path.basename(args.datapath).split(".")[0])
		h5_files.append(pd.read_hdf(args.datapath))
	elif os.path.isdir(args.datapath):
		for r, ds, fs in os.walk(args.datapath):
			for f in fs:
				if ".h5" in f:
					fnames.append(os.path.basename(f).split(".")[0])
					h5_files.append(pd.read_hdf(os.path.join(r,f)))
	else:
		raise ValueError("Given datapath is no file or dir!")

	similarity_measure = similarity_measures_dict[args.similarity]
	community_method = args.community
	transform = args.transformation
	transform_params = args.transformationparams
	

	json_dict = {"graphs": {}}
	for ds_idx, h5_file in enumerate(h5_files):
		h5_data, hierarchy_dict, adjacency_matrix, threshold = workflow(h5_file, ds_idx, similarity_measure, community_method, transform, transform_params, os.path.dirname(args.savepath), fnames[ds_idx])
		try:
			dataset_name = h5_data.index.get_level_values("dataset")[0]
		except:
			dataset_name = fnames[ds_idx]
		json_dict = build_json(hierarchy_dict, h5_data, dataset_name, nx.from_numpy_array(adjacency_matrix), json_dict, threshold)

	if not os.path.isdir(os.path.dirname(args.savepath)):
		os.makedirs(os.path.dirname(args.savepath))

	f = open(args.savepath, "w")
	with f as outputfile:
		json.dump(json_dict, outputfile)