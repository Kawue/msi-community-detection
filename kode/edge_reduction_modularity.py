import os
import networkx as nx
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy import stats
from itertools import permutations
from kode.pca import pca
from math import acos, ceil, sqrt
import skimage.filters as skif
from kode.community_detection import *
from kode.mmm_own import *

def build_hierarchy_dict(h5_data, ds_idx, community_method, adjacency_matrix_binary):
	hierarchy_dict = {}
	# Index if multiple graphs will be saved later on.
	hierarchy_dict["graph_idx"] = ds_idx
	lvl = -1
	while True:
		try:
			lvl += 1
			# Calculate communities
			if community_method == "louvain":
				community_list, _, dendro, inv_dendro = calc_louvain(adjacency_matrix_binary, level=lvl, return_c_graph=True)
			elif community_method == "eigenvector":
				if lvl > 0:
					raise ValueError("Non hierarchical method, only one hierarchy is computed")
				community_list = leading_eigenvector_community(adjacency_matrix_binary, None, False, False, None)
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
			print(e)
			hierarchy_dict["dendro"] = dendro
			hierarchy_dict["inv_dendro"] = inv_dendro
			break

	return hierarchy_dict, community_list


def transform_by_pca(similarity_matrix, transform_method, similarity_intervall, stepnumber, normalization, intersect, community_method, h5_data, ds_idx, savepath):
	threshold_list = np.linspace(similarity_intervall[0], similarity_intervall[1], stepnumber)
	
	# Calculate total number of edges
	nb_edges_result = calc_topology_pca_method(similarity_matrix, threshold_list, count_total_nb_edges, (), False)
	nb_edges_values = normalize(nb_edges_result)

	if transform_method in ["pca"]:
		community_list = None
		# Calculate Average Clustering Coefficient
		acc_result = calc_topology_pca_method(similarity_matrix, threshold_list, calc_acc, (), False)
		# Calculate Global Efficiency
		eff_result = calc_topology_pca_method(similarity_matrix, threshold_list, calc_global_eff, (), False)
		
		# Normalize every value into [0,1]
		acc_values = normalize(acc_result)
		eff_values = normalize(eff_result)

		# Use the total number of edges as baseline
		acc_diff_edges = np.array([x - y for x, y in zip(acc_values, nb_edges_values)])
		eff_diff_edges = np.array([x - y for x, y in zip(eff_values, nb_edges_values)])
		if normalization:
			acc_diff_edges = (acc_diff_edges - np.amin(acc_diff_edges)) / (np.amax(acc_diff_edges) - np.amin(acc_diff_edges))
			eff_diff_edges = (eff_diff_edges - np.amin(eff_diff_edges)) / (np.amax(eff_diff_edges) - np.amin(eff_diff_edges))
		# Build a matrix with baselined ACC and Global Efficiency. Samples = ACC & Eff, Features = Threshold
		matrix = np.array([acc_diff_edges, eff_diff_edges])
		# Samples = Threshold, Features = ACC & Eff
		matrix = matrix.T

		# Calculate PCA transformed data matrix
		matrix_transformed = pca(matrix.copy(), False, 2)

		# Extract the first PCA Component, i.e. data projection on the first pca axis (eigenvector)
		first_pca_component = matrix_transformed[:, 0]
		second_pca_component = matrix_transformed[:, 1]
		#mean_pca_component = matrix_transformed.mean(axis=1)

		if intersect:
			idx = np.where(np.diff(np.sign(acc_diff_edges-eff_diff_edges)))[0][-1]
			t = threshold_list[idx]
			print("Intersection method used.")
		else:
			if first_pca_component[0] > 0 and first_pca_component[-1] > 0:
				print()
				print("Second PCA Component is selected!")
				print()
				#max_value = np.amax(second_pca_component)
				#max_idx = np.argmax(second_pca_component)
				max_value = np.amin(first_pca_component)
				max_idx = np.argmin(first_pca_component)
				t = threshold_list[max_idx]
			else:
				print()
				print("First PCA Component is selected!")
				print()
				max_value = np.amax(first_pca_component)
				max_idx = np.argmax(first_pca_component)
				t = threshold_list[max_idx]
			print("Maximum Value of First PCA Component: " + str(max_value))
			print("Index of Maximum of First PCA Component: " + str(max_idx))
		print("Selected Threshold by PCA: " + str(t))


	if transform_method in ["modularity", "coverage", "performance"]:
		weighted=True
		modularity_values, coverage_values, performance_values, communities, hierarchy_dicts = calc_topology_community_statistics(similarity_matrix, threshold_list, community_method, transform_method, weighted, normalization, h5_data, ds_idx)
		modularity_values = (1-normalize(modularity_values)) - nb_edges_values[:]
		coverage_values = (1-normalize(coverage_values)) - nb_edges_values[:]
		performance_values = (1-normalize(performance_values)) - nb_edges_values[:]
		if transform_method == "modularity":
			max_value = np.amin(modularity_values)
			max_idx = np.argmin(modularity_values)
		elif transform_method == "coverage":
			max_value = np.amin(coverage_values)
			max_idx = np.argmin(coverage_values)
		elif transform_method == "performance":
			max_value = np.amin(performance_values)
			max_idx = np.argmin(performance_values)
		else:
			raise ValueError("Wrong transform method!")
		t = threshold_list[max_idx]
		community_list = communities[max_idx]
		hierarchy_dict = hierarchy_dicts[max_idx]

	# Plotting stuff
	if savepath:
		plt.figure()
		#fig = plt.gcf()
		#fig.set_size_inches(32, 18)
		#fig.tight_layout()
		plt.title("Network Measures", fontsize=20, y=1.01)
		plt.xlabel("Candidate Thresholds ($\mathbf{t}$)", fontsize=20, labelpad=15)
		plt.ylabel("Measure Values", fontsize=20, labelpad=15)
		plt.xticks(size=15)
		plt.yticks(size=15)
		threshold_list = threshold_list[:]
		if transform_method in ["pca"]:
			plt.plot(threshold_list, first_pca_component, "-X", color="red", label=r"$\mathbf{y_0}$")
			plt.plot(threshold_list, second_pca_component, "-X", color="hotpink", label=r"$\mathbf{y_1}$")
			plt.plot(threshold_list, nb_edges_values[:], "-^", color="black", label=r"$\mathbf{\nu}^{N_E}$")
			plt.plot(threshold_list, acc_values[:], "-s", color="blue", label=r"$\mathbf{\nu}^\zeta$")
			plt.plot(threshold_list, eff_values[:], "-D", color="peru", label=r"$\mathbf{\nu}^\xi$")
			plt.plot(threshold_list, acc_diff_edges[:], "-p", color="darkviolet", label=r"$\mathbf{\eta}^\zeta$")
			plt.plot(threshold_list, eff_diff_edges[:], "-o", color="brown", label=r"$\mathbf{\eta}^\xi$")
		
		if transform_method in ["modularity", "coverage", "performance"]:
			plt.plot(threshold_list, modularity_values, "-x", color="green", label="Mod2")
			plt.plot(threshold_list, coverage_values, "-x", color="lime", label="Cov2")
			plt.plot(threshold_list, performance_values, "-x", color="teal", label="Per2")
			
		plt.legend(fontsize=15)
		plt.show()
		if not os.path.exists(os.path.join(savepath, "community_detection_plots")):
			os.makedirs(os.path.join(savepath, "community_detection_plots"))
		plt.savefig(os.path.join(savepath, "community_detection_plots", "edge_reduction_plot.png"),bbox_inches='tight')

	return transform_by_global_statistics(similarity_matrix, t, 0, 0), t, community_list, hierarchy_dict


# Function to calculate most of the given topology measures in combination with multiprocessing.
def calc_topology_pca_method(similarity_matrix, threshold_list, measure, args, normalized):
	values = []
	#print(measure)
	#print(args)
	for t in threshold_list:
		thresholded_matrix = transform_by_global_statistics(similarity_matrix, t, 0, 0)
		G = nx.Graph(thresholded_matrix)
		values.append(measure(G, *args))

	if normalized:
		return normalize(np.array(values))
	else:
		return np.array(values)


def transform_by_global_statistics(similarity_matrix, center, dev, C):
	unlooped_matrix = similarity_matrix.copy()
	# Remove self-loops because it is obvious that the distance from a point to itself is 1
	np.fill_diagonal(unlooped_matrix, 0.0)
	transformed_matrix = np.zeros((len(unlooped_matrix), len(unlooped_matrix[0])))
	for x in range(len(unlooped_matrix)):
		for y in range(len(unlooped_matrix[x])):
			if not unlooped_matrix[x,y] < center + C * dev:
				transformed_matrix[x,y] = unlooped_matrix[x,y]
	return transformed_matrix


def normalize(values):
	max_val = max(values)
	min_val = min(values)
	return np.array([(x - min_val) / (max_val - min_val) for x in values])


def count_total_nb_edges(graph):
	G = graph
	return nx.number_of_edges(G)


# Average Clustering Coefficient
def calc_acc(graph, count_zeros=True):
	G = graph
	try:
		return nx.average_clustering(G, count_zeros=count_zeros)
	except ZeroDivisionError:
		print("ATTENTION: Division by Zero!")
		return 0


# Calculate the Efficiency between two nodes
def calc_eff(G, u, v):
	try:
		return 1 / nx.shortest_path_length(G, u, v)
	except nx.NetworkXNoPath:
		return 0


# Calculate the Global Efficiency of a network
def calc_global_eff(graph):
	G = graph
	n = len(G)
	denom = n * (n - 1)
	if denom != 0:
		return sum(calc_eff(G, u, v) for u, v in permutations(G, 2)) / denom
	else:
		return 0


def calc_topology_community_statistics(similarity_matrix, threshold_list, community_method, community_statistic, weighted, normalized, h5_data, ds_idx):
	modularity_values = []
	coverage_values = []
	performance_values = []
	communities = []
	hierarchy_dicts = []
	for t in threshold_list:
		thresholded_matrix = transform_by_global_statistics(similarity_matrix, t, 0, 0)
		G = nx.Graph(thresholded_matrix)
		hierarchy_dict, community_list = build_hierarchy_dict(h5_data, ds_idx, community_method, thresholded_matrix.astype(bool).astype(float))
		#community = calc_communities(thresholded_matrix.astype(bool).astype(float), community_method)
		if not weighted:
			thresholded_matrix = thresholded_matrix.astype(bool).astype(float)

		modularity_value = calc_modularity(G, community_list)
		coverage_value = calc_coverage(G, community_list)
		performance_value = calc_performance(G, community_list)

		modularity_values.append(modularity_value)
		coverage_values.append(coverage_value)
		performance_values.append(performance_value)

		communities.append(community_list)
		hierarchy_dicts.append(hierarchy_dict)

	if normalized:
		return normalize(np.array(modularity_values)), normalize(np.array(coverage_values)), normalize(np.array(performance_values)), communities, hierarchy_dicts
	else:
		return np.array(modularity_values), np.array(coverage_values), np.array(performance_values), communities, hierarchy_dicts

def calc_communities(adjacency_matrix_binary, community_method):
	lvl = -1
	while True:
		try:
			lvl += 1
			# Calculate communities
			if community_method == "louvain":
				communities, _, _, _ = calc_louvain(adjacency_matrix_binary, level=lvl, return_c_graph=True)
			elif community_method == "eigenvector":
				if lvl > 0:
					raise ValueError("Non hierarchical method, only one hierarchy is computed")
				communities = leading_eigenvector_community(adjacency_matrix_binary, None, False, False, None)
		except Exception as e:
			print(e)
			break
	return communities

# Modularity
def calc_modularity(graph, communities):
	return nx.algorithms.community.quality.modularity(graph, communities)

# Coverage
def calc_coverage(graph, communities):
	return nx.algorithms.community.quality.coverage(graph, communities)

# Performance
def calc_performance(graph, communities):
	return nx.algorithms.community.quality.performance(graph, communities)



def statistic(value_lists, statistic, normalize):
	if normalize:
		flat_values = [val for sublist in value_lists for val in sublist]
		max_val = max(flat_values)
		min_val = min(flat_values)
		value_lists_normalized = [[(x - min_val) / (max_val - min_val) for x in y] for y in value_lists]
	else:
		value_lists_normalized = value_lists

	if statistic == "min":
		return np.array([min(x) for x in value_lists_normalized])
	elif statistic == "max":
		return np.array([max(x) for x in value_lists_normalized])
	elif statistic == "mean":
		return np.array([np.mean(x) for x in value_lists_normalized])
	elif statistic == "median":
		return np.array([np.median(x) for x in value_lists_normalized])
	elif statistic == "std":
		return np.array([np.std(x) for x in value_lists_normalized])
	elif statistic == "modus":
		return np.array([stats.mode(x)[0][0] for x in value_lists_normalized])
	elif statistic == "entropy":
		return np.array([stats.entropy(x) for x in value_lists_normalized])
	else:
		raise ValueError("Wrong value for statistic!")


def min_max_based_interval(similarity_matrix):
	upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
	return [np.amin(upper_triangle), np.amax(upper_triangle)]


#communities : list or iterable of set of nodes
def modularity_optimization(similarity_matrix, transform, community_method, lower, upper, step, weight="weight"):
	threshold_list = np.linspace(lower, upper, step)



	pool = Pool(processes=cpu_count())
	acc_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, calc_acc, (), False,))
	eff_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, calc_global_eff, (), False,))
	nb_edges_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, count_total_nb_edges, (), False,))
	acc_result.wait()
	eff_result.wait()
	nb_edges_result.wait()
	acc_values = normalize(acc_result.get())
	eff_values = normalize(eff_result.get())
	nb_edges_values = normalize(nb_edges_result.get())
	acc_diff_edges = np.array([x - y for x, y in zip(acc_values, nb_edges_values)])
	eff_diff_edges = np.array([x - y for x, y in zip(eff_values, nb_edges_values)])
	matrix = np.array([acc_diff_edges, eff_diff_edges])
	matrix = matrix.T
	matrix_transformed = pca(matrix.copy(), False, 2)
	first_pca_component = matrix_transformed[:, 0]
	second_pca_component = matrix_transformed[:, 1]

	

	modularity_list = []
	for t in threshold_list:
		thresholded_matrix = transform_by_global_statistics(similarity_matrix, t, 0, 0)
		if transform == "modularity_weighted":
			modularity_G = nx.Graph(thresholded_matrix.astype(float))
		elif transform == "modularity_unweighted":
			modularity_G = nx.Graph(thresholded_matrix.astype(bool).astype(float))
		else:
			raise ValueError("Wrong transform method.")
		if community_method == "louvain":
			lvl = -1
			while True:
				try:
					lvl += 1
					community_list, _, _, _ = calc_louvain(thresholded_matrix.astype(bool).astype(float), level=lvl, return_c_graph=True)
				except:
					break
		elif community_method == "eigenvector":
			community_list = leading_eigenvector_community(thresholded_matrix.astype(bool).astype(float), None, False, False, None)
		modularity = nx.algorithms.community.modularity(modularity_G, community_list, weight=weight)
		modularity_list.append(modularity)
	plt.figure()
	plt.title("Modularity", fontsize=20, y=1.01)
	plt.xlabel("Candidate Thresholds ($\mathbf{t}$)", fontsize=20, labelpad=15)
	plt.ylabel("Modularity", fontsize=20, labelpad=15)
	plt.xticks(size=15)
	plt.yticks(size=15)
	plt.plot(threshold_list, normalize(modularity_list), "-X", color="green", label="$\mathcal{M}$")
	plt.plot(threshold_list, first_pca_component, "-X", color="red", label="$\mathbf{y0}$")
	plt.plot(threshold_list, second_pca_component, "-X", color="hotpink", label="$\mathbf{y1}$")
	plt.plot(threshold_list, nb_edges_values[:], "-^", color="black", label=r"$\mathbf{\nu}^{N_E}$")
	plt.plot(threshold_list, acc_values[:], "-s", color="blue", label=r"$\mathbf{\nu}^\zeta$")
	plt.plot(threshold_list, eff_values[:], "-D", color="peru", label=r"$\mathbf{\nu}^\xi$")
	plt.plot(threshold_list, acc_diff_edges[:], "-p", color="darkviolet", label=r"$\mathbf{\eta}^\zeta$")
	plt.plot(threshold_list, eff_diff_edges[:], "-o", color="brown", label=r"$\mathbf{\eta}^\xi$")
	plt.legend(fontsize=15)
	plt.show()
	max_value = np.amax(modularity_list)
	max_idx = np.argmax(modularity_list)
	t = threshold_list[max_idx]
	print("Maximum Modularity: " + str(max_value))
	print("Index of Maximum Modularity: " + str(max_idx))
	print("Selected Threshold by Modularity: " + str(t))
	return transform_by_global_statistics(similarity_matrix, t, 0, 0), t