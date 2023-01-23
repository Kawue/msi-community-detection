from sys import argv
import os
import pandas as pd
import argparse
from kode.community_detection import *

# Calculate Louvain method for community detection
# Level is calculated from behind, i.e. 0 is the highest level, i.e. level-n.
def calc_louvain(adj_matrix, level = 0, return_c_graph = False):
	nx_G = nx.from_numpy_array(adj_matrix)
	dendro = louvain.generate_dendrogram(nx_G, randomize=False, random_state=0) #Maybe set randomize True
	level = len(dendro) - level - 1

	if level < 0:
		raise Exception("The given Level is too deep. The maximum is: " + str(len(dendro)-1))

	communities = louvain.partition_at_level(dendro, level)
	number_communities = max(communities, key = lambda x: communities[x]) + 1

	# Maybe unnecessary after some code rework and unification
	community_list = []
	for i in range(number_communities):
		grp_list = []
		for grp in communities:
			if communities[grp] == i:
				grp_list.append(grp)
		else:
			if grp_list:
				community_list.append(grp_list)

	community_level_G = louvain.induced_graph(communities, nx_G)

	if return_c_graph:
		c_level_graph = nx.adjacency_matrix(community_level_G)
	else:
		c_level_graph = None

	inv_dendro = []
	for dct in dendro:
		inv_dct = {}
		for k,v in dct.items(): 
			inv_dct.setdefault(v,[]).append(k)
		inv_dendro.append(inv_dct)

	return community_list, c_level_graph, dendro, inv_dendro



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



def workflow_extern(similarity_matrix, transform=None, lower=None, upper=None, step=None, normalize=None, intersect=None, center_fct=None, dev_fct=None, C=None, community_method=None, savepath=None):
	if not (np.diag(similarity_matrix) == 1).all():
		raise ValueError("Diagonal of similarity matrix must be one.")

	if transform == "statistics":
		upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
		if center_fct == "mean":
			center = np.mean(upper_triangle)
		elif center_fct == "median":
			center = np.median(upper_triangle)
		if dev_fct == "std":
			dev =  np.std(upper_triangle)
		elif dev_fct == "mad":
			if center_fct == "mean":
				dev = np.mean(np.abs(upper_triangle - center))
			elif center_fct == "median":
				dev = np.median(np.abs(upper_triangle - center))
		adjacency_matrix = transform_by_global_statistics(similarity_matrix, center, dev, C)
		edge_reduction_threshold = center + dev*C
		print("Chosen threshold: %f"%(center+dev*C))
	if transform == "modularity_weighted" or transform == "modularity_unweighted":
		adjacency_matrix, edge_reduction_threshold = modularity_optimization(similarity_matrix, transform, community_method, lower, upper, step)

	# Transform weighted adjacency matrix to unweighted
	adjacency_matrix_binary = adjacency_matrix.astype(bool).astype(float)

	lvl = -1
	while True:
		try:
			lvl += 1
			# Calculate communities
			if community_method == "louvain":
				community_list, _, _, _ = calc_louvain(adjacency_matrix_binary, level=lvl, return_c_graph=True)
			# Calculate membership list
			membership_list = []
			for vertex in range(len(adjacency_matrix_binary)):
				for membership_id, community in enumerate(community_list):
					if vertex in community:
						membership_list.append(membership_id)
		except Exception as e:
			print(e)
			break
	
	return membership_list