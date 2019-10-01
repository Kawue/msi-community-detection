import networkx as nx
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy import stats
from itertools import permutations
from kode.pca import pca
from math import acos, ceil, sqrt
import skimage.filters as skif

def transform_by_pca(similarity_matrix, similarity_intervall, stepnumber):
	pool = Pool(processes=cpu_count())
	threshold_list = np.linspace(similarity_intervall[0], similarity_intervall[1], stepnumber)

	# Calculate Average Clustering Coefficient
	acc_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, calc_acc, (), False,))
	# Calculate Global Efficiency
	eff_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, calc_global_eff, (), False,))
	# Calculate total number of edges
	nb_edges_result = pool.apply_async(calc_topology_pca_method, args=(similarity_matrix, threshold_list, count_total_nb_edges, (), False,))

	acc_result.wait()
	eff_result.wait()
	nb_edges_result.wait()

	# Normalize every value into [0,1]
	acc_values = normalize(acc_result.get())
	eff_values = normalize(eff_result.get())
	nb_edges_values = normalize(nb_edges_result.get())

	# Use the total number of edges as baseline
	acc_diff_edges = [x - y for x, y in zip(acc_values, nb_edges_values)]
	eff_diff_edges = [x - y for x, y in zip(eff_values, nb_edges_values)]
	# Build a matrix with baselined ACC and Global Efficiency. Samples = ACC & Eff, Features = Threshold
	matrix = np.array([acc_diff_edges, eff_diff_edges])
	# Samples = Threshold, Features = ACC & Eff
	matrix = matrix.T

	# Calculate PCA transformed data matrix
	matrix_transformed = pca(matrix.copy(), False, 2)

	# Extract the first PCA Component, i.e. data projection on the first pca axis (eigenvector)
	first_pca_component = matrix_transformed[:, 0]

	# Plotting stuff
	plt.figure()
	plt.title("Network Measures", fontsize=20, y=1.01)
	plt.xlabel("Candidate Thresholds ($\mathbf{t}$)", fontsize=20, labelpad=15)
	plt.ylabel("Measure Values", fontsize=20, labelpad=15)
	plt.xticks(size=15)
	plt.yticks(size=15)
	threshold_list = threshold_list[:]
	plt.plot(threshold_list, first_pca_component, "-d", color="red", label="$\mathbf{y}$")
	plt.plot(threshold_list, nb_edges_values[:], "-^", color="black", label=r"$\mathbf{\nu}^{N_E}$")
	plt.plot(threshold_list, acc_values[:], "-s", color="blue", label=r"$\mathbf{\nu}^\zeta$")
	plt.plot(threshold_list, eff_values[:], "-D", color="peru", label=r"$\mathbf{\nu}^\xi$")
	plt.plot(threshold_list, acc_diff_edges[:], "-p", color="darkviolet", label=r"$\mathbf{\eta}^\zeta$")
	plt.plot(threshold_list, eff_diff_edges[:], "-o", color="brown", label=r"$\mathbf{\eta}^\xi$")
	plt.legend(fontsize=15)
	plt.show()


	max_value = np.amax(first_pca_component)
	max_idx = np.argmax(first_pca_component)
	t = threshold_list[max_idx]
	print("Maximum Value of First PCA Component: " + str(max_value))
	print("Index of Maximum of First PCA Component: " + str(max_idx))
	print("Selected Threshold by PCA: " + str(t))

	return transform_by_global_statistics(similarity_matrix, t, 0, 0), t


# Function to calculate most of the given topology measures in combination with multiprocessing.
def calc_topology_pca_method(similarity_matrix, threshold_list, measure, args, normalized):
	values = []
	print(measure)
	print(args)
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