from __future__ import print_function, division

import numpy as np
import scipy as sp
from numpy import corrcoef
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity as cosim
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numpy import sqrt


def calc_pearson_correlation(data_matrix):
	# Calculate pearson correlation matrix
	pearson_matrix = corrcoef(data_matrix)

	# Calculate upper half of pearson matrix without diagonal
	upper_triangle = pearson_matrix[np.triu_indices_from(pearson_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Pearson Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Pearson Matrix Maximum: " + str(np.amax(upper_triangle)))

	return pearson_matrix


def calc_euclidean_distance(data_matrix):
	print("Distance metric used: Euclidean")
	# Calculate euclidean distance matrix
	distance_matrix =  squareform(pdist(data_matrix, metric="euclidean"))
	similarity_matrix = calc_distance_based_similarity(distance_matrix)
	return similarity_matrix
    

def calc_normalized_euclidean_distance(data_matrix):
	print("Distance metric used: Euclidean")
	# Calculate euclidean distance matrix
	distance_matrix =  squareform(pdist(data_matrix, metric="euclidean"))
	similarity_matrix = calc_normalized_distance_based_similarity(distance_matrix)
	return similarity_matrix


def calc_distance_based_similarity(distance_matrix):
	#dbs - distance based similarity
    similarity_matrix = 1/(1+distance_matrix)

    # Calculate upper half of the mutual information matrix without diagonal
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    # Find similarity minimum and maximum
    print("Distance based Similarity Matrix Minimum: " + str(np.amin(upper_triangle)))
    print("Distance based Similarity Matrix Maximum: " + str(np.amax(upper_triangle)))

    return similarity_matrix
        
def calc_normalized_distance_based_similarity(distance_matrix):
	#ndbs - normalized distance based similarity
    max_value = np.amax(distance_matrix)
    similarity_matrix = 1/(1+(distance_matrix/max_value))

    # Calculate upper half of the mutual information matrix without diagonal
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    # Find similarity minimum and maximum
    print("Normalized Distance based Similarity Matrix  Minimum: " + str(np.amin(upper_triangle)))
    print("Normalized  Distance based Similarity Matrix Maximum: " + str(np.amax(upper_triangle)))

    return similarity_matrix


def calc_cosine_similarity(data_matrix):
	# Calculate cosine similarity matrix
	cosim_matrix = cosim(data_matrix)

	# Calculate upper half of the cosim matrix without diagonal
	upper_triangle = cosim_matrix[np.triu_indices_from(cosim_matrix, k=1)]
	# Find matrix minimum and maximum greater than zero
	print("Cosim Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Cosim Matrix Maximum: " + str(np.amax(upper_triangle)))

	return cosim_matrix


# Mutual Information Similarity Matrix
def calc_mutual_information(data_matrix):
	# Empty matrix for mutual information scores
	mutual_info_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))

	# Calculation of normalized mutual information score
	for idx_x, value_x in enumerate(data_matrix):
		for idx_y, value_y in enumerate(data_matrix[idx_x+1:], start=idx_x+1):
			mutual_info_score = normalized_mutual_info_score(value_x, value_y)
			mutual_info_matrix[idx_x][idx_y] = mutual_info_score
			mutual_info_matrix[idx_y][idx_x] = mutual_info_score

	# Calculate upper half of the mutual information matrix without diagonal
	upper_triangle = mutual_info_matrix[np.triu_indices_from(mutual_info_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Mutual Information Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Mutual Information Matrix Maximum: " + str(np.amax(upper_triangle)))

	return mutual_info_matrix


# Jaccard Index Similarity Matrix
def calc_jaccard_score(data_matrix_binary, data_matrix):
	# Empty matrix for jaccard similarity scores
	jaccard_matrix = np.zeros((data_matrix_binary.shape[0], data_matrix_binary.shape[0]))

	# Calculation of jaccard similarity score
	for idx_x, value_x in enumerate(data_matrix_binary):
		for idx_y, value_y in enumerate(data_matrix_binary[idx_x+1:], start=idx_x+1):
			# If data matrix for weighting is given, calculate weighted jaccard similarity score
			if data_matrix is not None:
				jaccard_score = jaccard_similarity_score(value_x, value_y,
				                                         sample_weight=(data_matrix[idx_x]+data_matrix[idx_y])/2)
			# If data matrix is not given, calculate unweighted jaccard similarity score
			else:
				jaccard_score = jaccard_similarity_score(value_x, value_y)
			jaccard_matrix[idx_x][idx_y] = jaccard_score
			jaccard_matrix[idx_y][idx_x] = jaccard_score

	# Calculate upper half of the jaccard similarity score matrix without diagonal
	upper_triangle = jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Jaccard Similarity Score Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Jaccard Similarity Score Matrix Maximum: " + str(np.amax(upper_triangle)))

	return jaccard_matrix


def calc_partial_correlation_coefficient(pearson_similarity_matrix):
	if np.linalg.matrix_rank(pearson_similarity_matrix) == pearson_similarity_matrix.shape[0]:
		inverse_matrix = sp.linalg.inv(pearson_similarity_matrix)
	else:
		inverse_matrix = sp.linalg.pinv2(pearson_similarity_matrix)

	partial_coerr_matrix = np.zeros((len(inverse_matrix),len(inverse_matrix)))

	for idx_i, _ in enumerate(partial_coerr_matrix):
		for idx_j, _ in enumerate(partial_coerr_matrix[idx_i]):
			partial_coerr_matrix[idx_i,idx_j] = -inverse_matrix[idx_i,idx_j]/np.sqrt(inverse_matrix[idx_i,idx_i] * inverse_matrix[idx_j,idx_j])

	return partial_coerr_matrix



def calc_distance_correlation(data_matrix):
	n = data_matrix.shape[1]
	dCor_matrix = np.zeros((n,n))

	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			a = squareform(pdist(sample_a[:, None]))
			b = squareform(pdist(sample_b[:, None]))
			A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
			B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
			dCov_ab = (A * B).sum() / n ** 2
			dVar_a = (A * A).sum() / n ** 2
			dVar_b = (B * B).sum() / n ** 2
			dCor = sqrt(dCov_ab) / sqrt(sqrt(dVar_a) * sqrt(dVar_b))
			dCor_matrix[idx_a, idx_b] = dCor

	return dCor_matrix



def contingency(img1, img2):
	t = max(img1.max(), img2.max()) * 0.7
	u = max(img1.max(), img2.max()) * 0.1
	c = np.zeros((3,3))
	for i in range(len(img1)):
		if img1[i] >= t and img2[i] >= t:
			c[0,0] += 1
		elif img1[i] >= t and u <= img2[i] < t:
			c[0,1] += 1
		elif u <= img1[i] < t and img2[i] >= t:
			c[1,0] += 1
		elif u <= img1[i] < t and u <= img2[i] < t:
			c[1,1] += 1
		elif img1[i] >= t and img2[i] < u:
			c[0,2] += 1
		elif u <= img1[i] < t and img2[i] < u:
			c[1,2] += 1
		elif img1[i] < u and img2[i] >= t:
			c[2,0] += 1
		elif img1[i] < u and u <= img2[i] < t:
			c[2,1] += 1
		elif img1[i] < u and img2[i] < u:
			c[2,2] += 1
		else:
			print("Something went wrong!")

	p = len(img1)
	s = (2*c[0,0] + 1.5*c[1,1] + c[2,2]) - (np.trace(c, offset=1) + np.trace(c, offset=-1)) - (2 * (np.trace(c, offset=2) + np.trace(c, offset=-2)))
	m = s/p
	return m