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
from kode.hypergeometric_similarity import *
from kode.multi_feature_similarity_variants import *
from kode.gradient_measures import *
from kode.local_standard_deviation_based_image_quality import *
from kode.mean_deviation_similarity_index import *
from kode.ssim_variants import *
from kode.shared_residual_similarity import *
from kode.contingency_similarity import *
import matplotlib.pyplot as plt

def calc_pearson_correlation(data_matrix):
	# Calculate pearson correlation matrix
	pearson_matrix = corrcoef(data_matrix)

	# Calculate upper half of pearson matrix without diagonal
	upper_triangle = pearson_matrix[np.triu_indices_from(pearson_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Pearson Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Pearson Matrix Maximum: " + str(np.amax(upper_triangle)))

	return pearson_matrix

def calc_spearman_correlation(data_matrix):
	# Calculate pearson correlation matrix
	spearman_matrix = sp.stats.spearmanr(data_matrix, axis=1)[1]

	# Calculate upper half of pearson matrix without diagonal
	upper_triangle = spearman_matrix[np.triu_indices_from(spearman_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Pearson Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Pearson Matrix Maximum: " + str(np.amax(upper_triangle)))

	return spearman_matrix



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


def calc_partial_correlation_coefficient(data_matrix):
	pearson_similarity_matrix = calc_pearson_correlation(data_matrix)
	if np.linalg.matrix_rank(pearson_similarity_matrix) == pearson_similarity_matrix.shape[0]:
		inverse_matrix = sp.linalg.inv(pearson_similarity_matrix)
	else:
		inverse_matrix = sp.linalg.pinv2(pearson_similarity_matrix)

	partial_coerr_matrix = np.zeros((len(inverse_matrix),len(inverse_matrix)))

	for idx_i, _ in enumerate(partial_coerr_matrix):
		for idx_j, _ in enumerate(partial_coerr_matrix[idx_i]):
			partial_coerr_matrix[idx_i,idx_j] = -inverse_matrix[idx_i,idx_j]/np.sqrt(inverse_matrix[idx_i,idx_i] * inverse_matrix[idx_j,idx_j])

	upper_triangle = partial_coerr_matrix[np.triu_indices_from(partial_coerr_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))

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
	upper_triangle = dCor_matrix[np.triu_indices_from(dCor_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return dCor_matrix



def create_index_mask(dframe, x_pixel_identifier="grid_x", y_pixel_identifier="grid_y"):
	x = (dframe.index.get_level_values(x_pixel_identifier)).astype(int)
	y = (dframe.index.get_level_values(y_pixel_identifier)).astype(int)
	img = np.zeros((y.max() + 1, x.max() + 1))
	img[(y,x)] = 1
	indices = np.where(img==1)
	#plt.imshow(img)
	#plt.show()
	return indices, img

def create_img(dframe, intens):
	grid_x = np.array(dframe.index.get_level_values("grid_x")).astype(int)
	grid_y = np.array(dframe.index.get_level_values("grid_y")).astype(int)
	height = grid_y.max() + 1
	width = grid_x.max() + 1
	img = np.zeros((height, width))
	img[(grid_y, grid_x)] = np.array(intens)
	return img

def calc_multifeature(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _ = multi_feature_similarity(X, Y, index_mask, weighted=False, wplus=False, pooling="max", win_size=13, sigma=None)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix

def calc_intmagan(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _ = int_mag_an(X, Y, index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix
	

def calc_hypergeometric(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score = hypergeometric_similarity(X, Y, index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix
	

def calc_local_std_similarity(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, lsdbiq_map = lsdbiq(X, Y, win_size=13, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from scipy.stats.mstats import winsorize
import skimage.filters as skif
import pandas as pd
import phik

def prepro(X, index_mask):
	X = winsorize(X, limits=[0, 0.01])
	t = skif.threshold_otsu(X[index_mask], nbins=256)
	X[X < t] = 0
	X = median_filter(X, size=3)
	return X

def prepro2(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	grid_x = np.array(dframe.index.get_level_values("grid_x")).astype(int)
	grid_y = np.array(dframe.index.get_level_values("grid_y")).astype(int)
	for idx, sample in enumerate(data_matrix):
		X = create_img(dframe, sample)
		X = winsorize(X, limits=[0, 0.01])
		t = skif.threshold_otsu(X[index_mask], nbins=256)
		X[X < t] = 0
		X = median_filter(X, size=5)
		data_matrix[idx] = X[(grid_y, grid_x)]
	return data_matrix

def calc_phik(dframe):
	sim_matrix = dframe.phik_matrix().values
		# Calculate upper half of pearson matrix without diagonal
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))

	return sim_matrix


def calc_phik_adj(dframe):
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	sim_matrix = dframe.phik_matrix().values
		# Calculate upper half of pearson matrix without diagonal
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	# Find similarity minimum and maximum
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))

	return sim_matrix


def calc_pearson_adj(dframe):
	data_matrix = prepro2(dframe)
	pearson_matrix = corrcoef(data_matrix)
	upper_triangle = pearson_matrix[np.triu_indices_from(pearson_matrix, k=1)]
	print("Pearson Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Pearson Matrix Maximum: " + str(np.amax(upper_triangle)))
	return pearson_matrix
	'''
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			Xp = prepro(X, index_mask)
			Yp = prepro(Y, index_mask)
			Xp = Xp[index_mask]
			Yp = Yp[index_mask]
	pearsonr(Xp, Yp)[0]
	return?
	'''

def calc_cosine_adj(dframe):
	data_matrix = prepro2(dframe)
	cosim_matrix = cosim(data_matrix)
	upper_triangle = cosim_matrix[np.triu_indices_from(cosim_matrix, k=1)]
	print("Cosim Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Cosim Matrix Maximum: " + str(np.amax(upper_triangle)))
	return cosim_matrix
	'''
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			Xp = prepro(X, index_mask)
			Yp = prepro(Y, index_mask)
			Xp = Xp[index_mask]
			Yp = Yp[index_mask]
	cosim(Xp[None,:], Yp[None,:])[0][0]
	return ? '''

def calc_hypergeometric_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score = hypergeometric_similarity(X, Y, index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix


def calc_local_std_similarity_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, lsdbiq_map = lsdbiq(X, Y, win_size=13, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix

def calc_contingency(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score = contingency(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_contingency_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score = contingency(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_mdsi(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			_, score, _ = mdsi(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_mdsi_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			_, score, _ = mdsi(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix

def calc_mdsi2(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _, _ = mdsi(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_mdsi2_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _, _ = mdsi(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix


	

def calc_ssim(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score= compare_ssim4(X, Y, index_mask=index_mask, win_size=13)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_ssim_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _ = compare_ssim(X, Y, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix


def calc_sr(dframe):
	data_matrix = dframe.values.transpose()
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _, _, _, _, _, _  = shared_residual_similarity(X, Y, count_zeros=False, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix



def calc_sr_adj(dframe):
	#data_matrix = dframe.values.transpose()
	data_matrix = prepro2(dframe)
	dframe = pd.DataFrame(data_matrix.T, index=dframe.index, columns=dframe.columns)
	index_mask,_ = create_index_mask(dframe)
	sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
	for idx_a, sample_a in enumerate(data_matrix):
		for idx_b, sample_b in enumerate(data_matrix):
			X = create_img(dframe, sample_a)
			Y = create_img(dframe, sample_b)
			score, _, _, _, _, _, _ = shared_residual_similarity(X, Y, count_zeros=False, index_mask=index_mask)
			sim_matrix[idx_a][idx_b] = score
	upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
	print("Matrix Minimum: " + str(np.amin(upper_triangle)))
	print("Matrix Maximum: " + str(np.amax(upper_triangle)))
	return sim_matrix