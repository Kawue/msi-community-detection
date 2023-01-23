from kode.image_operators import coincidence_image, residual_image, binarize
from skimage.morphology import opening, closing, square
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


# Calculates the shared and residual "information", normalizes with the entire "information" and subtracts both values.
# Value range [-1,1]
def shared_residual_similarity(X, Y, count_zeros, index_mask=None, smoothing=False, win_size=False, binarize=False, process=False):
    if type(index_mask) == type(None):
        index_mask = np.where(np.ones_like(X) == 1)
    
    if smoothing not in [False, "gaussian", "uniform"]:
        raise ValueError("smoothing must be False, 'gaussian' or 'uniform'.")

    if not smoothing:
        if binarize:
            X = binarize(X)
            Y = binarize(Y)
        if process:
            X = opening(closing(X, square(3)), square(3))
            Y = opening(closing(Y, square(3)), square(3))
    else:
        if binarize or process:
            raise ValueError("Smoothing cannot be combined with binarization and processing mode.")

    I = coincidence_image(X,Y)
    R = residual_image(X,Y)

    # P is a distance measure within [0,1], the higher the more dissimilar.
    P = R[index_mask].sum()/(X[index_mask].sum()+Y[index_mask].sum())
    # Q is a similarity measure within [0,1], the higher the more similar.
    Q = 2*I[index_mask].sum()/(X[index_mask].sum()+Y[index_mask].sum())
    # S is also a similarity measure, but within [-1,1]
    # S is basically equal to normalizing Q into [-1,1] or P into 1 - [-1,1]
    S = Q-P

    if smoothing == "uniform":
        if isinstance(win_size, int):
            if win_size%2 == 0:
                raise ValueError("Window size must be odd!")
            filter_func = uniform_filter
            filter_args = {"size": win_size}
        else:
            raise ValueError("For uniform smoothing win_size must be an integer, which provides the window size.")
    elif smoothing == "gaussian":
        if isinstance(win_size, float):
            filter_func = gaussian_filter
            filter_args = {"sigma": win_size}
        else:
            raise ValueError("For gaussian smoothing win_size must be a float, which describes the sigma of the gaussian kernel. The approximated window size is: 2 * int(4*sigma+0.5) + 1.")

    if count_zeros:
        c_map = (2*I+0.0001)/(X+Y+0.0001)
        r_map = (R+0.0001)/(X+Y+0.0001)
    else:
        c_map = (2*I)/(X+Y)
        c_map[np.isnan(c_map)] = 0
        r_map = (R)/(X+Y)
        r_map[np.isnan(r_map)] = 0

    if win_size:
        c_map = filter_func(c_map, **filter_args)
        r_map = filter_func(r_map, **filter_args)

    return S, P, Q, c_map, r_map, I, R