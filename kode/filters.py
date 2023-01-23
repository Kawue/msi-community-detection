import numpy as np
from scipy.ndimage.filters import uniform_filter, gaussian_filter

def std_filter(img, win_size, gaussian_weights=False):
    if win_size%2 == 0:
        raise ValueError("Window size needs to be odd.")

    if not gaussian_weights:
        m = uniform_filter(img, win_size, mode="reflect")
        ms = uniform_filter(img*img, win_size, mode="reflect")
        std = np.sqrt(ms - (m*m))
    elif gaussian_weights:
        s = 0.25 * ((0.5*win_size) - 0.5)
        m = gaussian_filter(img, sigma=s)
        ms = gaussian_filter(img*img, sigma=s)
        std = np.sqrt(ms - (m*m))

    if np.isinf(std).any():
        raise ValueError("Infinity Problem in std_filter.")

    np.nan_to_num(std, copy=False)
    if not np.isfinite(std).any():
        raise ValueError("std_filter includes non finite values!")

    return std