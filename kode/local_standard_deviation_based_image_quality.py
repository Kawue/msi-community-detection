import numpy as np
from kode.filters import std_filter

# Reference: Full reference image quality metrics for JPEG compressed images
# For Std Pooling: Lower values of lsdbiq indicate better similarity.
# For Mean Pooling: Higher values of lsdbiq indicate better similarity.
def lsdbiq(img1, img2, win_size, return_map=True, index_mask=None):
    img1, img2 = img1.copy(), img2.copy()

    c = 0.00001
    s1 = std_filter(img1, win_size)
    s2 = std_filter(img2, win_size)
    lsm = ((2*s1*s2) + c) / (s1**2 + s2**2 + c)

    # Pooling
    # Original reference uses std.
    # Meaning that very dissimilar images (universally low values) and very similar images (universally high values), both, show good results.
    # For simplicity this is changed to mean.
    if index_mask:
        #lsdbiq_score = np.std(lsm[index_mask])
        lsdbiq_score = np.mean(lsm[index_mask])
    else:
        #lsdbiq_score = np.std(lsm)
        lsdbiq_score = np.mean(lsm)

    if return_map:
        return lsdbiq_score, lsm
    else:
        return lsdbiq_score