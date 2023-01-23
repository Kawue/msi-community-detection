import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util import crop
from skimage import filters
from skimage.feature import canny
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn
from kode.gradient_helper import gradient_map, magnitude_map


# Reference: Wang et al "Image Quality Assessment: From Error Visibility to Structural Similarity"
def compare_ssim(X, Y, index_mask, L=1, win_size=None, gaussian_weights=True, sigma=1.5, sample_covariance=False):
    if not X.shape == Y.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    if L not in [1,255]:
        raise ValueError("L has to be 1 or 255 depending on the image encoding.")

    # Set constants for numerical instabilities
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    if win_size:
        if win_size%2 == 0:
            print(win_size)
            raise ValueError("Window size has to be odd.")

    # Prepare either a gaussian filter or uniform filter and their respective needed argument
    if gaussian_weights:
        if win_size and sigma:
            raise ValueError("With gaussian weights choose either a window size or sigma. The other value will be calculated automatically.")
        if win_size:
            sigma = 0.25 * ((0.5*win_size) - 0.5)
        if sigma:
            win_size = 2 * int(4*sigma+0.5) + 1
        filter_func = gaussian_filter
        filter_args = {"sigma": sigma}
    else:
        if sigma:
            raise ValueError("Without gaussian weights sigma needs to be None, since it is not needed.")
        filter_func = uniform_filter
        filter_args = {"size": win_size}

    if sample_covariance:
        ndim = X.ndim
        NP = win_size**ndim
        cov_norm = NP / (NP - 1)
    else:
        cov_norm = 1.0

    # Compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # Compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    # Compute ssim based on Wang et al
    ssim_map = ((2*ux*uy+C1)*(2*vxy+C2)) / ((ux**2+uy**2+C1)*(vx+vy+C2))

    # Compute (mean) pooling SSIM except for the border region to avoid border effects.
    pad = (win_size-1) // 2
    if index_mask:
        ssim_score = np.mean(ssim_map[index_mask])
    else:
        ssim_score = np.mean(crop(ssim_map, pad))
    
    return ssim_score, ssim_map




# Reference: "GRADIENT-BASED STRUCTURAL SIMILARITY FOR IMAGE QUALITY ASSESSMENT"
def compare_gssim(X, Y, index_mask, L=1, win_size=None, gaussian_weights=True, sigma=1.5, sample_covariance=False):
    if not X.shape == Y.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    if L not in [1,255]:
        raise ValueError("L has to be 1 or 255 depending on the image encoding.")

    # Set constants for numerical instabilities
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    if win_size:
        if win_size%2 == 0:
            raise ValueError("Window size has to be odd.")

    # Prepare either a gaussian filter or uniform filter and their respective needed argument
    if gaussian_weights:
        if win_size and sigma:
            raise ValueError("With gaussian weights choose either a window size or sigma. The other value will be calculated automatically.")
        if win_size:
            sigma = 0.25 * ((0.5*win_size) - 0.5)
        if sigma:
            win_size = int(2 * (4*sigma+0.5) + 1)
        filter_func = gaussian_filter
        filter_args = {"sigma": sigma}
    else:
        if sigma:
            raise ValueError("Without gaussian weights sigma needs to be None, since it is not needed.")
        filter_func = uniform_filter
        filter_args = {"size": win_size}

    # Compute gradient and gradient magnitude images
    Xdy, Xdx = filters.sobel_h(X), filters.sobel_v(X)
    Ydy, Ydx = filters.sobel_h(Y), filters.sobel_v(Y)
    Xm = np.sqrt(Xdy**2 + Xdx**2)
    Ym = np.sqrt(Ydy**2 + Ydx**2)

    if sample_covariance:
        ndim = X.ndim
        NP = win_size**ndim
        cov_norm = NP / (NP - 1)
    else:
        cov_norm = 1.0

    # Compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)
    uxG = filter_func(Xm, **filter_args)
    uyG = filter_func(Ym, **filter_args)

    # Compute (weighted) variances and covariances
    uxxG = filter_func(Xm * Xm, **filter_args)
    uyyG = filter_func(Ym * Ym, **filter_args)
    uxyG = filter_func(Xm * Ym, **filter_args)
    vxG = cov_norm * (uxxG - uxG * uxG)
    vyG = cov_norm * (uyyG - uyG * uyG)
    vxyG = cov_norm * (uxyG - uxG * uyG)

    # Compute gssim
    gssim_map = ((2*ux*uy+C1)*(2*vxyG+C2)) / ((ux**2+uy**2+C1)*(vxG+vyG+C2))

    # Compute (mean) pooling gssim except for the border region to avoid border effects.
    pad = (win_size-1) // 2
    if index_mask:
        gssim_score = np.mean(gssim_map[index_mask])
    else:
        gssim_score = np.mean(crop(gssim_map, pad))

    return gssim_score, gssim_map









# Reference: Content-partitioned structural similarity index for image quality assessment
# Formulation of regions deviates from the reference.
# The reference has soften rules for the degraded image, this does not apply in this application scenario.
# TODO: Allow weights as parameters
def compare_ssim4(img1, img2, index_mask, win_size=None, gaussian_weights=True, sigma=None, ssim_map=None, return_maps=False):
    # Calcuate gradient magnitude maps of both images
    dy1, dx1 = gradient_map(img1)
    mag1 = magnitude_map(dy1,dx1)

    dy2, dx2 = gradient_map(img2)
    mag2 = magnitude_map(dy2,dx2)

    # thresholds are determined by maximum gradient magnitude value
    gmax1 = np.amax(mag1)
    te1 = 0.12*gmax1
    ts1 = 0.06*gmax1

    gmax2 = np.amax(mag2)
    te2 = 0.12*gmax2
    ts2 = 0.06*gmax2

    #w_ep, w_ec, w_s, w_t = 0.25, 0.25, 0.25, 0.25
    w_ep, w_ec, w_s, w_t = 1, 1, 1, 1

    idx_ep = (mag1 > te1) * (mag2 > te2)
    ep_map, ec_map, s_map, t_map = ( np.zeros(img1.shape), np.zeros(img1.shape), np.zeros(img1.shape), np.zeros(img1.shape) )
    ep_map[idx_ep] = w_ep

    idx_ec = (mag1 > te1) * (mag2 <= te2) + ((mag1 <= te1 ) * (mag2 > te2))
    ec_map[idx_ec] = w_ec

    idx_s = (mag1 < ts1) * (mag2 < ts2)
    s_map[idx_s] = w_s

    idx_t = (((idx_ep + idx_ec + idx_s) * -1) + 1).astype(bool)
    t_map[idx_t] = w_t
    
    if ssim_map is None:
        _, ssim_map = compare_ssim(img1, img2, index_mask=index_mask, win_size=win_size, gaussian_weights=gaussian_weights, sigma=sigma)

    ssim_ep = ssim_map * ep_map
    ssim_ec = ssim_map * ec_map
    ssim_s = ssim_map * s_map
    ssim_t = ssim_map * t_map

    ssim_final = ssim_ep + ssim_ec + ssim_s + ssim_t

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # TODO: may apply more useful pooling, although this can be done seperately with the returned ssim _final map
    if index_mask:
        ssim4_mean = np.mean(ssim_final[index_mask])
    else:
        ssim4_mean = np.mean(crop(ssim_final, pad))
    # Region maps for preserved edge, changed edge, smooth, textured regions and the final combination.
    if return_maps:
        return ssim4_mean, ssim_final, ssim_ep, ssim_ec, ssim_s, ssim_t
    else:
        return ssim4_mean



# TODO: Allow weights as parameters
def compare_gssim4(img1, img2, index_mask, win_size=None, gaussian_weights=True, sigma=None, ssim_map=None, return_maps=True):
    if ssim_map is None:
        _, ssim_map = compare_gssim(img1, img2, index_mask=index_mask, win_size=win_size, gaussian_weights=gaussian_weights, sigma=sigma)
    return compare_ssim4(img1, img2, ssim_map=ssim_map, index_mask=index_mask, win_size=win_size, return_maps=return_maps)



def compare_gmsd(img1, img2, index_mask=False, return_map=True):
    dy, dx = gradient_map(img1)
    m1 = magnitude_map(dy,dx)
    dy, dx = gradient_map(img2)
    m2 = magnitude_map(dy,dx)

    c = 0.0026
    score_map = (2*m1*m2 + c) / (m1**2 + m2**2 + c)
    if index_mask:
        score = np.std(score_map[index_mask])
    else:
        score = np.std(score_map)

    if return_map:
        return score, score_map
    else:
        return score



# Has some round because of machine delta numerical instabilities
def compare_stsim(X, Y, index_mask, L=1, win_size=None, gaussian_weights=True, sigma=1.5, sample_covariance=False):
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if L not in [1,255]:
        raise ValueError("L has to be 1 or 255 depending on the image encoding.")

    K1 = 0.01
    K2 = 0.03

    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")

    # Prepare either a gaussian filter or uniform filter and their respective needed argument
    # win_size=11 to match Wang et. al. 2004
    if gaussian_weights:
        if win_size and sigma:
            raise ValueError("With gaussian weights choose either a window size or sigma. The other value will be calculated automatically.")
        if win_size:
            sigma = 0.25 * ((0.5*win_size) - 0.5)
        if sigma:
            win_size = 2 * int(4*sigma+0.5) + 1
        filter_func = gaussian_filter
        filter_args = {"sigma": sigma}
    else:
        if sigma:
            raise ValueError("Without gaussian weights sigma needs to be None, since it is not needed.")
        filter_func = uniform_filter
        filter_args = {"size": win_size}

    if sigma < 0:
        raise ValueError("sigma must be positive")
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    
    if X.dtype != Y.dtype:
        warn("Inputs have mismatched dtype.")
    dmin, dmax = dtype_range[X.dtype.type]
    data_range = dmax - dmin
    ndim = X.ndim

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    Xdy, Xdx = filters.sobel_h(X), filters.sobel_v(X)
    Ydy, Ydx = filters.sobel_h(Y), filters.sobel_v(Y)
    Xmag = np.sqrt(Xdy**2 + Xdx**2)
    Ymag = np.sqrt(Ydy**2 + Ydx**2)

    # filter are already normalized by NP
    if sample_covariance:
        ndim = X.ndim
        NP = win_size**ndim
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    vx = np.around(cov_norm * (uxx - ux * ux), 5)
    vy = np.around(cov_norm * (uyy - uy * uy), 5)
    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    sx[np.isnan(sx)] = 0
    sy[np.isnan(sy)] = 0

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * sx * sy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    
    L = A1/B1
    C = A2/B2

    shiftX_01 = np.roll(X, shift=-1, axis=1)
    shiftY_01 = np.roll(Y, shift=-1, axis=1)
    shiftX_10 = np.roll(X, shift=-1, axis=0)
    shiftY_10 = np.roll(Y, shift=-1, axis=0)
    
    p = 1

    autocovX_01 = np.around(uniform_filter( (X - ux) * (shiftX_01 - ux), size=win_size), 5)
    autocovY_01 = np.around(uniform_filter( (Y - uy) * (shiftY_01 - uy), size=win_size), 5)
    px_01 = np.divide(autocovX_01, vx, out=np.zeros_like(autocovX_01), where=vx!=0)
    py_01 = np.divide(autocovY_01, vy, out=np.zeros_like(autocovY_01), where=vy!=0)
    C_01 = 1 - (0.5 * np.abs(px_01 - py_01)**p)

    autocovX_10 = np.around(uniform_filter( (X - ux) * (shiftX_10 - ux), size=win_size), 5)
    autocovY_10 = np.around(uniform_filter( (Y - uy) * (shiftY_10 - uy), size=win_size), 5)
    px_10 = np.divide(autocovX_10, vx, out=np.zeros_like(autocovX_10), where=vx!=0)
    py_10 = np.divide(autocovY_10, vy, out=np.zeros_like(autocovY_10), where=vy!=0)
    C_10 = 1 - (0.5 * np.abs(px_10 - py_10)**p)


    # This approach is different from the original reference.
    # The original reference uses L**a ... because there should be no negative value in any map.
    # In this case C_01 and C_10 had negative values, leading to the given change.
    a,b,c,d = 0.25, 0.25, 0.25, 0.25
    stsim_map = a*L + b*C + c*C_01 + d*C_10

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2
    # compute (weighted) mean of ssim
    if index_mask:
        stsim_score = np.mean(stsim_map[index_mask])
    else:
        stsim_score = np.mean(crop(stsim_map, pad))

    return stsim_score, stsim_map

