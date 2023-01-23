import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from kode.gradient_helper import gradient_map, magnitude_map, direction_map
from kode.shared_residual_similarity import shared_residual_similarity

def multi_feature_similarity(X, Y, index_mask, weighted, wplus, pooling="max", gaussian_weights=True, sigma=1.5, sample_covariance=False, win_size=None):
    if pooling not in ["max", "mean"]:
        raise ValueError("pooling must be 'max' or 'mean'.")

    # Constant to avoid division by zero
    c = 0.0001
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
        cov_norm = 1

    # Calculate similarity based on intensity image
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)
    uxx = filter_func(X*X, **filter_args)
    uyy = filter_func(Y*Y, **filter_args)
    uxy = filter_func(X*Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    # Avoid numerical instabilities
    vx[vx<0] = 0
    vy[vy<0] = 0
    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    vxy = cov_norm * (uxy - ux * uy)
    I = (2*ux*uy+c) / (ux**2+uy**2+c)
    Iv = (2*sx*sy+c) / (vx+vy+c)
    Icv = (vxy+c) / (sx*sy+c)

    # Calculate ssim based on magnitude image
    Xdy, Xdx = gradient_map(X)
    Ydy, Ydx = gradient_map(Y)
    Xm = magnitude_map(Xdy, Xdx)
    Ym = magnitude_map(Ydy, Ydx)
    uxm = filter_func(Xm, **filter_args)
    uym = filter_func(Ym, **filter_args)
    uxxm = filter_func(Xm*Xm, **filter_args)
    uyym = filter_func(Ym*Ym, **filter_args)
    uxym = filter_func(Xm*Ym, **filter_args)
    vxm = cov_norm * (uxxm - uxm * uxm)
    vym = cov_norm * (uyym - uym * uym)
    # Avoid numerical instabilities
    vxm[vxm<0] = 0
    vym[vym<0] = 0
    sxm = np.sqrt(vxm)
    sym = np.sqrt(vym)
    vxym = cov_norm * (uxym - uxm * uym)      
    M = (2*uxm*uym+c) / (uxm**2+uym**2+c)
    Mv = (2*sxm*sym+c) / (vxm+vym+c)
    Mcv = (vxym+c) / (sxm*sym+c)

    # Calculate ssim based on orientation image
    Xo = direction_map(Xdy, Xdx)+180
    Yo = direction_map(Ydy, Ydx)+180
    uxo = filter_func(Xo, **filter_args)
    uyo = filter_func(Yo, **filter_args)
    uxxo = filter_func(Xo*Xo, **filter_args)
    uyyo = filter_func(Yo*Yo, **filter_args)
    uxyo = filter_func(Xo*Yo, **filter_args)
    vxo = cov_norm * (uxxo - uxo * uxo)
    vyo = cov_norm * (uyyo - uyo * uyo)
    # Avoid numerical instabilities
    vxo[vxo<0] = 0
    vyo[vyo<0] = 0
    sxo = np.sqrt(vxo)
    syo = np.sqrt(vyo)
    vxyo = cov_norm * (uxyo - uxo * uyo) 
    O = (2*uxo*uyo+c) / (uxo**2+uyo**2+c)
    Ov = (2*sxo*syo+c) / (vxo+vyo+c)
    Ocv = (vxyo+c) / (sxo*syo+c)
        
    # Calculate shared information map
    if weighted:
        if wplus:  
            _, _, _, w, _, _, _ = shared_residual_similarity(X, Y, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wm, _, _, _ = shared_residual_similarity(Xm, Ym, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wo, _, _, _ = shared_residual_similarity(Xo, Yo, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            w = filter_func(w, **filter_args)
            wm = filter_func(wm, **filter_args)
            wo = filter_func(wo, **filter_args)
        else:
            _, _, _, w, _, _, _ = shared_residual_similarity(X, Y, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wm, _, _, _ = shared_residual_similarity(Xm, Ym, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wo, _, _, _ = shared_residual_similarity(Xo, Yo, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            w = filter_func(w, **filter_args)
            wm = filter_func(wm, **filter_args)
            wo = filter_func(wo, **filter_args)
    else:
        if wplus:
            raise ValueError("Weighted plus (wplus) can only be used with weighted.")


    Itot = I*Iv*Icv
    Mtot = M*Mv*Mcv
    Otot = O*Ov*Ocv
    Itot = np.abs(Itot)**(1/3) * np.sign(Itot)
    Mtot = np.abs(Mtot)**(1/3) * np.sign(Mtot)
    Otot = np.abs(Otot)**(1/3) * np.sign(Otot)

    if weighted:
        if pooling == "max":
            score_map = np.amax([Itot, Mtot, Otot], axis=0)
            weight_map = np.amax([w,wm,wo], axis=0)
            sim_map = (score_map*weight_map)/np.sum(weight_map[index_mask])
            if (weight_map == 0).all():
                score = 0
            else:
                score = np.sum((score_map*weight_map)[index_mask]) / np.sum(weight_map[index_mask])
        elif pooling == "mean":
            if (w == 0).all():
                Issim_score = 0
            else:
                Issim_score = np.sum((Itot*w)[index_mask])/np.sum(w[index_mask])
            if (wm == 0).all():
                Mssim_score = 0
            else:
                Mssim_score = np.sum((Mtot*wm)[index_mask])/np.sum(wm[index_mask])
            if (wo == 0).all():
                Ossim_score = 0
            else:
                Ossim_score = np.sum((Otot*wo)[index_mask])/np.sum(wo[index_mask])
            sim_map = np.mean([Itot*w, Mtot*wm, Otot*wo])
            score = np.mean([Issim_score, Mssim_score, Ossim_score])
    else:
        if pooling == "max":
            score_map = np.amax([Itot, Mtot, Otot], axis=0)
            sim_map = score_map
            score = np.mean(score_map[index_mask])
        elif pooling == "mean":
            Issim_score = np.mean(Itot[index_mask])
            Mssim_score = np.mean(Mtot[index_mask])
            Ossim_score = np.mean(Otot[index_mask])
            sim_map = np.mean([Itot, Mtot, Otot])
            score = np.mean([Issim_score, Mssim_score, Ossim_score])

    return score, sim_map





def multi_feature_ssim(X, Y, index_mask, weighted=False, wplus=False, gaussian_weights=True, sigma=1.5, sample_covariance=False, win_size=None):
    # Constant to avoid division by zero
    c = 0.0001
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
        cov_norm = 1

    # Calculate ssim based on intensity image
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)
    uxx = filter_func(X*X, **filter_args)
    uyy = filter_func(Y*Y, **filter_args)
    uxy = filter_func(X*Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    ssI = ((2*ux*uy+c)*(2*vxy+c)) / ((ux**2+uy**2+c)*(vx+vy+c))
        
    # Calculate ssim based on magnitude image
    Xdy, Xdx = gradient_map(X)
    Ydy, Ydx = gradient_map(Y)
    Xm = magnitude_map(Xdy, Xdx)
    Ym = magnitude_map(Ydy, Ydx)
    uxm = filter_func(Xm, **filter_args)
    uym = filter_func(Ym, **filter_args)
    uxxm = filter_func(Xm*Xm, **filter_args)
    uyym = filter_func(Ym*Ym, **filter_args)
    uxym = filter_func(Xm*Ym, **filter_args)
    vxm = cov_norm * (uxxm - uxm * uxm)
    vym = cov_norm * (uyym - uym * uym)
    vxym = cov_norm * (uxym - uxm * uym)
    ssM = ((2*uxm*uym+c)*(2*vxym+c)) / ((uxm**2+uym**2+c)*(vxm+vym+c))
        
    # Calculate ssim based on orientation image
    Xo = direction_map(Xdy, Xdx)+180
    Yo = direction_map(Ydy, Ydx)+180
    uxo = filter_func(Xo, **filter_args)
    uyo = filter_func(Yo, **filter_args)
    uxxo = filter_func(Xo*Xo, **filter_args)
    uyyo = filter_func(Yo*Yo, **filter_args)
    uxyo = filter_func(Xo*Yo, **filter_args)
    vxo = cov_norm * (uxxo - uxo * uxo)
    vyo = cov_norm * (uyyo - uyo * uyo)
    vxyo = cov_norm * (uxyo - uxo * uyo)
    ssO = ((2*uxo*uyo+c)*(2*vxyo+c)) / ((uxo**2+uyo**2+c)*(vxo+vyo+c))

    # Calculate shared information map
    if weighted:
        if wplus:  
            _, _, _, w, _, _, _ = shared_residual_similarity(X,Y, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wm, _, _, _ = shared_residual_similarity(Xm,Ym, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wo, _, _, _ = shared_residual_similarity(Xo,Yo, count_zeros=False, smoothing= False, win_size=False, binarize=False, process=False)
            w = filter_func(w, **filter_args)
            wm = filter_func(wm, **filter_args)
            wo = filter_func(wo, **filter_args)
        else:
            _, _, _, w ,_, _, _ = shared_residual_similarity(X,Y, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wm ,_, _, _ = shared_residual_similarity(Xm,Ym, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            _, _, _, wo ,_, _, _ = shared_residual_similarity(Xo,Yo, count_zeros=True, smoothing= False, win_size=False, binarize=False, process=False)
            w = filter_func(w, **filter_args)
            wm = filter_func(wm, **filter_args)
            wo = filter_func(wo, **filter_args)
    else:
        if wplus:
            raise ValueError("Weighted plus (wplus) can only be used with weighted.")
        w = 1
        wm = 1
        wo = 1

    ssim_map = np.mean([ssI*w, ssM*wm, ssO*wo], axis=0)

    score = np.mean(ssim_map[index_mask])

    return score, ssim_map



def multi_mean_absolute_correlation(X, Y, index_mask, gaussian_weights=True, sigma=1.5, sample_covariance=False, win_size=None):
    # Constant to avoid division by zero
    c = 0.0001
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
        cov_norm = 1

    Imadcorr = (filter_func((np.abs(X+Y) - np.abs(X-Y)), **filter_args) + c) / (filter_func((np.abs(X) + np.abs(Y)), **filter_args) + c)

    # Calculate ssim based on magnitude image
    Xdy, Xdx = gradient_map(X)
    Ydy, Ydx = gradient_map(Y)
    Xm = magnitude_map(Xdy, Xdx)
    Ym = magnitude_map(Ydy, Ydx)
    Mmadcorr = (filter_func(np.abs(Xm+Ym) - np.abs(Xm-Ym), **filter_args) + c) / (filter_func(np.abs(Xm) + np.abs(Ym), **filter_args) + c)

    # Calculate ssim based on orientation image
    Xo = direction_map(Xdy, Xdx)+180
    Yo = direction_map(Ydy, Ydx)+180
    Omadcorr = (filter_func(np.abs(Xo+Yo) - np.abs(Xo-Yo), **filter_args) + c) / (filter_func(np.abs(Xo) + np.abs(Yo), **filter_args) + c)

    a,b,c = 1/3,1/3,1/3
    sim_map = (a*Imadcorr + b*Mmadcorr + c*Omadcorr)

    score = np.mean(sim_map[index_mask])

    return score, sim_map