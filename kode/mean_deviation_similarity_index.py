import numpy as np
from kode.gradient_helper import gradient_map, magnitude_map
from kode.pooling import deviation_pooling
from scipy.ndimage import uniform_filter, gaussian_filter


# Source: "Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image Quality Evaluator"
# Adjustments made.
# ATTENTION: dev_sim is actually a distance measure and not a similarity measure, look at the return statement for more details!
def mdsi(X, Y, index_mask, emphase=None, log=True, return_map=True):
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    # ndimage filters need floating point data
    X = X.astype(np.float)
    Y = Y.astype(np.float)

    F = 0.5 * (X + Y)

    #C1, C2, C3, a = (140, 55, 550, 0.6) # Original paper parameter
    C1, C2, C3, a = (0.0001, 0.0001, 0.0001, 0.5)

    # Gradient measure
    Xdy, Xdx = gradient_map(X)
    GX = magnitude_map(Xdy, Xdx)
    Ydy, Ydx = gradient_map(Y)
    GY = magnitude_map(Ydy, Ydx)
    Fdy, Fdx = gradient_map(F)
    GF = magnitude_map(Fdy, Fdx)	

    GS_XY = (2 * GX * GY + C1) / (GX**2 + GY**2 + C1)
    GS_XF = (2 * GX * GF + C2) / (GX**2 + GF**2 + C2)
    GS_YF = (2 * GY * GF + C2) / (GY**2 + GF**2 + C2)

    # Weak added/removed edges are likely to smooth out in F. Therefore the subtraction terms put less emphasis on weak edges.
    if emphase is None:
        # Without emphase GS is bound between [0, 1]
        GS = GS_XY
    else:
        # With emphase GS is bound between ]-1, +2[
        if (emphase == X).all():
            # Emphases removed edges from X over added edges to Y
            GS = GS_XY + (GS_YF - GS_XF)
        elif (emphase == Y).all():
            # Emphases removed edges from Y over added edges to X
            GS = GS_XY + (GS_XF - GS_YF)
        else:
            raise ValueError("Emphase must be None or equal one of the input images.")

    # TODO Think about including luminance and contrast of SSIM here
    # Intensity measure
    IS = (2 * X * Y + C3) / (X**2 + Y**2 + C3)

    # Combine gradient and intensity measure
    # a bound in [0,1]
    # GCS < 0: highly distorted pixels.
    # GCS > (1-e): less/non-distorted pixels, where e<1 is a very small number.
    GCS = a * GS + (1-a) * IS

    p, q = 1, 0.25
    moc = np.mean(GCS[index_mask]**q)
    mean_sim = moc

    if log:
        # smaller values indicate more severe distorted images, while perfect similarity equals inf.
        #sim = -np.log(deviation_pooling(GCS, moc, weight=None, p=p, q=q))
        dev_sim = -np.log(deviation_pooling(GCS[index_mask], moc, weight=None))
    else:
        # Larger values indicate more severe distorted images, while perfect similarity equals zero.
        #sim = deviation_pooling(GCS, moc, weight=None, p=p, q=q)
        dev_sim = deviation_pooling(GCS[index_mask], moc, weight=None, p=p,q=q)

    if return_map:
        return dev_sim, mean_sim, GCS
    else:
        return dev_sim, mean_sim