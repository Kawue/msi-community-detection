import numpy as np
from kode.gradient_helper import gradient_map, magnitude_map, dot_product_map


# Position, Magnitude and Angular measure combined
# Basis Paper: Similarity Measure for Vector Field Learning
# Own adaption: Including pixel intensity
def int_mag_an(img1, img2, index_mask, a=(1/3), b=(1/3), pxint=True, c=(1/3), mark_homogeneous=1):
    if pxint:
        if a + b + c != 1:
                raise ValueError("a, b and c must sum to one!")
    else:
        if c and c > 0:
            raise ValueError("If pxint is False factor c will be ignored!")
        if a + b != 1:
            raise ValueError("a and b must sum to one!")
    
    dy1, dx1 = gradient_map(img1) 
    gmag1 = magnitude_map(dy1, dx1)
    
    dy2, dx2 = gradient_map(img2)
    gmag2 = magnitude_map(dy2, dx2)
    
    # Angular measure
    vectormap1 = np.dstack((dx1,dy1))
    vectormap2 = np.dstack((dx2,dy2))
    ang = dot_product_map(vectormap1, vectormap2, mark_homogeneous=mark_homogeneous)
    hr_map1 = (vectormap1[:,:,0] == 0) * (vectormap1[:,:,1] == 0)
    hr_map2 = (vectormap2[:,:,0] == 0) * (vectormap2[:,:,1] == 0)
    idx_hr12 = np.where(hr_map1 * hr_map2)
    ang[idx_hr12] = 1

    # Constant to avoid division by 0
    const = 0.00001

    # Magnitude measure
    mag = (2*gmag1*gmag2 + const) / (gmag1**2 + gmag2**2 + const)

    # Not sclaing ang and use multiplication like ang * (a*mag + c*pxmag) would have the interesting effect that a negative ang makes the whole term negative.
    if pxint:
        pxmag = (2*img1*img2 + const) / (img1**2 + img2**2 + const)
        #return a*np.e**(1-((ang+1)/2)) + b*np.e**(-mag) + c*np.e**(-pxmag)
        # Scale dot product in [0,1] to have the same value ranges for all measures.
        sim_map = a*((ang+1)/2) + b*mag + c*pxmag
    else:
        #return a*np.e**(1-((ang+1)/2)) + b*np.e**(-mag)
        sim_map = a*((ang+1)/2) + b*mag
    score = np.mean(sim_map[index_mask])
    return score, sim_map

