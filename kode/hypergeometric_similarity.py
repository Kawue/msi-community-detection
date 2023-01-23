import numpy as np
from kode.image_operators import binarize

# Reference: Hypergeometric Similarity Measure for Spatial Analysis in Tissue Imaging Mass Spectrometry
# Adjustments made since there
def hypergeometric_similarity(X, Y, index_mask):
    
    def one_sided(X, Y, index_mask):
        N = len(index_mask[0])
        Xb = binarize(X, index_mask)
        nX = np.where(Xb == 1)[0].size
        Yb = binarize(Y, index_mask)
        nY = np.where(Yb == 1)[0].size
        k = np.where((Xb+Yb) == 2)[0].size
        w = 2 # "width" of value distribution, see opriginal paper
        
        pA = (N-nY) / N
        tA = ((nX-k) / nX) - pA
        pB = nY / N
        tB = (k/nX) - pB

        if tA < 0:
            tA = 0 
        if tB < 0:
            tB = 0
        
        if 1-pA-tA > 0 and pA+tA > 0:
            A = ((pA/(pA+tA))**(pA+tA) * ((1-pA)/(1-pA-tA))**(1-pA-tA))**w
        else:
            A = np.inf
        if 1-pB-tB > 0 and pB+tB > 0:
            B = ((pB/(pB+tB))**(pB+tB) * ((1-pB)/(1-pB-tB))**(1-pB-tB))**w
        else:
            B = np.inf
            
        sim = A-B
        if sim == np.inf:
            sim = -1
        if sim == -np.inf:
            sim = 1

        return sim

    return (one_sided(X,Y,index_mask) + one_sided(Y,X,index_mask)) / 2