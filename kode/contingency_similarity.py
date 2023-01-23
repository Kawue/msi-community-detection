import numpy as np
from sklearn.metrics.cluster import contingency_matrix

def contingency(X, Y, index_mask, tl=0.2, tu=0.8):
    if not 0 <= tl <=1:
        raise ValueError("Lower threshold has to be between 0 and 1.")
    if not 0 <= tu <=1:
        raise ValueError("Upper threshold has to be between 0 and 1.")

    adjust = X.size-len(index_mask[0])

    l = max(X.max(), Y.max()) * tl
    u = max(X.max(), Y.max()) * tu

    Xlbl = X.copy()
    Xlbl[Xlbl<l] = -1
    Xlbl[(l<=Xlbl)&(Xlbl<=u)] = -2
    Xlbl[u<Xlbl] = -3
    Xlbl = Xlbl*-1

    Ylbl = Y.copy()
    Ylbl[Ylbl<l] = -1
    Ylbl[(l<=Ylbl)&(Ylbl<=u)] = -2
    Ylbl[u<Ylbl] = -3
    Ylbl = Ylbl*-1

    cm = contingency_matrix(Xlbl.flatten(), Ylbl.flatten(), sparse=False)
    cm[0,0] = cm[0,0] - adjust

    if not (Xlbl == 3).any():
        cm = np.vstack((cm, [0,0,0]))
    elif not (Ylbl == 3).any():
        cm = np.hstack((cm, [[0],[0],[0]]))
    elif not (Xlbl == 1).any():
        cm = np.vstack(([0,0,0],cm))
    elif not (Ylbl == 1).any():
        cm = np.hstack(([[0],[0],[0]], cm))
    elif not (Xlbl == 2).any():
        cm = np.vstack((cm[0,:], [0,0,0], cm[1,:]))
    elif not (Ylbl == 2).any():
        cm = np.hstack((cm[:, 0, None], [[0],[0],[0]], cm[:, 1,None]))
    
    N = cm.sum()

    a,b,c = 1,1,1
    s = (a * (cm[0,0] + cm[1,1] + cm[2,2])) - (b * (np.trace(cm, offset=1) + np.trace(cm, offset=-1))) - (c * (np.trace(cm, offset=2) + np.trace(cm, offset=-2)))
    m = s/(a*N)
    
    return m