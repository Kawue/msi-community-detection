import numpy as np
import pandas as pd
import os

def grine_dimreduce(dframe, embedding, name, method, savepath):
    data = np.array([
        np.concatenate([embedding[:,0]]),
        np.concatenate([embedding[:,1]]),
        np.concatenate([embedding[:,2]])
        ]).T
    gx = list(dframe.index.get_level_values("grid_x"))
    gy = list(dframe.index.get_level_values("grid_y"))
    dn = [name]*len(list(dframe.index.get_level_values("grid_y")))
    idx = pd.MultiIndex.from_tuples(zip(gx,gy,dn), names=["grid_x","grid_y","dataset"])
    dframe = pd.DataFrame(data, index = idx, columns=["R", "G", "B"])
    dframe.to_hdf(os.path.join(savepath, "dimreduce_%s_%s"%(name, method)), key="dimreduce_%s_%s"%(name, method), complib="blosc", complevel=9)