from sys import argv
import json
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import affinity_propagation, k_means
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import collections
import csv

def assign_mol_to_mz(communities, h5_data):
    # Assign molecule classes to m/z values within communities
    molclasses = collections.OrderedDict()
    mz_molclass_pairs = collections.OrderedDict()

    # Init empty list for molecule classes
    for com in set(communities):
        molclasses[com] = []
        mz_molclass_pairs[com] = []

    csvfile = open("mz-to-molclass.csv")
    readCSV = list(csv.reader(csvfile, delimiter=","))
    #print(communities)
    for idx, com in enumerate(communities):
        #for mz in list(h5_data.columns[com]):
        for entry in readCSV:
            #print(entry)
            f = False
            if str(h5_data.columns[idx]) == entry[0]:
                molclasses[com].append(entry[1])
                mz_molclass_pairs[com].append(entry)
                f = True
                break
    if f == False:
        raise ValueError("missing mz")
    
    #print(molclasses)
    #print(mz_molclass_pairs)
    for x in mz_molclass_pairs:
        #print(molclasses[x])
        for y in mz_molclass_pairs[x]:
            print(y)
        print()
        print()

#python C:\Users\kwuellems\Github\msi-community-detection\diss_eval\eval.py ..\..\..\..\barley101.h5 "correlation" .\output.json

# correlation for pearson
# cosine for cosine
def hierarchical(dmatrix, nr_cluster):
    cond_dmatrix = squareform(dmatrix)
    Z = linkage(cond_dmatrix, method="average", optimal_ordering=True)
    if nr_cluster == -1:
        mean_dd = np.mean(Z[:,2])
        std_dd = np.std(Z[:,2])
        C = 1
        labels = fcluster(Z, t=C*std_dd+mean_dd, criterion="distance") - 1
    else:    
        labels = fcluster(Z, t=nr_cluster, criterion="maxclust") - 1
    return labels

def eval(data, dmatrix, labels):
    ch = round(calinski_harabasz_score(data, labels), 5)
    db = round(davies_bouldin_score(data, labels), 5)
    ss = round(silhouette_score(dmatrix, labels, metric="precomputed"), 5)
    #print(collections.Counter(labels).values())
    #print(max(collections.Counter(labels).values()))
    size = sorted(collections.Counter(labels).values(), reverse=True)
    ch_adj = round(calinski_harabasz_score(data, labels), 5) / (size[0] / size[1])
    db_adj = round(davies_bouldin_score(data, labels), 5) * (size[0] / size[1])
    return ch, ch_adj, ss, db, db_adj, str(size[0]) + "/" + str(size[1]) + "=" + str(round(size[0]/size[1],3)), len(set(labels))

dframe = pd.read_hdf(argv[1])
data = dframe.T.values

with open(argv[3]) as f:
    d = json.load(f)

idx = []
mz = []
label_cd = []
for key, item in d["graphs"]["graph0"]["graph"]["hierarchy" + str(max([int(x.split("hierarchy")[1]) for x in d["graphs"]["graph0"]["graph"].keys()]))]["nodes"].items():
    idx.append(item["index"])
    mz.append(item["name"])
    label_cd.append(item["membership"])
    if len(item["mzs"]) > 1:
        raise ValueError("Error in json file!")

if label_cd == [x for _,x in sorted(zip(mz, label_cd))] == False:
    raise ValueError("oops?!")
#print(sorted(zip(mz, label_cd)))
label_cd = [x for _,x in sorted(zip(mz, label_cd))]

#print(label_cd)
assign_mol_to_mz(label_cd, dframe)

'''
#print("-----")
#print(len(idx))
#print(len(set(label_cd)))
dmatrix = squareform(pdist(data, metric=argv[2]))
#print(data.shape)
#print(dmatrix.shape)
#print("-----")
nr_cluster = len(set(label_cd))
label_hc1 = hierarchical(dmatrix, -1)
label_hc2 = hierarchical(dmatrix, nr_cluster)
_, label_aff = affinity_propagation(1-dmatrix)
_, label_kmeans, _ = k_means(data, nr_cluster, random_state=0)
#_, label_kmeans2 = kmeans2(data, k=nr_cluster)
#_, label_kmeans3 = kmeans2(data, k=nr_cluster, minit="++")

#print("-----")
#print(nr_cluster)
#print("-----")

print()
print()
print("**************************************************************************")
print("Community Detection (%i)         : "%(len(set(label_cd))), end="")
print(collections.OrderedDict(sorted(collections.Counter(label_cd).items())))
print("Hierarchical Clustering Auto (%i): "%(len(set(label_hc1))), end="")
print(collections.OrderedDict(sorted(collections.Counter(label_hc1).items())))
print("Hierarchical Clustering (%i)     : "%(len(set(label_hc2))), end="")
print(collections.OrderedDict(sorted(collections.Counter(label_hc2).items())))
print("kMeans (%i)                : "%(len(set(label_kmeans))), end="")
print(collections.OrderedDict(sorted(collections.Counter(label_kmeans).items())))
print("Affinity Propagation (%i)        : "%(len(set(label_aff))), end="")
print(collections.OrderedDict(sorted(collections.Counter(label_aff).items())))
print("--------------------------------------------------------------------------")

print("Calinski-Harabasz, Silhoutte-Score, Davies-Bouldin, Adjusted Calinski-Harabasz, Maximum Cluster Size, Number of Clusters \n")
print("\n       higher     ,      higher    ,     lower     ,           higher          ,         'lower'     ,          -         \n")

print("Community Detection (%i)         : "%(len(set(label_cd))), end="")
print(eval(data, dmatrix, label_cd))

print("Hierarchical Clustering Auto (%i): "%(len(set(label_hc1))), end="")
print(eval(data, dmatrix, label_hc1))
print("Hierarchical Clustering (%i)     : "%(len(set(label_hc2))), end="")
print(eval(data, dmatrix, label_hc2))

print("kMeans (%i)                : "%(len(set(label_kmeans))), end="")
print(eval(data, dmatrix, label_kmeans))
#print("kMeans scipy (%i)              : "%(len(set(label_kmeans2))), end="")
#print(eval(data, dmatrix, label_kmeans2))
#print("kMeans++ scipy (%i)              : "%(len(set(label_kmeans3))), end="")
#print(eval(data, dmatrix, label_kmeans3))

print("Affinity Propagation (%i)        : "%(len(set(label_aff))), end="")
print(eval(data, dmatrix, label_aff))

print("**************************************************************************")
print()
print()
'''