import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import approximation as nxapprox
import scipy.stats as stats
import fim
from itertools import chain
import argparse
import os
import json
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from time import time
import multiprocessing as mp
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import correlation, cosine


class Graph:
    def __init__(self, adjacency_matrix, similarity_matrix, node_labels=None, labeltag=None, graphs=None, cluster_labels=None, calc_feature_vectors=True):
        self.adjacency_matrix = adjacency_matrix
        self.similarity_matrix = similarity_matrix
        self.graph = nx.from_numpy_matrix(adjacency_matrix)
        self.connected_graph = self.force_connected(self.graph, adjacency_matrix, similarity_matrix)
        self.filled_graph = None
        self.community_list = None
        self.node_labels = None
        if node_labels is not None:
            if labeltag is None:
                self.labeltag = "generic"
            else:
                self.labeltag = labeltag
            self.node_labels = node_labels
            self.label_graph(self.graph, node_labels, labeltag)
        if graphs is not None:
            self.force_mastergraph(graphs)            
        if cluster_labels:
            if not (isinstance(cluster_labels, list) or isinstance(cluster_labels, dict) or isinstance(cluster_labels, np.ndarray)):
                raise ValueError("cluster_labels must be of type list, dict or ndarray. For list or ndarray provide: [[identifier, label], ...]. For dict provide {identifier: label, ...}")
            if isinstance(cluster_labels, dict):
                cluster_label_tuples = [[i,j] for i,j in cluster_labels.items()]
                cluster_label_list = [x[1] for x in cluster_label_tuples]
                self.get_community_list(cluster_label_list)
                self.cluster_labels = cluster_labels
            elif isinstance(cluster_labels, list) or isinstance(cluster_labels, np.ndarray):
                self.get_community_list(cluster_labels)
                cluster_label_list = [x[1] for x in cluster_labels]
                self.cluster_labels = {tup[0]: tup[1] for tup in cluster_labels}
        if calc_feature_vectors:
            if cluster_labels is None:
                raise ValueError("cluster_labels must be provided to calculate feature vectors.")
            self.get_feature_vector(self.graph, cluster_label_list, force_connected=False)
            self.get_feature_vector(self.connected_graph, cluster_label_list, force_connected=True)


    def label_graph(self, graph, node_labels, labeltag):
        attrs = {}
        for idx, label in enumerate(node_labels):
            attrs[idx] = {labeltag: label}
        nx.set_node_attributes(graph, attrs)

    
    def force_connected(self, graph, adj_matrix, sim_matrix):
        while len(list(nx.connected_components(graph))) > 1:
            components = list([np.array(list(c)) for c in nx.connected_components(graph)])
            candidate_edges = []
            for i, c1 in enumerate(components):
                for j, c2 in enumerate(components[i+1:], start=i+1):
                    try:
                        bool_matrix = np.zeros_like(sim_matrix)
                        bool_matrix[np.ix_(c1,c2)] = 1
                        bool_matrix[bool_matrix == 0] = -np.inf
                        candidate_edges.append(np.unravel_index(np.argmax(sim_matrix * bool_matrix), sim_matrix.shape))
                    except Exception as e:
                        print(e)
                        print(i,j)
            edge_to_add = candidate_edges[np.argmax([sim_matrix[e[0], e[1]] for e in candidate_edges])]
            graph.add_edge(edge_to_add[0], edge_to_add[1], weight=sim_matrix[edge_to_add[0], edge_to_add[1]])
        
        if len(list(nx.connected_components(graph))) > 1:
            raise ValueError("Bug in force_connected. Connection unsuccessful!")
        
        return graph


    def force_mastergraph(self, graphs, labeltag=None, labels=None):
        if type(graphs) != list:
            raise ValueError("Type of graphs in force_mastergraph() must be of type list.")
        
        if labeltag is None and labels is None:
            raise ValueError("For labeld graphs a labeltag must be provided. For unlabeld graphs a list of labels must be provided (labeltag is optional).")
        if labels is not None and labeltag is None:
            labeltag = "generic"
        
        if labels is not None:
            unique_labels = sorted(list(set([l for lbls in labels for l in lbls])))
        if labeltag is not None:
            unique_labels = sorted(list(set([l for g in graphs for l in nx.get_node_attributes(g.graph, labeltag).values()])))

        own_labels = nx.get_node_attributes(self.graph, labeltag)
        if len(own_labels) == 0:
            raise ValueError("Graph has no labels that match the given labeltag.")

        matrix = np.array(nx.adjacency_matrix(self.graph).todense())

        for idx, lbl in enumerate(unique_labels):
            if lbl not in own_labels:
                arr = np.zeros(matrix.shape[0])
                matrix = np.insert(matrix, idx, arr, axis=0)
                arr = np.zeros(matrix.shape[0])
                matrix = np.insert(matrix, idx, arr, axis=1)
        
        G = nx.from_numpy_matrix(matrix)
        self.label_graph(G, unique_labels, labeltag)
        self.filled_graph = G
        return G


    def get_community_list(self, memb_tuples):
        community_list = []
        for i in range (0, max([x[1] for x in memb_tuples])+1):
            community_list.append([x[0] for x in memb_tuples if x[1] == i])
        self.community_list = community_list
        return community_list


    def get_feature_vector(self, g, memberships, force_connected):
        nr_singletons = nx.algorithms.isolate.number_of_isolates(g)
        nr_nodes = len(g.nodes)
        nr_edges = len(g.edges)
        nr_communities = np.amax(memberships)
        degree_assortativity_coefficient = nx.algorithms.assortativity.degree_assortativity_coefficient(g) #weight="weight"
        estrada_index = nx.algorithms.centrality.estrada_index(g)
        transistivity = nx.algorithms.cluster.transitivity(g)
        average_clustering_coefficient = nx.algorithms.cluster.average_clustering(g) #weight="weight"
        average_node_connectivity = nx.algorithms.connectivity.connectivity.average_node_connectivity(g) # Time consuming
        #local_efficiency = nx.algorithms.efficiency_measures.local_efficiency(g) # <- which for edge reduction
        global_efficiency = nx.algorithms.efficiency_measures.global_efficiency(g) #  <- which for edge reduction
        overall_reciprocity = nx.overall_reciprocity(g)
        s_metric = nx.algorithms.smetric.s_metric(g, normalized=False)

        if force_connected:
            if len(list(nx.connected_components(g))) > 1:
                raise ValueError("The provided Graph is not connected. Call force_connected() or provide a connected Grph.")
            average_shortest_path_length = nx.algorithms.shortest_paths.generic.average_shortest_path_length(g) #weight="weight" # Problems if not connected
            diameter = nx.algorithms.distance_measures.diameter(g) # Problems if not connected 
            radius = nx.algorithms.distance_measures.radius(g) # Problems if not connected
            #sw_sigma = nx.algorithms.smallworld.sigma(g, seed=0, niter=100, nrand=10) # Problems if not connected #(default=100)niter=number of rewiring per edge, (default=10)nrand=number of random graphs # Time consuming
            #sw_omega= nx.algorithms.smallworld.omega(g, seed=0, niter=100, nrand=10) # Problems if not connected #(default=100)niter=number of rewiring per edge, (default=10)nrand=number of random graphs # Time consuming
            wiener_index = nx.algorithms.wiener.wiener_index(g) # Problems if not connected #weight="weight"
            feature_vector = [
                nr_singletons,
                nr_nodes,
                nr_edges,
                nr_communities,
                degree_assortativity_coefficient,
                estrada_index,
                transistivity,
                average_clustering_coefficient,
                average_node_connectivity,
                #local_efficiency,
                global_efficiency,
                overall_reciprocity,
                s_metric,
                average_shortest_path_length,
                diameter,
                radius,
                #sw_sigma,
                #sw_omega,
                wiener_index
            ]
            self.feature_vector_connected = feature_vector
        else:
            feature_vector = [
                nr_singletons,
                nr_nodes,
                nr_edges,
                nr_communities,
                degree_assortativity_coefficient,
                estrada_index,
                transistivity,
                average_clustering_coefficient,
                average_node_connectivity,
                #local_efficiency,
                global_efficiency,
                overall_reciprocity,
                s_metric
            ]
            self.feature_vector = feature_vector

        #node_connectivity = nxapprox.connectivity.node_connectivity(g) #useless?
        #edge_connectivity = nx.algorithms.connectivity.connectivity.edge_connectivity(g) #useless?

        #print(feature_vector)
        
        return feature_vector
        
    
    

    



class GraphComparison:
    def __init__(self):
        pass
    
    def get_adjacency_matrix(self, g):
        return nx.linalg.graphmatrix.adjacency_matrix(g).toarray()

    def get_adjacency_spectrum(self, g):
        return nx.linalg.spectrum.adjacency_spectrum(g)


    def get_laplacian_matrix(self, g, normalized=False, signless=False):
        if normalized:
            if signless:
                return np.abs(nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g).toarray())
            else:
                return nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g).toarray()
        else:
            if signless:
                return np.abs(nx.linalg.laplacianmatrix.laplacian_matrix(g).toarray())
            else:
                return nx.linalg.laplacianmatrix.laplacian_matrix(g).toarray()
    
    def get_laplacian_spectrum(self, g, normalized=False):
        if normalized:
            return nx.linalg.spectrum.normalized_laplacian_spectrum(g)
        else:
            return nx.linalg.spectrum.laplacian_spectrum(g)

    
    def get_modularity_matrix(self, g):
        return nx.linalg.modularitymatrix.modularity_matrix(g).toarray()

    def get_modularity_spectrum(self, g):
        return nx.linalg.spectrum.modularity_spectrum(g)

    
    def get_degree_matrix(self, g):
        n = max(nx.nodes(g))
        degree_matrix = np.zeros((n,n))
        degree_matrix[np.diag_indices(n,2)] = list(zip(*list(nx.degree(g))))[1]
        return degree_matrix


    # https://www.sciencedirect.com/science/article/pii/S0031320308000927
    def edit_distance(self, g1, g2, graph="adjacency", node_match=None, edge_match=None, node_subst_cost=None, node_del_cost=None, node_ins_cost=None, edge_subst_cost=None, edge_del_cost=None, edge_ins_cost=None, upper_bound=None):
        if graph not in ["adjacency", "laplacian", "modularity"]:
            raise ValueError("Graph parameter must be 'adjacency', 'laplacian' or 'modularity'.")
            
        if graph == "laplacian":
            g1 = nx.Graph(self.get_laplacian_matrix(g1))
            g2 = nx.Graph(self.get_laplacian_matrix(g2))

        if graph == "modularity":
            g1 = nx.Graph(self.get_modularity_matrix(g1))
            g2 = nx.Graph(self.get_modularity_matrix(g2))
        
        distance = nx.algorithms.similarity.graph_edit_distance(g1,g2, node_match=node_match, edge_match=edge_match, node_subst_cost=node_subst_cost, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost, edge_subst_cost=edge_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost, upper_bound=10)

        return distance


    # https://www.sciencedirect.com/science/article/pii/S0024379511006021
    def spectral_distance(self, g1, g2, variant="adjacency", fill=False, t=None):
        if variant not in ["adjacency", "laplacian", "modularity"]:
            raise ValueError("Variant parameter must be 'adjacency', 'laplacian' or 'modularity'.")

        if variant == "laplacian":
            s1 = self.get_laplacian_spectrum(g1)
            s2 = self.get_laplacian_spectrum(g2)
        elif variant == "modularity":
            s1 = self.get_modularity_spectrum(g1)
            s2 = self.get_modularity_spectrum(g2)
        else:
            s1 = self.get_adjacency_spectrum(g1)
            s2 = self.get_adjacency_spectrum(g2)

        
        if s1.size > s2.size:
            print()
            print("G1 Bigger!")
            print()
            if fill:
                zeros = [0] * (s1.size - s2.size)
                s2 = np.concatenate([s2, zeros])
            else:
                s1 = s1[:s2.size]
        elif s1.size < s2.size:
            print()
            print("G2 Bigger!")
            print()
            if fill:
                zeros = [0] * (s2.size - s1.size)
                s1 = np.concatenate([s1, zeros])
            else:
                s2 = s2[:s1.size]
        
        if t is not None:             
            #https://stackoverflow.com/questions/12122021/python-implementation-of-a-graph-similarity-grading-algorithm
            def select_k(spectrum, minimum_energy=t):
                running_total = 0.0
                total = sum(spectrum)
                if total == 0.0:
                    return len(spectrum)
                for i in range(len(spectrum)):
                    running_total += spectrum[i]
                    if running_total / total >= minimum_energy:
                        return i + 1
                return len(spectrum)

            k1 = select_k(s1)
            k2 = select_k(s2)
            k = min(k1, k2)
            #distance = sum((laplacian1[:k] - laplacian2[:k])**2)
            distance = np.sum(np.abs(s1[:k] - s2[:k]))
        else:
            distance = np.sum(np.abs(s1 - s2))

        return distance


    def feature_distance(self, feature_vec1, feature_vec2, func=stats.pearsonr):
        return func(feature_vec1, feature_vec2)
        

    # Sum of overlap size distance.
    def freq_set_mining(self, community_list_1, community_list_2):
        tracts = []
        for cl in [community_list_1, community_list_2]:
            for c in cl:
                tracts.append(c)
        fip = fim.fpgrowth(tracts, "m", -2)
        # size of overlap weighted by number of clusters in which this overlap occurs
        distance = np.sum([len(tup[0]) * tup[1] for tup in fip]) / np.array([community_list_1, community_list_2]).size
        return distance




    def print_community_overlaps(self, community_list_1, community_list_2):
        mz_interception = sorted(list(set(chain(*community_list_1)) & set(chain(*community_list_2))))
        intercept_matrix = np.zeros((len(community_list_1), len(community_list_2)))
        symdiff_matrix = np.zeros((len(community_list_1), len(community_list_2)))
        union_matrix = np.zeros((len(community_list_1), len(community_list_2)))
        
        for i, c1 in enumerate(community_list_1):
            for j, c2 in enumerate(community_list_2):
                union = len([m for m in (set(c1) | set(c2)) if m in mz_interception])
                intercept = len([m for m in (set(c1) & set(c2)) if m in mz_interception])
                symdiff = len([m for m in (set(c1) ^ set(c2)) if m in mz_interception])
                union_matrix[(i,j)] = union
                intercept_matrix[(i,j)] = intercept
                symdiff_matrix[(i,j)] = symdiff

        pct_overlap_matrix = np.round((intercept_matrix / union_matrix) * 100, 2)
        pct_overlap_matrix[np.isnan(pct_overlap_matrix)] = 0

        R = pd.Index(range(len(community_list_1)), name="Community 1")
        C = pd.Index(range(len(community_list_2)), name="Community 2")
        df_intercept = pd.DataFrame(intercept_matrix, index=R, columns=C)
        df_symdiff = pd.DataFrame(symdiff_matrix, index=R, columns=C)
        df_union = pd.DataFrame(union_matrix, index=R, columns=C)
        #print("High Values are Good, Low Values are Bad, Zero is Good.")
        #print("High Values along an Axis show that most of the Mass of one Community is one Community of the other Sample.")
        #print("Low values along an Axis indicated that the Mass of this Community is distributed over many Communities in the other Sample.")
        df_pct_overlap = pd.DataFrame(pct_overlap_matrix, index=R, columns=C)

        #print("Union")
        #print(df_union)
        #print("")
        #print("Interception")
        #print(df_intercept)
        #print("")
        #print("Symmetric Difference")
        #print(df_symdiff)
        #print("")
        #print("Percentual Overlap")
        #print(df_pct_overlap)

        # Zero is ok and has to be ignored.
        non_zero = np.where(intercept_matrix != 0)
        N = union_matrix[non_zero].size
        D = union_matrix[non_zero] - intercept_matrix[non_zero]
        G = union_matrix[non_zero]
        distance = np.sum(1-(D/G)) / N

        return distance




    
    def minimum_subgraph_distance(self, g1, g2):
        def get_max_common_subgraph(g1, g2):
            subgraph = nx.Graph()



        # minimum common supergraph
        supergraph = nx.compose(g1, g2)
        

        check stuff below
        #https://stackoverflow.com/questions/43108481/maximum-common-subgraph-in-a-directed-graph
        def getMCS(self, G_source, G_new):
            matching_graph=nx.Graph()

            for n1,n2,attr in G_new.edges(data=True):
                if G_source.has_edge(n1,n2) :
                    matching_graph.add_edge(n1,n2,weight=1)

            graphs = list(nx.connected_component_subgraphs(matching_graph))

            mcs_length = 0
            mcs_graph = nx.Graph()
            for i, graph in enumerate(graphs):

                if len(graph.nodes()) > mcs_length:
                    mcs_length = len(graph.nodes())
                    mcs_graph = graph

            return mcs_graph


    
    








def read_json(graphdict):
    mz_dict = graphdict["mzs"]
    memb_tuples = []
    for mz_str, hierarchies_dict in mz_dict.items():
        mz = float(mz_str) 
        tmp = [(x, int(x.split("hierarchy")[1])) for x in list(hierarchies_dict.keys())]
        tmp_sort = sorted(tmp, key=lambda tup: tup[1], reverse=True)
        memb = int(hierarchies_dict[tmp_sort[1][0]].split("n")[1]) # tmp_sort[1][0]] because the lowest clustered hierarchy is always hierarchy1
        memb_tuples.append((mz, memb))
    memb_tuples.sort(key=lambda tup: tup[0])
    memb_list = [x[1] for x in memb_tuples]
    mz_list = [float(x) for x in mz_dict.keys()]
    return mz_list, memb_tuples, memb_list

def calc_clustering(dmatrix, names, savepath, methodname):
    np.save(os.path.join(savepath, "distance-matrix-" + methodname + ".npy"), dmatrix)
    nr_cluster = 1
    Z = linkage(dmatrix, method="average", optimal_ordering=True)
    labels = fcluster(Z, t=nr_cluster, criterion="maxclust")
    plt.figure(figsize=(16, 10))
    dendro = dendrogram(Z, labels=names, orientation="right")
    plt.savefig(os.path.join(savepath, methodname + ".png"), dpi = 200, bbox_inches='tight')

def prepare_graph(dirpath, filename, json_dct):
    adjacency_matrix = np.load(os.path.join(dirpath, "adjacency-matrix-" + filename + ".npy"))
    similarity_matrix = np.load(os.path.join(dirpath, "similarity-matrix-" + filename + ".npy"))
    mz_list, memb_tuples, _ = read_json(json_dct[filename])
    node_labels = sorted(mz_list)
    return Graph(adjacency_matrix=adjacency_matrix, similarity_matrix=similarity_matrix, node_labels=node_labels, labeltag="mz", graphs=None, cluster_labels=memb_tuples, calc_feature_vectors=True)

def process(fct, name, g1, g2, f1, f2, kwargs):
    #template = "{method} between \n -{name1}- and -{name2}- \n -----> {distance} \n -------------------------------------------------------------------- \n"
    d = fct(g1, g2, **kwargs)
    #print(template.format(method=name, name1=f1, name2=f2, distance=d))
    return d



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dirpath", type=str, required=True, help="Path to folder.")
    parser.add_argument("-f", "--files", type=str, required=True, nargs="+", help="File names (similarity matrix, adjacency matrix and json graph names should have a shared file name.).")
    parser.add_argument("-a", "--alias", type=str, required=False, nargs="+", default=None, help="File alias names.")
    parser.add_argument("-j", "--jsons", type=str, required=True, nargs="+", help="File names of jsons (without file extension).")
    parser.add_argument("-s", "--savepath", type=str, required=True, help="Path to save output.")
    parser.add_argument("-c", "--cpucount", type=int, required=False, default=1, help="Number of CPUs for parallel processing. -1 takes all available CPUs.")
    args = parser.parse_args()

    print("")
    print("Start Graph Generation ..... ", end="")

    dirpath = args.dirpath
    filenames = args.files
    if args.alias:
        if len(args.alias) != len(args.files):
            raise ValueError("Number of filenames and alias must be equal!")
        filealias = args.alias
    else:
        filealias = args.files
    savepath = args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    jsons = args.jsons
    nr_cpu = args.cpucount

    json_dct = {}

    if nr_cpu == -1:
        pool = mp.Pool(mp.cpu_count)
    else:
        pool = mp.Pool(nr_cpu)


    for json_name in jsons:
        if ".json" not in json_name:
            json_name += ".json"
        with open(os.path.join(dirpath, json_name)) as f:
            json_file = json.load(f)
            for graph, dct in json_file["graphs"].items():
                json_dct[dct["dataset"]] = dct

    # Parallel preparation of graphs
    graphs = {}
    for f in filenames:
        graphs[f] = pool.apply_async(prepare_graph, args=(dirpath, f, json_dct,))

    for key, job in graphs.items():
        job.wait()
    
    for key, job in graphs.items():
        graphs[key] = job.get()

    print("successfully completed!")
    print("")

    # Create filled Graphs
    labeltag = "mz"
    graphs_list = list(graphs.values())
    for key, G in graphs.items():
        _ = G.force_mastergraph(graphs_list, labeltag)
        

    gc = GraphComparison()
    
    timestamppath = os.path.join(savepath)
    timetemplate = "Time for: {method} --> {time}"

    spectral_distance_adjacency = np.zeros((len(filenames),len(filenames)))
    spectral_distance_adjacency_fill = np.zeros((len(filenames),len(filenames)))
    spectral_distance_laplacian = np.zeros((len(filenames),len(filenames)))
    spectral_distance_laplacian_fill = np.zeros((len(filenames),len(filenames)))
    spectral_distance_mocularity = np.zeros((len(filenames),len(filenames)))
    spectral_distance_modularity_fill = np.zeros((len(filenames),len(filenames)))
    feature_distance_pearson = np.zeros((len(filenames),len(filenames)))
    feature_distance_cosine = np.zeros((len(filenames),len(filenames)))
    feature_distance_mutualinfo = np.zeros((len(filenames),len(filenames)))
    feature_distance_pearson_connected = np.zeros((len(filenames),len(filenames)))
    feature_distance_cosine_connected = np.zeros((len(filenames),len(filenames)))
    feature_distance_mutualinfo_connected = np.zeros((len(filenames),len(filenames)))
    feature_distance_fisscore = np.zeros((len(filenames),len(filenames)))
    feature_distance_overlap = np.zeros((len(filenames),len(filenames)))
    feature_distance_editdistance = np.zeros((len(filenames),len(filenames)))

    #start = time()
    #end = time()
    #with open(os.path.join(timestamppath, "computationTimes.txt"), "w+") as f:
    #    print(timetemplate.format(method="", time=np.around(end-start, 4)), file=f)

    print("Start Graph Comparison Computation ..... ")
    for i, f1 in enumerate(filenames):
        for j, f2 in enumerate(filenames[i+1:], start=i+1):
            G1 = graphs[f1]
            G2 = graphs[f2]

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Adjacency Distance Prune", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"adjacency", "fill":False})).get()
            spectral_distance_adjacency[i,j] = d
            spectral_distance_adjacency[j,i] = d

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Adjacency Distance Fill", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"adjacency", "fill":True})).get()
            spectral_distance_adjacency_fill[i,j] = d
            spectral_distance_adjacency_fill[j,i] = d        

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Laplacian Distance Prune", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"laplacian", "fill":False})).get()
            spectral_distance_laplacian[i,j] = d
            spectral_distance_laplacian[j,i] = d

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Laplacian Distance Fill", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"laplacian", "fill":True})).get()
            spectral_distance_adjacency_fill[i,j] = d
            spectral_distance_laplacian_fill[j,i] = d

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Modularity Distance Prune", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"modularity", "fill":False})).get()
            spectral_distance_mocularity[i,j] = d
            spectral_distance_mocularity[j,i] = d

            d = pool.apply_async(process, args=(gc.spectral_distance, "Spectral Modularity Distance Fill", G1.filled_graph, G2.filled_graph, f1, f2, {"spectrum":False, "variant":"modularity", "fill":True})).get()
            spectral_distance_mocularity[i,j] = d
            spectral_distance_mocularity[j,i] = d

        
            
            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance Pearson", G1.feature_vector, G2.feature_vector, f1, f2, {"func":correlation})).get()
            feature_distance_pearson[i,j] = d
            feature_distance_pearson[j,i] = d

            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance Cosine", G1.feature_vector, G2.feature_vector, f1, f2, {"func":cosine})).get()
            feature_distance_cosine[i,j] = d
            feature_distance_cosine[j,i] = d

            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance MI", G1.feature_vector, G2.feature_vector, f1, f2, {"func":mutual_info_score})).get()
            feature_distance_mutualinfo[i,j] = 1-d
            feature_distance_mutualinfo[j,i] = 1-d

            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance Pearson", G1.feature_vector_connected, G2.feature_vector_connected, f1, f2, {"func":correlation})).get()
            feature_distance_pearson_connected[i,j] = d
            feature_distance_pearson_connected[j,i] = d

            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance Cosine", G1.feature_vector_connected, G2.feature_vector_connected, f1, f2, {"func":cosine})).get()
            feature_distance_cosine_connected[i,j] = d
            feature_distance_cosine_connected[j,i] = d

            d = pool.apply_async(process, args=(gc.feature_distance, "Feature Vector Distance MI", G1.feature_vector_connected, G2.feature_vector_connected, f1, f2, {"func":mutual_info_score})).get()
            feature_distance_mutualinfo_connected[i,j] = 1-d
            feature_distance_mutualinfo_connected[j,i] = 1-d
            


            d = pool.apply_async(process, args=(gc.freq_set_mining, "Frequent Item Set Distance", G1.community_list, G2.community_list, f1, f2, {})).get()
            feature_distance_fisscore[i,j] = d
            feature_distance_fisscore[j,i] = d


            d = pool.apply_async(process, args=(gc.print_community_overlaps, "Matrix Overlap Distance", G1.community_list, G2.community_list, f1, f2, {})).get()
            feature_distance_overlap[i,j] = d
            feature_distance_overlap[j,i] = d

            


            d = pool.apply_async(process, args=(gc.edit_distance, "Edit Distance",  G1.graph, G2.graph, f1, f2, {})).get()
            feature_distance_editdistance[i,j] = d
            feature_distance_editdistance[j,i] = d


            print("")
            print("")
            print("#################")
            print("FIRST EDIT DONE!")
            print("#################")
            print("")
            print("")

            print("\n\n\n")

    pool.close()
    pool.join()

    calc_clustering(spectral_distance_adjacency, filealias, savepath, "spectral-distance-adjacency")
    calc_clustering(spectral_distance_adjacency_fill, filealias, savepath, "spectral-distance-adjacency-fill")
    calc_clustering(spectral_distance_laplacian, filealias, savepath, "spectral-distance-laplacian")
    calc_clustering(spectral_distance_laplacian_fill, filealias, savepath, "spectral-distance-laplacian-fill")
    calc_clustering(spectral_distance_mocularity, filealias, savepath, "spectral-distance-mocularity")
    calc_clustering(spectral_distance_modularity_fill, filealias, savepath, "spectral-distance-modularity-fill")
    calc_clustering(feature_distance_pearson, filealias, savepath, "feature-distance-pearson")
    calc_clustering(feature_distance_cosine, filealias, savepath, "feature-distance-cosine")
    calc_clustering(feature_distance_mutualinfo, filealias, savepath, "feature-distance-mutualinfo")
    calc_clustering(feature_distance_pearson_connected, filealias, savepath, "feature-distance-pearson-connected")
    calc_clustering(feature_distance_cosine_connected, filealias, savepath, "feature-distance-cosine-connected")
    calc_clustering(feature_distance_mutualinfo_connected, filealias, savepath, "feature-distance-mutualinfo-connected")
    calc_clustering(feature_distance_fisscore, filealias, savepath, "feature-distance-fisscore")
    calc_clustering(feature_distance_overlap, filealias, savepath, "feature-distance-overlap")
    calc_clustering(feature_distance_editdistance, filealias, savepath, "feature-distance-editdistance")
    























