from __future__ import print_function, division

import numpy as np
import scipy as sp

def leading_eigenvector_community(adjacency_matrix, number_boundary=None, ker_lin_flag=False, ker_lin_n_partite_flag=False, ew_limit=None):
    #https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.kernighan_lin.kernighan_lin_bisection.html#networkx.algorithms.community.kernighan_lin.kernighan_lin_bisection
    #https://networkx.github.io/documentation/stable/reference/linalg.html?highlight=modularity#module-networkx.linalg.modularitymatrix
    ##### Check room for optimization by using NetworkX

    if ew_limit == None:
        ew_limit = 0

    ker_lin_classic = ker_lin_flag
    ker_lin_n_partite = ker_lin_n_partite_flag

    # Tolerance to set very small values to zero
    tol = 0.0000000001


    # m: Total number of edges
    m = len([x for x in adjacency_matrix[np.triu_indices_from(adjacency_matrix, k=1)] if x != 0])

    #print("calc B")
    # B: Modularity Matrix
    B = np.zeros((len(adjacency_matrix), len(adjacency_matrix[0])))
    #print("B Done!")

    '''
    # Test Weighted Variant
    # Calculate network "entropy"
    m = np.sum([x for x in adjacency_matrix[np.triu_indices_from(adjacency_matrix, k=1)] if x != 0])

    for i in range(len(B)):
        for j in range(len(B)):
            # Calculate vertex strength
            k_i = np.sum([x for x in adjacency_matrix[i] if x != 0])
            k_j = np.sum([x for x in adjacency_matrix[j] if x != 0])
            B[i][j] = adjacency_matrix[i][j] - ((k_i*k_j)/(2*m))
    '''

    #print("calc Bij")
    # Build B_ij with A_ij - (deg_i*deg_j)/(2m), where A is the adjacency matrix and k is the degree of node i,j
    for i in range(len(B)):
        for j in range(len(B)):
            k_i = np.where(adjacency_matrix[i]!=0)[0].size
            k_j = np.where(adjacency_matrix[j]!=0)[0].size
            B[i,j] = adjacency_matrix[i,j] - ((k_i*k_j)/(2*m))
    #print("Bij Done!")

    # Set very small values to zero
    B[np.abs(B) < tol] = 0

    # Initialize index for all vertices
    vertex_indices = range(len(B))

    # Initialize variables
    cut = True
    groups = []
    communities = []
    connected_components = []

    # Detect connected components and set them as Communities
    cnct_cmp_labels = sp.sparse.csgraph.connected_components(adjacency_matrix, directed=False)
    for label in range(cnct_cmp_labels[0]):
        component = []
        for vertex in vertex_indices:
            if cnct_cmp_labels[1][vertex] == label:
                component.append(vertex)
        else:
            groups.append(component[:])
            connected_components.append(component[:])
    #print(vertex_indices)
    #print(groups)


    # Cut as long as a cut increases modularity
    while cut:
        for g in groups[:]:
            # If number of desired clusters is reached after last division break loop and stop calculation
            if number_boundary:
                if len(communities) + len(groups) >= number_boundary:
                    communities.extend(groups)
                    cut = False
                    break

            # Sort to avoid possible idx, v Problems in B_g, B if there are some.
            g.sort()
            # Remove "old" group from the list of groups
            groups.remove(g)

            # Is needed for singleton groups if an approximation is used that needs a number of eigenvalues that is smaller than dimension B_g but bigger than one
            #if len(g) < 2:
            #    communities.append(g[:])
            #    break

            # B_g: Modularity increase matrix delta B
            B_g = np.zeros((len(g), len(g)))
            #print(B_g.shape)
            # Build B_g_ij with B_ij and set B_ii to B_ii - (nodes connected with i and in group g)
            # -> Rows in B_g become 0
            for idx_i, v_i in enumerate(g):
                for idx_j, v_j in enumerate(g):
                    # Kronecker Delta
                    kron = 1 if idx_i == idx_j else 0
                    B_g[idx_i,idx_j] = B[v_i,v_j] - (kron * np.sum([x for i, x in enumerate(B[v_i]) if i in g]))
            # Set very small values to zero
            B_g[np.abs(B_g) < tol] = 0
            # Calculate eigenvalues and eigenvectors
            ew, ev = np.linalg.eig(B_g) #np.linalg.eig(B_g) #np.linalg.eigh(B_g) #sp.linalg.eig(B_g) #sp.linalg.eigh(B_g) #sp.sparse.linalg.eigsh(B_g, k=len(B_g)-1) #sp.sparse.linalg.eigs(B_g, k=len(B_g)-1)
            # Set very small values to zero
            ew[np.abs(ew) < tol] = 0
            # If the maximal eigenvalue is smaller equal zero the group is undivisible
            # because splitting would reduce modularity.
            # Add this group as community and break the loop
            if np.max(ew) <= ew_limit:
                communities.append(g[:])
                break
            # Get eigenvector to leading eigenvalue
            leading_ev = ev[:,np.argmax(ew)]
            # Initialize splitgroups
            g_a, g_b = [], []
            # Split vertices corresponding to the sign of their index in the leading eigenvector
            for i, v in enumerate(leading_ev):
                if v >= 0:
                    g_a.append(g[i])
                else:
                    g_b.append(g[i])
            # After loop use Kernighan-Lin algorithm to improve the result
            else:
                if ker_lin_classic:
                    improved = True
                else:
                    improved = False
                while improved:
                    swap_list = []
                    g_a, g_b, improved = kernighan_lin_classic(g_a[:], g_b[:], adjacency_matrix, swap_list)
                    #print("swapped")

            # Append splitted groups to the list of groups to be split
            groups = [g_a] + groups
            groups = [g_b] + groups
            #print(groups)

        # If all groups are indivisible stop splitting and return list of communities
        if not groups:
            cut = False
            break

    if ker_lin_n_partite:
        # If there exist multiple disconnected subgraphs,
        # then calculate kernighan_lin_n_groups variant for each group of interconnected groups independently
        swapped_communities = []
        for component in connected_components:
            connected_community_tuple = [(c[:],i) for i,c in enumerate(communities) if not set(c)-set(component)]
            connected_community = [c_t[0] for c_t in connected_community_tuple]
            if len(connected_community) > 1:
                while True:
                    connected_community, improved = kernighan_lin_n_groups(connected_community, adjacency_matrix, [])
                    #connected_community, improved = kernighan_lin_n_groups_direct_swap(connected_community, adjacency_matrix)
                    if improved == False:
                        swapped_communities.extend(connected_community)
                        break
            else:
                swapped_communities.extend(connected_community)

        #print("end")
        return swapped_communities
    else:
        #print(communities)
        #print("end")
        return communities





def kernighan_lin_classic(g_a, g_b, adj_matrix, swap_list):
    swap_list = swap_list

    improve_g_a = []
    improve_g_b = []

    # Calculate weights for the sum of internal and external edges for each node
    # and the improvement if exchanged for group a
    for i in g_a:
        internal = np.sum(adj_matrix[i][x] for x in g_a if x != i)
        external = np.sum(adj_matrix[i][x] for x in g_b if x != i) # Case x == i should not happen here as i is not part of g_b
        improvement = external-internal
        improve_g_a.append(improvement)

    # Calculate weights for the sum of internal and external edges for each node
    # and the improvement if exchanged for group b
    for i in g_b:
        internal = np.sum(adj_matrix[i][x] for x in g_b if x != i)
        external = np.sum(adj_matrix[i][x] for x in g_a if x != i)
        improvement = external - internal
        improve_g_b.append(improvement)

    # Dict swaps and their corresponding influence (score) on the improvement of the partition
    swap_dict = {}

    # Calculate improvement scores if two nodes are swapped
    for idx_i, v_i in enumerate(g_a):
        for idx_j, v_j in enumerate(g_b):
            # score = improvement(a) + improvement(b) - 2*weight(edge(a,b))
            swap_score = improve_g_a[idx_i] + improve_g_b[idx_j] - 2*adj_matrix[v_i][v_j]
            # Add swap score to dict
            swap_dict[(v_i,v_j)] = swap_score

    try:
        # Determine maximal swap score
        max_swap = max(swap_dict, key=swap_dict.get)
    except:
        #print(swap_list)
        improved = False
        # Swap
        for tuple in swap_list:
            g_a.append(tuple[1])
            g_b.append(tuple[0])
            improved = True
        return g_a, g_b, improved
    else:
        # If at least one swap is positive calculate further swaps without the "to-be-"swapped pair
        if swap_dict[max_swap] > 0:
            #print("Swap!")
            g_a.remove(max_swap[0])
            g_b.remove(max_swap[1])
            swap_list.append(max_swap)
            return kernighan_lin_classic(g_a, g_b, adj_matrix, swap_list)
        # If no positive swap exists, perform all determined swaps and return new partition
        else:
            improved = False
            # Swap
            for tuple in swap_list:
                g_a.append(tuple[1])
                g_b.append(tuple[0])
                improved = True
            return g_a, g_b, improved



def kernighan_lin_direct_swap(g_a, g_b, adj_matrix, swap_list):
    improved = True
    while True:
        improve_g_a = []
        improve_g_b = []

        # Calculate weights for the sum of internal and external edges for each node
        # and the improvement if exchanged for group a
        for i in g_a:
            internal = np.sum(adj_matrix[i][x] for x in g_a if x != i)
            external = np.sum(adj_matrix[i][x] for x in g_b if x != i)
            improvement = external-internal
            improve_g_a.append(improvement)

        # Calculate weights for the sum of internal and external edges for each node
        # and the improvement if exchanged for group b
        for i in g_b:
            internal = np.sum(adj_matrix[i][x] for x in g_b if x != i)
            external = np.sum(adj_matrix[i][x] for x in g_a if x != i)
            improvement = external - internal
            improve_g_b.append(improvement)

        # Dict swaps and their corresponding influence (score) on the improvement of the partition
        swap_dict = {}

        # Calculate improvement scores if two nodes are swapped
        for idx_i, v_i in enumerate(g_a):
            for idx_j, v_j in enumerate(g_b):
                # score = improvement(a) + improvement(b) - 2*weight(edge(a,b))
                swap_score = improve_g_a[idx_i] + improve_g_b[idx_j] - 2*adj_matrix[v_i][v_j]
                # Add swap score to dict
                swap_dict[(v_i,v_j)] = swap_score

        # Determine maximal swap score
        max_swap = max(swap_dict, key=swap_dict.get)
        #print(max_swap)
        # If at least one swap is positive calculate further swaps without the "to-be-"swapped pair
        if swap_dict[max_swap] > 0:
            # swap
            g_a.remove(max_swap[0])
            g_b.remove(max_swap[1])
            g_a.append(max_swap[1])
            g_b.append(max_swap[0])
        # If no positive swap exists, perform all determined swaps and return new partition
        else:
            improved = False
            break
    return g_a, g_b, improved


def kernighan_lin_n_groups(partitions, adj_matrix, swap_list):
    swap_list = swap_list

    # For each vertex calculate each improvement value against each group
    improve_dict = {}
    for idx_i, p_i in enumerate(partitions):
        for idx_j, p_j in enumerate(partitions):
            if idx_i != idx_j:
                for vp_i in p_i:
                    internal = np.sum(adj_matrix[vp_i][x] for x in p_i if x != vp_i)
                    external = np.sum(adj_matrix[vp_i][x] for x in p_j if x != vp_i)
                    improvement = external - internal
                    improve_dict[((idx_i, idx_j), vp_i)] = improvement

    # Dict swaps and their corresponding influence (score) on the improvement of the partition
    swap_dict = {}

    # Calculate swap score only for two vertices if:
    # the external edge score from vertex one belongs to the group of vertex two and
    # the external edge score from vertex two belongs to the group of vertex one.
    # I.e. the correct pairs have the following structure:
    # ((G1,G2),v1) ((G2,G1),v2), with G=Group; v=Vertex
    # Moreover avoid double calculation.
    for idx_a, entry_a in enumerate(improve_dict):
        for idx_b, entry_b in enumerate(improve_dict):
            if entry_a[0][0] == entry_b[0][1] and entry_a[0][1] == entry_b[0][0] and entry_a[0][0] < entry_b[0][0]:
                # score = improvement(a) + improvement(b) - 2*weight(edge(a,b))
                swap_score = improve_dict[entry_a] + improve_dict[entry_b] - 2 * adj_matrix[entry_a[1]][entry_b[1]]
                # Add swap score to dict
                swap_dict[(entry_a[0][0], entry_b[0][0]), (entry_a[1], entry_b[1])] = swap_score


    # Determine maximal swap score
    max_swap = max(swap_dict, key=swap_dict.get)
    #print(max_swap)
    #print(swap_dict[max_swap])
    # If at least one swap is positive calculate further swaps without the "to-be-"swapped pair
    if swap_dict[max_swap] > 0:
        #print("swap: " + str(max_swap) + " : " + str(swap_dict[max_swap]))
        partitions[max_swap[0][0]].remove(max_swap[1][0])
        partitions[max_swap[0][1]].remove(max_swap[1][1])
        swap_list.append(max_swap)
        return kernighan_lin_n_groups(partitions, adj_matrix, swap_list)
    # If no positive swap exists, perform all determined swaps and return new partition
    else:
        improved = False
        # Swap
        for tuple in swap_list:
            partitions[tuple[0][0]].append(tuple[1][1])
            partitions[tuple[0][1]].append(tuple[1][0])
            improved = True
        return partitions, improved


def kernighan_lin_n_groups_direct_swap(partitions, adj_matrix):
    improved = True
    while True:
        improve_dict = {}
        for idx_i, p_i in enumerate(partitions):
            for idx_j, p_j in enumerate(partitions):
                if idx_i != idx_j:
                    for vp_i in p_i:
                        internal = np.sum(adj_matrix[vp_i][x] for x in p_i if x != vp_i)
                        external = np.sum(adj_matrix[vp_i][x] for x in p_j if x != vp_i)
                        improvement = external - internal
                        improve_dict[((idx_i, idx_j), vp_i)] = improvement

        # Dict swaps and their corresponding influence (score) on the improvement of the partition
        swap_dict = {}

        # Calculate swap score only for two vertices if:
        # the external edge score from vertex one belongs to the group of vertex two and
        # the external edge score from vertex two belongs to the group of vertex one.
        # I.e. the correct pairs have the following structure:
        # ((G1,G2),v1) ((G2,G1),v2), with G=Group; v=Vertex
        # Moreover avoid double calculation.
        for idx_a, entry_a in enumerate(improve_dict):
            for idx_b, entry_b in enumerate(improve_dict):
                if entry_a[0][0] == entry_b[0][1] and entry_a[0][1] == entry_b[0][0] and entry_a[0][0] < entry_b[0][0]:
                    # score = improvement(a) + improvement(b) - 2*weight(edge(a,b))
                    swap_score = improve_dict[entry_a] + improve_dict[entry_b] - 2 * adj_matrix[entry_a[1]][entry_b[1]]
                    # Add swap score to dict
                    swap_dict[(entry_a[0][0], entry_b[0][0]), (entry_a[1], entry_b[1])] = swap_score

        # Determine maximal swap score
        max_swap = max(swap_dict, key=swap_dict.get)
        #print(max_swap)
        #print(swap_dict[max_swap])
        # If at least one swap is positive calculate further swaps without the "to-be-"swapped pair
        if swap_dict[max_swap] > 0:
            #print(partitions)
            #Swap
            #print("swap: " + str(max_swap) + " : " + str(swap_dict[max_swap]))
            partitions[max_swap[0][0]].remove(max_swap[1][0])
            partitions[max_swap[0][1]].remove(max_swap[1][1])
            partitions[max_swap[0][0]].append(max_swap[1][1])
            partitions[max_swap[0][1]].append(max_swap[1][0])
            #print(partitions)
        # If no positive swap exists, perform all determined swaps and return new partition
        else:
            improved = False
            break
    return partitions, improved