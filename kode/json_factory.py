import simplejson as json
import igraph as ig
import numpy as np
import os
import community as louvain
##### !!!!! This json uses a modified version of community_louvain.json, which includes a edge attribute named 'count' to calculate the mean edge weight of community graphs !!!!! #####
def build_json(hierarchy_dict, h5_data, dataset_name, graph, json, threshold):
	# data set dict
	ds_dict = {}

	# graph dict
	g_dict = {}

	# Maximum hierarchy size
	hmax = len(hierarchy_dict["dendro"]) - 1

	# Add pseudo entry to trigger single node dict creation
	hierarchy_dict[hmax+1] = {}
	for hidx, hdict in hierarchy_dict.items():
		if not isinstance(hidx, int):
			continue
		# Dendrogram list is sorted inversely to hierarchy dict. Therefore, the dendrogram index has to be recalculated.
		didx = hmax - hidx
		# edge dict
		e_dict = {}
		# node dict
		n_dict = {}
		# hierarchy dicr
		h_dict = {}
		if didx > -1:
			# Nodes
			for com, nodes in hierarchy_dict["inv_dendro"][didx].items():
				# attribute dict
				a_dict = {}
				a_dict["index"] = com
				a_dict["name"] = "h%in%i"%(hidx,com)
				a_dict["childs"] = nodes
				a_dict["mzs"] = list(h5_data.columns[hdict["communities"][com]])
				try:
					a_dict["membership"] = hierarchy_dict["dendro"][didx+1][com]
				except Exception as e:
					print(e)
				n_dict["h%in%i"%(hidx,com)] = a_dict
		else:
			# single nodes are always first entry in dendro
			for node, com in hierarchy_dict["dendro"][0].items():
				a_dict = {}
				a_dict["index"] = node
				a_dict["name"] = h5_data.columns[node]
				a_dict["membership"] = com
				a_dict["mzs"] = [h5_data.columns[node]]
				n_dict["h%in%i"%(hidx,node)] = a_dict
		# Edges
		if didx > -1:
			community = louvain.partition_at_level(hierarchy_dict["dendro"], didx)
			edges = louvain.induced_graph(community, graph).edges(data=True)
		else:
			edges = graph.edges(data=True)
		idx = 0
		for source, target, weight in edges:
			# Include source == target for inner edge weight.
			#print(weight)
			if source != target:
				a_dict = {}
				a_dict["index"] = idx
				a_dict["name"] = "h%ie%i"%(hidx,idx)
				a_dict["source"] = "h%in%i"%(hidx,source)
				a_dict["target"] = "h%in%i"%(hidx,target)
				try:
					count = weight["count"]
				except:
					count = 1
				#print(count)
				a_dict["weight"] = weight["weight"] / count
				e_dict["h%ie%i"%(hidx,idx)] = a_dict
				idx += 1
			
		h_dict["nodes"] =  n_dict
		h_dict["edges"] =  e_dict
		g_dict["hierarchy%i"%(hidx)] = h_dict

	ds_dict["graph"] = g_dict
	ds_dict["dataset"] = dataset_name
	ds_dict["threshold"] = threshold

	#mzs = [x for x in np.round(h5_data.columns, 3)]
	mzs = [x for x in h5_data.columns]
	mzs_dict = {}
	for mz in mzs:
		mzs_dict[str(mz)] = {}
		for hy, vals in g_dict.items():
			for nid, props  in vals["nodes"].items():
				try:
					if mz in props["mzs"]:
						mzs_dict[str(mz)][hy] = nid
						break
				# Last hierarchy has no "mzs" prop
				except Exception as e:
					print(e)
					if mz == props["name"]:
						mzs_dict[str(mz)][hy] = nid


	ds_dict["mzs"] = mzs_dict

	json["graphs"]["graph%i"%(hierarchy_dict["graph_idx"])] = ds_dict

	return json