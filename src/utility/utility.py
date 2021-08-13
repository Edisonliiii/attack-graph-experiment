# essential
import os
from typing import List, Union
import uuid
import itertools
import random
from typing import List
# networkx
import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.function import nodes
from networkx.drawing.nx_agraph import to_agraph
from networkx.utils import pairwise
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.readwrite.gml import read_gml
from fibheap import *
#torch
import torch
from torch_geometric.utils import from_networkx
# numpy
import numpy as np
# customized
from nodes import Nodes
from computer import Computer
from user import User

# global lists
nodes_attributes = ['layer', 'in_degree', 'out_degree']
edges_attributes = ['blockable', 'connected_entries', 'level_gap']


# ---------- dataset build utilities
def networkx_to_torch(G: nx.Graph):
  """
    Transfer netowrkx to fit torch geo
  """
  layer_list = list(nx.get_node_attributes(G, 'layer').values())
  in_degree_list = [val[1] for val in list(G.in_degree)]
  out_degree_list = [val[1] for val in list(G.out_degree)]
  # layer
  print("[Node]: In networkx_to_torch layer: ", np.array(layer_list))
  # in_degree & out_degree
  print("[Node]: In network_to_torch in_degree: ", np.array(in_degree_list))
  print("[Node]: In network_to_torch out_degree: ", np.array(out_degree_list))
  node_matrix = np.column_stack((layer_list, in_degree_list, out_degree_list))
  print(f'[Node Matrix]: {node_matrix}')

  # blockable
  blockable_list = list(nx.get_edge_attributes(G, 'blockable').values())
  connected_entries_list = list(
      nx.get_edge_attributes(G, 'connected_entries').values())
  level_gap_list = list(nx.get_edge_attributes(G, 'level_gap').values())
  print("[Edge]: In networkx_to_torch blockable: ", np.array(blockable_list))
  # connected_entries
  print("[Edge]: In networkx_to_torch connected_entries: ", np.array(connected_entries_list))
  # level_gap
  print("[Edge]: In networkx_to_torch level_gap: ", np.array(level_gap_list))
  edge_matrix = np.column_stack((blockable_list, connected_entries_list, level_gap_list))
  print(f'[Edge Matrix]: {edge_matrix}')
  # for loop on nodes_attributes & edges_attributes respectively
  #   - nx.get_[node/edge]_attributes(G, '{attribute_name}').values()
  data = from_networkx(G)
  #data.nodes = torch.tensor(list(G.nodes))
  # crap torch geometrics, need to build your own node feature matrix
  return data



# ---------- necessary information for aggregation
def linkable_entries(G: nx.Graph, edge: tuple, entries: list) -> list:
  """
    entries could reach the edge
    
    [Parameter]
      G -- graph
      edge -- target edge
      entries -- entry points
    [Return]
      entries could reach the edge
  """
  linkable_entries = []
  for entry in entries:
    if nx.has_path(G, entry, edge[0]) == True:
      linkable_entries.append(entry)
  return linkable_entries

def find_all_path(G: nx.Graph, src: int, dst: int) -> list:
  """
  Return all path from src to dst

  [Parameters]
    G -- graph
    src -- source node
    dst -- target node
  [Return]
    sorted list according to path's length
  """
  return sorted(list(nx.all_simple_paths(G, src, dst)), key=lambda x : len(x))

def find_all_leaves(G: nx.Graph) -> list:
  """
    Get all leaf nodes

    [Parameters]
      G: nx.Graph -- graph
    [Return]
      leaves as list
  """
  return [v for v, d in G.in_degree() if d == 0]
# ---------- graph utilities

def graph_debug(G: nx.Graph) -> None:
  """
  print out the basic informations
  """
  # debug information
  print("---------------START---------------")
  print("\nTest Basic Information......")
  print("Nodes: ", G.nodes(data=True))
  print("Edges ", G.edges(data=True))
  print("\n----------------END----------------")

def read_graph():
  """
    Read graph from .gml

    [Parameters]
      Read from where
    [Return]
      should be G(nx graph), but needs to fix impl
  """
  path = "./data/train/"
  for filename in os.listdir(path):
    if filename.endswith(".gml"): # read out graph
      G_tmp = nx.read_gml(os.path.join(path, filename), label="label")
      pos_tmp = nx.multipartite_layout(G_tmp, subset_key="layer")
      nx.draw(G_tmp, pos_tmp,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=0.1,
        font_size=10)
      print(os.path.join(path, filename))
      plt.show()

def store_graph(G: nx.Graph):
  """
    Store graph as file

    [Parameters]
      G -- graph
    [Return]
      None
  """
  name = "./data/train/" + "Graph-" +  uuid.uuid4().hex + ".gml"
  nx.write_gml(G, name)
  pass

def ad_concepts():
  """
    Define graph nodes and edges(relationships) in the context of BloodHound
  """
  computers = []
  users = []
  groups = []
  gpos = []
  outs = []

  relationship = {
    "AdminTo": [
      ("Group", "Computer")
    ],
    "GenericAll": [
      ("Group", "Computer")
    ],
    "AddMember": [
      ("Group", "Group")
    ],
    "AllowedToDelegate": [
      ("User", "Computer"),
      ("Computer", "Computer")
    ],
    "CanRDP": [
      ("Group", "Computer"),
      ("User", "Computer")
    ],
    "Contains": [
      ("Domain", "OU"),
      ("OU", "Computer")
    ],
    "DCSync": [
      ("Group", "Domain")
    ],
    "ExecuteDCOM": [
      ("User", "Computer"),
      ("Group", "Computer")
    ],
    "ForceChangePassword": [
      ("Group", "User")
    ],
    "GenericWrite": [],
    "GetChanges": [],
    "GetChangesAll": [],
    "GpLink": [],
    "HasSession": [],
    "MemberOf": [],
    "Owns": [],
    "ReadLAPSPassword": [],
    "WriteDacl": [],
    "WriteOwner": []
  }
  pass

def neo4j_mapper():
  """
    Build neo4j according to generated graph
  """
  pass

def add_new_attributes(G: nx.Graph, target: str, attr: str, default_value: any) -> None:
  """
    add new attributes to each node / edges

    [Parameters]
      G: graph
      target: "nodes" or "edges"
      attr: new attribtue name
      default_value: default value for the attribtues
    [Return]
      None
  """
  global nodes_attributes, edges_attributes
  print(getattr(G, target))
  if target == 'nodes':
    nodes_attributes.append(attr)
    for node in G.nodes:
      G.nodes[node][attr] = default_value
  elif target == 'edges':
    edges_attributes.append(attr)
    for edge in G.edges:
      G[edge[0]][edge[1]][attr] = default_value
  return
  
def struct_graph(*layer_sizes, nonjump_percentage: float,
                blockable_percentage: float,
                outgoing_lower_bound: int,
                outgoing_upper_bound: int):
  """
    Generate randomized graph according to the pattern of BloodHound topologies
    [Parameters]
      layer_sizes: layer list, each element represents the number of nodes in each layer
      extra_edges_percentage: the percentage of normal link, 1-extra_edges+percentage = # of jump
      outgoing_lower_bound: min val of outgoing degree
      outgoing_upper_bound: max val of outgoing degree, has to be +1 than .._lower_bound (python random limit[,))
    [Return]
      generated graph as Graph
  """
  # declare graph object
  G = nx.DiGraph()
  # split the array in terms of layer_sizes
  extends = pairwise(itertools.accumulate((0,) + layer_sizes))
  # range for each layer
  layers = [range(start, end) for start, end in extends]
  # i - index for each range
  # layer - range per se
  # [Add Nodes]
  for (i, layer) in enumerate(layers):
    G.add_nodes_from(layer, layer=i)
  # [Add Edges]
  for idx in range(len(layers)-1):
    # sequentially choose src node
    for node in layers[idx]:
      # loop for the number of outgoing edge of each node
      for j in range(1, max(2, random.choice(range(outgoing_lower_bound, outgoing_upper_bound)))):
        # randomly choose dst node
        if random.random() < nonjump_percentage: # 70% percent of time, bipartite
          v = random.choice(layers[idx+1])
        else:                                         # 30% percent of time, jump
          # choose layer, randomly jump in the rage of (idx+1, len(layers))
          if (idx+2 == len(layers)):
            continue
          l = random.choice(range(min(idx+2, len(layers)), len(layers)))
          # choose node
          v = random.choice(layers[l])
        # connect!
        if (G.has_edge(node, v) == False):
          tmp = random.random()
          if tmp < blockable_percentage:
            G.add_edge(node, v, blockable=True)
          else:
            G.add_edge(node, v, blockable=False)
        G[node][v]['level_gap'] = G.nodes[v]['layer'] - G.nodes[node]['layer']
  # prepare necessary attributess
  print("\nTest add_new_attributes......")
  add_new_attributes(G, 'edges', 'connected_entries', 0)
  add_new_attributes(G, 'nodes', 'in_degree', 0)
  add_new_attributes(G, 'nodes', 'out_degree', 0)
  print("Add connected entries rate as new edge attribtues: ", G.edges(data=True))
  # add_new_attributes(G, 'edges', 'level_gap', 0)
  # print("Add layer gap as new edge attribtues: ", G.edges(data=True))
  return G

# ---------- graph algorithms
def algorithm_2(G: nx.Graph):
  """
    Security Cascade Graphs : algorithm 2 O((l^b)n)
  """
  # get all leaves
  leaves = find_all_leaves(G)
  dst = G.number_of_nodes() - 1
  print(leaves)
  # find the stp from all leaves
  stps = []
  for leaf in leaves:
    shortest_path = nx.shortest_path(G, source=leaf, target=dst)
    print("STP: ", shortest_path)
    stps.append(shortest_path)
  print(stps)
  # delete one edge at a time
  for stp in stps:
    d_tor_length = len(stp)
    new_stp = []
    # tmp_G = G
    src = stp[0]
    for i in range(0, len(stp)-1):
      tmp_G = G.copy() # recover
      tmp_G.remove_edge(stp[i], stp[i+1])
      print(f'From {src} Removing {stp[i]}, {stp[i+1]}')
      try:
        tmp_stp = nx.shortest_path(tmp_G, source=src, target=dst)
      except nx.NetworkXNoPath:
        print(f'No path between {src} and {dst}')
        continue
      if (len(tmp_stp) >= d_tor_length):
        new_stp = tmp_stp
        print("Updated stp: ", new_stp)
  pass

def algorithm_tree(G: nx.Graph, total_layer: int):
  """
    Tree topology, most basic one O(blogn+n)
    ONLY works on simple tree
    has to be at least one path from node -> DA
    or it would always be zero
  """
  # check if it is simple tree or not
  for (node,value) in G.out_degree():
    if value > 1:
      print("This is not a simple tree.")
      return
  # get all successful rate
  sr_list = successful_rate(0.9, total_layer)
  print(sr_list)
  # find all leaves
  leaf_nodes = find_all_leaves(G)
  print(f'Leaves are: {leaf_nodes}')
  paths_from_leaves = []
  for leaf in leaf_nodes:
    paths = nx.all_simple_paths(G, leaf, G.number_of_nodes()-1) # return as a list of lists [[]]
    tmp = list(paths)
    if len(tmp) > 0:
      paths_from_leaves.append(tmp[0]) # only get the first one, because we can ensure there is at most one path
  print(f'all paths from leaves: {paths_from_leaves}')
  for edge in paths_from_leaves:
    for idx in range(0, len(edge)-1):
      if G[edge[idx]][edge[idx+1]]['blockable']==True:
        print("!!!!!!!!!!!!Debug for another blockable edge!!!!!!!!!!!!!")
        print(f'** successful rate for the current step: {sr_list[idx+1]}')
        print(f'-- layer info {G.nodes[edge[idx]]}')
        print(f'-- Blockable {edge[idx]}, {edge[idx+1]}')
        # calculate how many entries involved with current edge
        G[edge[idx]][edge[idx+1]]['connected_entries'] += 1
        print(f'-- total sr: {G[edge[idx]][edge[idx+1]]["connected_entries"]}')
  
def successful_rate(sr: float, total_layer: int) -> list:
  return [sr**i for i in range(total_layer)]

def cost_function(G: nx.Graph, s: list, DA: int, prob: float) -> float:
  """
    Calculate successful rate in average
    [Parameter]
      s -- list of starting points
      DA -- index of the root domain admin
      prob -- all the same, probability of successful attack
    [Return]
  """
  total_successful_rate = 0.0
  for node in s:
    try:
      stp = nx.shortest_path(G, source=node, target=DA)
    except nx.NetworkXNoPath:
      print(f'No path between {src} and {dst}')
      continue
    cur_path_successful_rate = 1.0
    for i in range(len(stp)):
       cur_path_successful_rate *= prob
    total_successful_rate += cur_path_successful_rate
    return total_successful_rate/len(s)
