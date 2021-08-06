# customized
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.readwrite.gml import read_gml
from nodes import Nodes
from computer import Computer
from user import User
# networkx
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.utils import pairwise
import matplotlib.pyplot as plt
# essential
import os
import uuid
import itertools
import random

def read_graph():
  """
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
      # plt.show()

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

def find_all_leaves(G: nx.Graph) -> list:
  """
    [Parameters]
      G: nx.Graph -- graph
    [Return]
      leaves as list
  """
  return [v for v, d in G.in_degree() if d == 0]

def build_graph():
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

def neo4j_builder():
  """
    Build neo4j according to generated graph
  """
  pass

def struct_graph(*layer_sizes, nonjump_percentage: float, 
                  extra_edge_per_node_lower_bound: int,
                  extra_edge_per_node_upper_bound: int):
  """
    Generate randomized graph according to the pattern of BloodHound topologies
    [Parameters]
      layer_sizes: layer list, each element represents the number of nodes in each layer
      extra_edges_percentage: the percentage of normal link, 1-extra_edges+percentage = # of jump
      extra_edge_per_node_lower_bound: min val of outgoing degree
      extra_edge_per_node_upper_bound: max val of outgoing degree
    [Return]
      generated graph
  """
  # split the array in terms of layer_sizes
  extends = pairwise(itertools.accumulate((0,) + layer_sizes))
  # range for each layer
  layers = [range(start, end) for start, end in extends]
  G = nx.DiGraph()
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
      for j in range(random.choice(range(extra_edge_per_node_lower_bound, extra_edge_per_node_upper_bound))):
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
          G.add_edges_from([(node, v)])
  return G

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

print("----networkx----")
layer_sizes = [5,21,14,12,11,1]
# layer_color = ["gold", "violet", "blue", "black", "red"]
computer_list = []
user_list = []

G = struct_graph(*layer_sizes, nonjump_percentage=0.9, extra_edge_per_node_lower_bound=10, extra_edge_per_node_upper_bound=12)
# color = [layer_color[data["layer"]] for v, data in G.nodes(data=True)]
print("Nodes: ", G.nodes())
print("Node attributes: ", nx.get_node_attributes(G,"layer"))
pos = nx.multipartite_layout(G, subset_key="layer")
nx.draw(G, pos,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=0.1,
        labels={k: k for k in range(sum(layer_sizes))},
        font_size=10)
print("Adjacency Matrix: ")
print(nx.to_dict_of_dicts(G))
print(nx.info(G))
algorithm_2(G)

# store graph
#store_graph(G)
plt.show()
# read_graph()
