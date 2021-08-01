# customized
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

def struct_graph(*layer_sizes):
  """
    Generate randomized graph according to the pattern of BloodHound topologies
    [Parameters]
      layer_sizes: layer list, each element represents the number of nodes in each layer
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
    for node in layers[idx]:
      for j in range(random.choice(range(1, 3))):
        if random.random() < 0.7: # 70% percent of time, bipartite
          v = random.choice(layers[idx+1])
        else: # 30% percent of time, jump
          # choose layer
          l = random.choice(range(idx+1, len(layers)))
          # choose node
          v = random.choice(layers[l])
        if (G.has_edge(node, v) == False):
          G.add_edges_from([(node, v)])
  return G

print("----networkx----")
layer_sizes = [5,4,6,4,4,3,1]
# layer_color = ["gold", "violet", "blue", "black", "red"]
computer_list = []
user_list = []

G = struct_graph(*layer_sizes)
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
print("[[All leaves]]: ",find_all_leaves(G))
#print(nx.shortest_path(G)[0][11])

# store graph
store_graph(G)
plt.show()
read_graph()
