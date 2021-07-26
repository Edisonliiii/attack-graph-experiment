# customized
from nodes import Nodes
from computer import Computer
from user import User
# networkx
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.utils import pairwise
import matplotlib.pyplot as plt
# essential
import itertools
import random


def find_all_leaves(G: nx.Graph) -> list:
  return [v for v, d in G.in_degree() if d == 0]

print("----networkx----")
layer_sizes = [5,4,6,4,4,3,1]
# layer_color = ["gold", "violet", "blue", "black", "red"]

computer_list = []
user_list = []

# draw using number only
def build_graph():
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
  pass

def struct_graph(*layer_sizes):
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
# print(G.nodes[0], G.in_degree[0], G.out_degree[0])
# print(G.nodes[1], G.in_degree[1], G.out_degree[1])
# print(G.nodes[2], G.in_degree[2], G.out_degree[2])
print("Adjacency Matrix: ")
print(nx.to_dict_of_dicts(G))
print(nx.info(G))
print("[[All leaves]]: ",find_all_leaves(G))
#print(nx.shortest_path(G)[0][11])

# store graph
nx.write_gml(G, "test.gml")
plt.show()

# rerender the graph, ignore label, because there is already one out there
G2 = nx.read_gml("./test.gml", label="label")
pos2 = nx.multipartite_layout(G2, subset_key="layer")
nx.draw(G2, pos2,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=0.1,
        font_size=10)
plt.show()
