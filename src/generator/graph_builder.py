# networkx
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.utils import pairwise
import matplotlib.pyplot as plt

import sys
import itertools
import random

print("----networkx----")
layer_sizes = [4, 5, 2, 1]
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
  pass


def neo4j_builder():
  pass


def struct_graph(*layer_sizes):
  # split the array in terms of layer_sizes
  extends = pairwise(itertools.accumulate((0,) + layer_sizes))
  # range for each layer
  layers = [range(start, end) for start, end in extends]
  print("Total layers: ", layers)
  G = nx.DiGraph()
  # i - index for each range
  # layer - range per se
  # [Add Nodes] to each layer by using linear seq
  for (i, layer) in enumerate(layers):
    G.add_nodes_from(layer, layer=i)
  # # [Add Edges] 2 layers in a pair, iterating on layers
  # for layer_l, layer_r in pairwise(layers):
  #   if random.random() < 0.7:
  #     # i - idx for each node in the current layer
  #     for i in layer_l:
  #       print("Current layer: ",layer_l, "->",i)
  #       # outgoing degree is within [1,3)
  #       for j in range(random.choice(range(1, 3))):
  #         v = random.choice(layer_r)
  #         if (G.has_edge(i, v)==False):
  #           G.add_edges_from([(i, v)])
  for idx in range(len(layers)-1):
    for node in layers[idx]:
      print("Current range: ", layers[idx])
      for j in range(random.choice(range(1, 3))):
        if random.random() < 0.7:
          v = random.choice(layers[idx+1])
        else:
          # choose layer
          l = random.choice(range(idx+1, len(layers)))
          # choose node
          v = random.choice(layers[l])
        if (G.has_edge(node, v) == False):
          G.add_edges_from([(node, v)])
  return G


G = struct_graph(*layer_sizes)
print(G.nodes())
# color = [layer_color[data["layer"]] for v, data in G.nodes(data=True)]
pos = nx.multipartite_layout(G, subset_key="layer")
nx.draw(G, pos,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=0.1,
        labels={k: k for k in range(12)},
        font_size=10)

print(nx.shortest_path(G)[0][11])
plt.show()


print("----memory usage----")
print("Nodes: ", sys.getsizeof(Nodes()))
print("c: ", sys.getsizeof(c))
print("u: ", sys.getsizeof(u))
# print("ch: ", sys.getsizeof(ch))
# print("uh: ", sys.getsizeof(uh))
print("G: ", sys.getsizeof(G))
# plt.figure(figsize=(8, 8))
# nx.draw(G, pos, node_color=color, with_labels=False)
# plt.axis("equal")
# plt.show()
