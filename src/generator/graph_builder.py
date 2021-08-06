# essential
import itertools
import random
import sys
sys.path.insert(1, '../utility')
# networkx
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.utils import pairwise
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.readwrite.gml import read_gml
import matplotlib.pyplot as plt
# customized
from nodes import Nodes
from computer import Computer
from user import User
from utility import read_graph

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
          print(f'tmp: {tmp}; percent: {blockable_percentage}')
          if tmp < blockable_percentage:
            G.add_edge(node, v, blockable=True)
          else:
            G.add_edge(node, v, blockable=False)
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


layer_sizes = [7,6,5,4,3,2,1]#[5,4,3,4,6,3,2,1]
G = struct_graph(*layer_sizes, nonjump_percentage=0.8, outgoing_lower_bound=3, 
                outgoing_upper_bound=5, blockable_percentage=0.3)

print("----networkx----")
# configure drawing parameters
edge_color = [G[u][v]['blockable'] for u,v in G.edges] # draw according to blockable or not
computer_list = []
user_list = []
# determine position as multipartite graph
pos = nx.multipartite_layout(G, subset_key="layer")
# draw graph
nx.draw(G, pos,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=1,
        edge_color = edge_color,
        labels={k: k for k in range(sum(layer_sizes))},
        font_size=10)
# debug information
print("Nodes: ", G.nodes())
print("Node attributes: ", nx.get_node_attributes(G,"layer"))
print("Edge attributes: ", nx.get_edge_attributes(G,"blockable"))
print("Adjacency Matrix: ", nx.to_dict_of_dicts(G))
print("Graph info: ", nx.info(G))
# algorithm_2(G)
# store graph
# store_graph(G)
plt.show()
# read_graph()
