# essential
# from networkx.readwrite import edgelist
# from dp_learn.bh.src.utility.utility import find_all_leaves
# from dp_learn.bh.src.utility.utility import graph_utility
import os
import uuid
import itertools
import random
from typing import List
# networkx
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.centrality.current_flow_betweenness import edge_current_flow_betweenness_centrality
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


class GraphGenerator:
  def __init__(self, layer_sizes: list) -> None:
    """
    [Round]
    randomization -> picked edges -> get ground truth(classification) -> make it as one dataset
    dataset should have both original graph and picked edges classification

    Do the process one by one
    """
    self.G = nx.DiGraph()
    self.layer_sizes = layer_sizes
    self.DA = sum(layer_sizes)-1
    self.blockable_edges = []
    self.entries = []
    # in_degree & out_degree hasn't been updated yet
    # but they are in attribtue matrix now
    self.nodes_attributes = ['layer', 'in_degree', 'out_degree']
    self.edges_attributes = ['blockable', 'connected_entries', 'level_gap']
  
  # debug
  def get_graph(self) -> nx.DiGraph():
    return self.G

  def get_blockable_edges(self) -> list:
    return self.blockable_edges

  def get_entries(self) -> list:
    return self.entries

  # utilities
  def __delete_graph(self) -> None:
    """
    Should be called every time we finish a complete data [Round]
    """
    self.G.clear()

  def draw_graph(self):
    """
    Should only be called for testing purpose on small graph
    """
    # configure drawing parameters
    edge_color = [self.G[u][v]['blockable']
                  for u, v in self.G.edges]  # draw according to blockable or not
    # determine position as multipartite graph
    # 'layer' is fixed here!
    pos = nx.multipartite_layout(self.G, subset_key="layer")
    # draw graph
    nx.draw(self.G, pos,
            with_labels=True,
            node_size=5,
            connectionstyle="arc3,rad=-0.2",
            width=1,
            edge_color = edge_color,
            labels={k: k for k in range(sum(self.layer_sizes))},
            font_size=10)
    plt.show()
    pass

  def graph_debug(self) -> None:
    """
    print out the basic informations
    """
    # debug information
    print("---------------DEBUG START---------------")
    print("\nTest Basic Information......")
    print("Nodes: ", self.G.nodes(data=True))
    print("Edges ", self.G.edges(data=True))
    print("\n----------------DEBUG END----------------")

  def read_graph(self):
    """
      Read graph from .gml, need to config nx.draw() to recover the original layout
      But all of the data could be normally read and create as a new nx.Graph
      Read all .gmls under the folder one time!

      [Parameters]
        Read from where
      [Return]
        should be a list of G(nx graph)
    """
    path = "./data/train/"
    for filename in os.listdir(path):
      if filename.endswith(".gml"):  # read out graph
        G_tmp = nx.read_gml(os.path.join(path, filename), label="label")
        pos_tmp = nx.multipartite_layout(G_tmp, subset_key="layer")
        nx.draw(G_tmp, pos_tmp,
                with_labels=True,
                node_size=5,
                connectionstyle="arc3,rad=-0.2",
                edge_color=[G_tmp[u][v]['blockable'] for u, v in G_tmp.edges],
                width=1,
                font_size=10)
        # print(os.path.join(path, filename))
        # print(G_tmp.nodes(data=True))
        # print(G_tmp.edges(data=True))
        plt.show()

  def store_graph(self) -> None:
    """
      Store graph as file

      [Parameters]
        G -- graph
      [Return]
        None
    """
    name = "./data/train/" + "Graph-" + uuid.uuid4().hex + ".gml"
    nx.write_gml(self.G, name)
    pass

  def add_new_attributes(self, target: str, attr: str, default_value: any) -> None:
    """
      add new attributes to each node / edges

      [Parameters]
        G: graph
        target: "nodes" or "edges"
        attr: new attribtue name
        default_value: default value for the attribtues (ONLY int for now)
      [Return]
        None
    """
    print(getattr(self.G, target))
    if target == 'nodes':
      self.nodes_attributes.append(attr)
      for node in self.G.nodes:
        self.G.nodes[node][attr] = default_value
    elif target == 'edges':
      self.edges_attributes.append(attr)
      for edge in self.G.edges:
        self.G[edge[0]][edge[1]][attr] = default_value
    return

  def edge_filter(self, attr: str, attr_val: any) -> list:
    """
    Return all blockable edges as a list [(src, dst),(),()...]
    
    [Parameters]
      attr -- attribtue name
      attr_val -- atttribute value you want to get
    [Return]
      filtered edges
    """
    edge_list = []
    for edge in self.G.edges:
      if self.G[edge[0]][edge[1]][attr] == attr_val:
        edge_list.append((edge[0], edge[1]))
    return edge_list

  def struct_graph(self,
                   nonjump_percentage: float,
                   blockable_percentage: float,
                   outgoing_lower_bound: int,
                   outgoing_upper_bound: int) -> None:
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
    extends = pairwise(itertools.accumulate((0,) + (*self.layer_sizes,)))
    # range for each layer
    layers = [range(start, end) for start, end in extends]
    # i - index for each range
    # layer - range per se
    # [Add Nodes]
    for (i, layer) in enumerate(layers):
      self.G.add_nodes_from(layer, layer=i)
    # [Add Edges]
    for idx in range(len(layers)-1):
      # sequentially choose src node
      for node in layers[idx]:
        # loop for the number of outgoing edge of each node
        for j in range(1, max(2, random.choice(range(outgoing_lower_bound, outgoing_upper_bound)))):
          # randomly choose dst node
          if random.random() < nonjump_percentage:  # 70% percent of time, bipartite
            v = random.choice(layers[idx+1])
          else:                                         # 30% percent of time, jump
            # choose layer, randomly jump in the rage of (idx+1, len(layers))
            if (idx+2 == len(layers)):
              continue
            l = random.choice(range(min(idx+2, len(layers)), len(layers)))
            # choose node
            v = random.choice(layers[l])
          # connect!
          if (self.G.has_edge(node, v) == False):
            tmp = random.random()
            if tmp < blockable_percentage:
              self.G.add_edge(node, v, blockable=True)
            else:
              self.G.add_edge(node, v, blockable=False)
          self.G[node][v]['level_gap'] = self.G.nodes[v]['layer'] - self.G.nodes[node]['layer']
    # prepare necessary attributess
    print("\nTest add_new_attributes......")
    self.add_new_attributes('edges', 'connected_entries', 0)
    self.add_new_attributes('nodes', 'in_degree', 0)
    self.add_new_attributes('nodes', 'out_degree', 0)
    print("Add connected entries rate as new edge attribtues: ", self.G.edges(data=True))
    # set blockable_edges
    self.entries = self.__find_all_leaves()
    self.blockable_edges = self.edge_filter('blockable', True)

  # algorithms
  def __find_all_leaves(self) -> list:
    """
      Get all leaf nodes

      [Parameters]
        G: nx.Graph -- graph
      [Return]
        leaves as list
    """
    return [v for v, d in self.G.in_degree() if d == 0]

  def find_all_path(self, src: int, dst: int) -> list:
    """
    Return all path from src to dst

    [Parameters]
      G -- graph
      src -- source node idx
      dst -- target node idx
    [Return]
      sorted list according to path's length
    """
    return sorted(list(nx.all_simple_paths(self.G, src, dst)), key=lambda x: len(x))

  def linkable_entries(self, edge: tuple) -> list:
    """
      entries could reach the edge
      
      [Parameter]
        G -- graph
        edge -- target edge (src, dst)
      [Return]
        entries could reach the edge
    """
    linkable_entries = []
    for entry in self.entries:
      if nx.has_path(self.G, entry, edge[0]) == True:
        linkable_entries.append(entry)
    return linkable_entries

  def __successful_rate(self, sr: float) -> list:
    """
    return all possible successful rate as a list
    we assume all edge have the united successful rate for now, will improve in the future

    [Parameters]
      sr -- successful rate for each edge
    [Return]
      sr list
    """
    return [sr**i for i in range(len(self.layer_sizes))]

  def graph_utility(self, sr_prob: float) -> float:
    """
    Calculate the utility for the whole graph
    Used to evaluate the effect of randomization for each graph (used for building(build the label) loss function)
    we try to minimize the sr for each graph
    we assume the possibilities getting each entry are exactly the same, we improve
    """
    total_successful_rate: float = 0.0
    for entry in self.entries:
      try:
        stp = nx.shortest_path(self.G, source=entry, target=self.DA)
      except nx.NetworkXNoPath:
        print(f'No path between {entry} and {self.DA}')
        continue
      # calculate sr
      cur_path_successful_rate = 1.0
      for i in range(len(stp)):
        cur_path_successful_rate *= sr_prob
      # update total sr
      total_successful_rate += cur_path_successful_rate
    return total_successful_rate/len(self.entries)

  def simplify_to_tree(self, epoch: int):
    """
    cut the graph into simple tree
    
    [Parameters]
      epoch -- # of iteration
    [Return]
      None
    """
    # for i in range(epoch):
    #   total_sr = 1.0
    #   for entry in entries:
    pass

  def cut_strategy(self, budget: int, epoch: int):
    """
    Randomly pick blockable edges and find stp for each of the entires
    """
    if budget >= len(self.blockable_edges):
      print("[WARNING]: budget should never larger than the number of blockable edges!")
      exit(1)
    # store status
    G_tmp = self.G.copy()
    worst_sr = 1.0
    for i in range(epoch):
      blocked_edges = random.sample(self.blockable_edges, budget)
      print("\nTest blocked edges: ", blocked_edges)
      self.G.remove_edges_from(blocked_edges)
      total_sr = self.graph_utility(0.6)
      print("SR after blocking blockable edges: ", total_sr)
      worst_sr = min(total_sr, worst_sr)
      # recover status
      self.G = G_tmp.copy()
    print(f'Best performance: {worst_sr}')
    pass

  # to torch
  def networkx_to_torch(self):
    """
      Transfer netowrkx to fit torch geo
    """
    layer_list = list(nx.get_node_attributes(self.G, 'layer').values())
    in_degree_list = [val[1] for val in list(self.G.in_degree)]
    out_degree_list = [val[1] for val in list(self.G.out_degree)]
    # layer
    print("[Node]: In networkx_to_torch layer: ", np.array(layer_list))
    # in_degree & out_degree
    print("[Node]: In network_to_torch in_degree: ", np.array(in_degree_list))
    print("[Node]: In network_to_torch out_degree: ", np.array(out_degree_list))
    node_matrix = np.column_stack((layer_list, in_degree_list, out_degree_list))
    print(f'[Node Matrix]: {node_matrix}')

    # blockable
    blockable_list = list(nx.get_edge_attributes(self.G, 'blockable').values())
    connected_entries_list = list(
        nx.get_edge_attributes(self.G, 'connected_entries').values())
    level_gap_list = list(nx.get_edge_attributes(self.G, 'level_gap').values())
    print("[Edge]: In networkx_to_torch blockable: ", np.array(blockable_list))
    # connected_entries
    print("[Edge]: In networkx_to_torch connected_entries: ",
          np.array(connected_entries_list))
    # level_gap
    print("[Edge]: In networkx_to_torch level_gap: ", np.array(level_gap_list))
    edge_matrix = np.column_stack(
        (blockable_list, connected_entries_list, level_gap_list))
    print(f'[Edge Matrix]: {edge_matrix}')
    # for loop on nodes_attributes & edges_attributes respectively
    #   - nx.get_[node/edge]_attributes(self.G, '{attribute_name}').values()
    data = from_networkx(self.G)
    data.x = torch.tensor(node_matrix)
    #data.nodes = torch.tensor(list(self.G.nodes))
    # crap torch geometrics, need to build your own node feature matrix
    return data












