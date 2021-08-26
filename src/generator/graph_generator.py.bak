# essential
import enum
import copy
import heapq
import uuid
import itertools
import random
from typing import List
from warnings import catch_warnings
# networkx
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.centrality.current_flow_betweenness import edge_current_flow_betweenness_centrality
from networkx.algorithms.clique import number_of_cliques
from networkx.classes.function import nodes
from networkx.drawing.nx_agraph import to_agraph
from networkx.readwrite import edgelist
from networkx.utils import pairwise
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.readwrite.gml import read_gml
from fibheap import *
#torch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
# numpy
import numpy as np


class MaxHeapObj(object):
  def __init__(self, G: nx.Graph, attr: str, edge: tuple):
    """
    :param
      G -- in which graph
      attr -- on which attribute
      edge -- comparale obj
    """
    self.G = G
    self.attr = attr
    self.edge = edge
  def __lt__(self, other):
    return self.G[self.edge[0]][self.edge[1]][self.attr] > self.G[other.edge[0]][other.edge[1]][self.attr]
  def __eq__(self, other):
    return self.G[self.edge[0]][self.edge[1]][self.attr] == self.G[other.edge[0]][other.edge[1]][self.attr]

class MinHeap(object):
  def __init__(self): self.h = []
  def heappush(self, x): heapq.heappush(self.h, x)
  def heappop(self): return heapq.heappop(self.h)
  def __getitem__(self, i): return self.h[i]
  def __len__(self): return len(self.h)

class MaxHeap(MinHeap):
  def __init__(self, G, attr):
    self.h = []
    self.G = G
    self.attr = attr
  def heappush(self, x): heapq.heappush(self.h, MaxHeapObj(self.G, self.attr, x))
  def heappop(self): return heapq.heappop(self.h).edge
  def __getitem__(self, i): return self.h[i].edge

class EDGE_CLASS(enum.Enum):
  """
  class for edge tagging after classification
  """
  TAKEN=0
  NOTTAKEN=1
  BLOCKED=2

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
    self.nodes_attributes = ['layer', 'in_degree', 'out_degree', 'stp_from_entries', 'stp_to_da']
    self.edges_attributes = ['blockable', 'connected_entries', 'level_gap', 'class']
  
  # debug
  def graph_debug(self) -> None:
    """
    print out the basic informations
    """
    # debug information
    print("---------------DEBUG START---------------")
    print("\nTest Basic Information......")
    print("\nNodes: ", self.G.nodes(data=True))
    print("\nEdges ", self.G.edges(data=True))
    print("\nGraph: ", self.G.graph)
    print("\n----------------DEBUG END----------------")

  def torch_debug(self, data: Data) -> None:
    """
    debugger for torch data
    """
    # data structure
    print(f'Data structure:\n {data}')
    # # node feature
    # print(f'[torch_debug](Node feature)data.x:\n {data.x}')
    # # edge classes
    # print(f'[torch_debug](Edge Classes)data.y:\n {data.y}')
    # # edges
    # print(f'[torch_debug](Edges)data.edge_index:\n {data.edge_index}')
    # # edge features
    # print(f'[torch_debug](Edge Feature)data.edge_attr:\n {data.edge_attr}')
    # # block features
    # print(f'[torch_debug](Block Feature)data.new_x:\n {data.new_x}')

  def get_graph(self) -> nx.DiGraph():
    return self.G

  def get_blockable_edges(self) -> list:
    return self.blockable_edges

  def get_entries(self) -> list:
    return self.entries

  def draw_graph(self, G=None) -> None:
    """
    Should only be called for testing purpose on small graph
    """
    if G==None:
      G = self.G
    # configure drawing parameters
    edge_color = [G[u][v]['blockable']
                  for u, v in G.edges]  # draw according to blockable or not
    # determine position as multipartite graph
    # 'layer' is fixed here!
    pos = nx.multipartite_layout(G, subset_key="layer")
    # draw graph
    nx.draw(G, pos,
            with_labels=True,
            node_size=5,
            connectionstyle="arc3,rad=-0.2",
            width=1,
            edge_color=edge_color,
            #labels={k: k for k in range(sum(self.layer_sizes))},
            font_size=10)
    plt.show()
    pass

  def draw_grpah_after_remove_edges(self, remove_list: list) -> None: 
    pass

  # utilities
  def __delete_graph(self) -> None:
    """
    Should be called every time we finish a complete data [Round]
    """
    self.G.clear()

  def read_graph(self, path: str):
    """
      Read graph from .gml, need to config nx.draw() to recover the original layout
      But all of the data could be normally read and create as a new nx.Graph
      Read all .gmls under the folder one time!

      need to config :     
        self.G
        self.layer_sizes
        self.DA
        self.blockable_edges
        self.entries

      [Parameters]
        Read from where
        path -- should be path+filename, read graph one by one
      [Return]
        should be a list of G(nx graph)
    """
    self.G = nx.read_gml(path, label="label", destringizer=int)
    self.layer_sizes = self.G.graph['layer_sizes']
        # G_tmp = nx.read_gml(os.path.join(path, filename), label="label")
        # This part should not be delete untile config draw_after_read()
        # pos_tmp = nx.multipartite_layout(G_tmp, subset_key="layer")
        # nx.draw(G_tmp, pos_tmp,
        #         with_labels=True,
        #         node_size=5,
        #         connectionstyle="arc3,rad=-0.2",
        #         edge_color=[G_tmp[u][v]['blockable'] for u, v in G_tmp.edges],
        #         width=1,
        #         font_size=10)
        # print(os.path.join(path, filename))
        # print(G_tmp.nodes(data=True))
        # print(G_tmp.edges(data=True))
        # plt.show()

  def store_graph(self, type: str) -> None:
    """
      Store graph as file

      [Parameters]
        G -- graph
        type -- 'train' / 'test' / 'validation'
      [Return]
        None
    """ 
    name = f'./data/{type}/' + "Graph-" + uuid.uuid4().hex + ".gml"
    nx.write_gml(self.G, name)
    pass

  def add_new_attributes(self, target: str, attr: str, default_value: any) -> None:
    """
      add new attributes to each nodes / edges / graph

      [Parameters]
        G: graph
        target: "nodes" or "edges" or "graph"
        attr: new attribtue name
        default_value: default value for the attribtues (ONLY int for now)
      [Return]
        None
    """
    print('\nTest in add_new_attributes.....')
    if target == 'nodes':
      self.nodes_attributes.append(attr)
      for node in self.G.nodes:
        self.G.nodes[node][attr] = default_value
    elif target == 'edges':
      self.edges_attributes.append(attr)
      for edge in self.G.edges:
        self.G[edge[0]][edge[1]][attr] = default_value
    elif target == 'graph':
      attr_g = {attr : default_value}
      self.G.graph.update(attr_g)
    print(f'After adding {attr}: {getattr(self.G, target)}')
    return

  def edge_filter(self, attr: str, attr_val: any, G: nx.DiGraph = None, src: int = None, dst: int = None) -> list:
    """
    Fetch edge according to {attr:attr_val}
    
    [Parameters]
      G -- which graph
      attr -- attribtue name
      attr_val -- atttribute value you want to get
    [Return]
      filtered edges
    """
    if G == None:
      G = self.G
    if src != None and dst != None:
      return G[src][dst].get(attr, attr_val)
    # else
    edge_list = []
    for edge in G.edges:
      if G[edge[0]][edge[1]][attr] == attr_val:
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
    print("\nTest in struct_graph......")
    self.add_new_attributes('edges', 'connected_entries', 0)
    self.add_new_attributes('nodes', 'in_degree', 0)          # no need to set for now, will add to matrix directly
    self.add_new_attributes('nodes', 'out_degree', 0)         # no need to set for now, will add to matrix directly
    self.add_new_attributes('graph', 'layer_sizes', self.layer_sizes)
    self.add_new_attributes('edges', 'class', EDGE_CLASS.NOTTAKEN.value)
    self.add_new_attributes('edges', 'average_entry_to_here', 0)
    
    self.__set_entries()
    self.__set_blockable()
    # self.__set_connected_entries()

  # config node embedding
  def __set_entries(self):
    """
    set entry points
    """
    self.entries = self.__find_all_entries()
  
  def __set_blockable(self):
    """
    set blockable edges
    """
    self.blockable_edges = self.edge_filter('blockable', True)

  def __set_connected_entries(self):
    """
    set connected_entries
    set linkable edges (Here, we only calculate the blockable edges)
    """
    for edge in self.blockable_edges:
      self.G[edge[0]][edge[1]]['connected_entries'] = len(self.linkable_entries(edge))

  def __set_to_da(self, node: int):
    stp = nx.shortest_path(self.G, source=node, target=self.DA)
    self.G.nodes[node]['']
    pass

  def __find_all_entries(self) -> list:
    """
      Get all leaf nodes (should be randomized as any node in the graph)

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

  def graph_utility(self, sr_prob: float) -> tuple:
    """
    Calculate the utility(successful rate) for the whole graph
    Used to evaluate the effect of randomization for each graph (used for building(build the label) loss function)
    we try to minimize the sr for each graph
    we assume the possibilities getting each entry are exactly the same, we improve
    """
    total_successful_rate: float = 0.0
    stps = []
    successful_rate_list = self.__successful_rate(sr_prob)
    for entry in self.entries:
      try:
        stp = nx.shortest_path(self.G, source=entry, target=self.DA)
      except nx.NetworkXNoPath:
        print(f'No path between {entry} and {self.DA}')
        continue
      # reserve current stp
      stps.append(stp)
      # calculate sr
      cur_path_successful_rate = successful_rate_list[len(stp)-1]
      # update total sr
      # print("--",cur_path_successful_rate)
      total_successful_rate += cur_path_successful_rate

    # return average successful rate and current stps
    return (total_successful_rate/len(self.entries), stps)

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

  def __cut_strategy(self, budget: int, epoch: int) -> tuple:
    """
    1. Randomly pick blockable edges
    2. Delete picked blockable edges
    3. Find STP
    4. calculate successful rate(SR)
    :parameter
      budget -- budget in paper
      epoch -- how many times we do randomization
    :return
      best performance block choice
      best stp
    """
    if budget > len(self.blockable_edges):
      print("[WARNING]: budget should never larger than the number of blockable edges!")
      exit(1)
    # store status
    G_tmp = self.G.copy()
    worst_sr = 1.0
    worst_block_choices = []
    worst_stps = []
    for i in range(epoch):
      blocked_edges = random.sample(self.blockable_edges, budget)
      print("\nTest blocked edges: ", blocked_edges)
      self.G.remove_edges_from(blocked_edges)
      total_sr, stps = self.graph_utility(0.6)
      print("SR after blocking blockable edges: ", total_sr)
      # worst_sr = min(total_sr, worst_sr)
      if worst_sr > total_sr:
        worst_sr = total_sr
        worst_block_choices = blocked_edges
        worst_stps = stps
      # recover status
      self.G = G_tmp.copy()
    print(f'\nBest performance: {worst_sr}')
    print(f'Best performance blocked edges: {worst_block_choices}')
    return (worst_block_choices, worst_stps)

  def edge_classification_sample(self):
    """
    config __cut_strategy
    all edges initialized as NOT_TAKEN
    """
    # always cut .5 of all blockable edges
    num_of_cutted_edges = (int)(len(self.blockable_edges)/2)
    blocked, taken = self.__cut_strategy(num_of_cutted_edges, 1000)
    for edge in blocked:
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.BLOCKED.value
    for stp in taken:
      for i in range(len(stp)-1):
        self.G[stp[i]][stp[i+1]]['class'] = EDGE_CLASS.TAKEN.value
    
    # print("\nTest on edge class tag: ",
    #       nx.get_edge_attributes(self.G, 'class').values())
    pass

  # to torch
  def networkx_to_torch(self) -> Data:
    """
      Transfer netowrkx to fit torch geo
    """
    # [Build Node Embedding]
    layer_list = list(nx.get_node_attributes(self.G, 'layer').values())
    in_degree_list = [val[1] for val in list(self.G.in_degree)]
    out_degree_list = [val[1] for val in list(self.G.out_degree)]
    node_matrix = np.column_stack((layer_list, in_degree_list, out_degree_list))

    # [Build Edge Embedding] all tensors are aligned
    blockable_list = []
    connected_entries_list = []
    level_gap_list = []
    class_list = []
    edges_list = [[],[]]
    for u, v in self.G.edges:
      edges_list[0].append(int(u))
      edges_list[1].append(int(v))
      blockable_list.append(self.G[u][v]['blockable'])
      connected_entries_list.append(self.G[u][v]['connected_entries'])
      level_gap_list.append(self.G[u][v]['level_gap'])
      class_list.append(self.G[u][v]['class'])
    # [blockable, connected_entires, level_gap]
    edge_matrix = np.column_stack(
        (blockable_list, connected_entries_list, level_gap_list))

    # [Build Edge Label]
    # print(list(nx.get_edge_attributes(self.G, 'class').values()))
    # for loop on nodes_attributes & edges_attributes respectively
    #   - nx.get_[node/edge]_attributes(self.G, '{attribute_name}').values()

    # [Convert networkx to torch]
    data = from_networkx(self.G)
    data.x = torch.tensor(node_matrix, dtype=torch.float)         # node feature matrix
    data.y = torch.tensor(class_list)                             # edge classification
    data.edge_index = torch.tensor(edges_list)                    # edges
    data.edge_attr = torch.tensor(edge_matrix, dtype=torch.float) # edge feature matrix
    # after buidling up virutal block
    data.new_x = []                                               # block embedding (edge attr + src_node_attr + tar_node_attr)
    data.new_edge_index = []                                      # virtual edges

    # build edge embedding
    for i in range(data.edge_index.shape[1]):
      from_node = edges_list[0][i]
      to_node = edges_list[1][i]
      # add current posisiton as an attribute to original graph (it should not be held)
      self.G[from_node][to_node]['ind_in_edge_list'] = i
      data.new_x.append(torch.cat((data.edge_attr[i], data.x[from_node], data.x[to_node]), -1))
    data.new_x = torch.stack((data.new_x))

    new_edge_index = [[],[]]
    # data.new_edge_index = []
    for i in range(data.edge_index.shape[1]):
      to_node = edges_list[1][i] # get target node
      # loop over all `connected edges`
      out_edges = list(self.G.out_edges(to_node))
      if len(out_edges) == 0:
        new_edge_index[0].append(i)
        new_edge_index[1].append(i)
        continue
      for edge in out_edges:
        # print(f'from {(edges_list[0][i],edges_list[1][i])}(index: {self.G[edges_list[0][i]][edges_list[1][i]]["ind_in_edge_list"]}) to {edge}(index: {self.G[edge[0]][edge[1]]["ind_in_edge_list"]})')
        # from which position in edge_list
        from_ = i
        # to which posistion in edge_list
        to_ = self.G[edge[0]][edge[1]]['ind_in_edge_list']
        new_edge_index[0].append(from_)
        new_edge_index[0].append(to_)
        new_edge_index[1].append(to_)
        new_edge_index[1].append(from_)
    # make new_edge_index bidirectional edges
    # tmp = copy.deepcopy(new_edge_index[0])
    # new_edge_index[0] += new_edge_index[1]
    # new_edge_index[1] += tmp
    # print(new_edge_index)
    data.new_edge_index = torch.tensor(new_edge_index)
    return data

  # algorithms (previous algorithm 1 & 2 need to be retrieved from former commit)
  def algorithm_1(self, G: nx.DiGraph=None) -> None:
    """
    algorithm on simple tree O(blogn + n)
    using fib heap to choose the worthest blockable edges
    use the rest of the budget to block them
    """
    if G == None:
      G = self.G
    for (node, value) in G.out_degree():
      if value > 1:
        print("this is not a simple tree.")
        return
    # get successful rate list for each layer
    sr_list = self.__successful_rate(0.9)
    print(sr_list)
    # walk through every path from entry to DA
    for entry in self.entries:
      try:
        G.degree(entry)
      except:
        print("[algorithm 1] This node does not exit in this graph, it has been deleted.")
        continue
      dst = entry
      print(f'new entry: {dst}')
      distance = 0
      while dst != self.DA and dst != None:
        src = dst
        if (len(list(G.neighbors(dst))) == 0):
          break
        else:
          dst = list(G.neighbors(dst))[0]
          distance+=1
        print(f'--walking on: {dst}')
        G[src][dst]['average_entry_to_here'] += distance
        G[src][dst]['connected_entries'] += 1
    print(G.edges(data=True))
    # put all blockable edges into fibheap and sort them according to its benefit
    maxh = MaxHeap(G, 'average_entry_to_here')
    for edge in self.edge_filter('blockable', True, G=G):
      G[edge[0]][edge[1]]['average_entry_to_here'] = G[edge[0]][edge[1]]['average_entry_to_here'] / \
          G[edge[0]][edge[1]]['connected_entries']
      maxh.heappush(edge)
    # using rest of the budget pick the worthiest blockable edges
    while maxh.__len__() != 0:
      tmp = maxh.heappop()
      print(tmp, G[tmp[0]][tmp[1]]['average_entry_to_here'])
    pass

  def algorithm_5(self) -> None:
    """
    1. first time classification for cutting the graph to simple tree
    2. second time classification for cutting the rest edges
    3. record the blocked edges from Step 1 & 2, reflect them back to the original graph
    4. run STP algorithm and calculate the performance
    Here, the performance is calculate using graph_utility

    Then we have a NN:
     input: one edge embedding
     output: possibility on classification, one-hot or just classification label
     loss function: the gap between the result from Step 4 and original classification
    """
    print('\nTest in algorithm_5.....')
    print('Before walking through...', nx.get_edge_attributes(self.G, "class"))
    # [Step 1:] cut the graph to simple tree
    picked_nodes:set = set({}) # taken nodes
    picked_edges = []          # taken edges

    # iter over all entry points
    # for each entry point, algorithm keeps going until touch DA or no outgoing edge
    # randomly pick any forwarding edge from the current node, it should iter over all of them
    for entry in self.entries:
      dst = entry # setup entry node
      picked_nodes.add(dst)
      while dst != self.DA: # before touching the DA
        # randomly taken one from out_degree
        out_edges = list(self.G.edges(dst)) # get out edges
        if len(out_edges) == 0:
          break
        taken = random.choice(out_edges)    # randomly pick one
        # make sure there is only one out-edge is taken for each node
        # label the taken edge as TAKEN
        for o_e in out_edges:
          if self.G[o_e[0]][o_e[1]]['class'] == EDGE_CLASS.TAKEN.value:
            taken = o_e
            break
        self.G[taken[0]][taken[1]]['class']=EDGE_CLASS.TAKEN.value
        dst = taken[1] # update dst
        picked_nodes.add(dst)
        #print(taken)
    picked_edges = self.edge_filter('class', EDGE_CLASS.TAKEN.value)
    print('After walking through...', nx.get_edge_attributes(self.G, "class"))
    print('Taken edges: ', picked_edges)
    print('Taken nodes: ', picked_nodes)
    simple_tree = self.G.edge_subgraph(picked_edges).copy() # has to be deep copy
    print(simple_tree.nodes(data=True))
    print(simple_tree.edges(data=True))
    # self.draw_graph(simple_tree)

    # [Step 2]: simple tree, use rest of the budget to pick blockable edges
    self.algorithm_1(simple_tree)
      












