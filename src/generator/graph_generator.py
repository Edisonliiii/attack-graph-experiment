# essential
import decimal
import enum
import copy
import heapq
import math
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
from networkx.algorithms.components.weakly_connected import weakly_connected_components
from networkx.algorithms.cuts import cut_size
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
    if self.G.has_edge(self.edge[0], self.edge[1]) and self.G.has_edge(other.edge[0], other.edge[1]):
      return self.G[self.edge[0]][self.edge[1]][self.attr] > self.G[other.edge[0]][other.edge[1]][self.attr]
  def __eq__(self, other):
    if self.G.has_edge(self.edge[0], self.edge[1]) and self.G.has_edge(other.edge[0], other.edge[1]):
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
  def __init__(self, layer_sizes: list = None, budget: int = None, sr: float = 0.9) -> None:
    """
    [Round]
    randomization -> picked edges -> get ground truth(classification) -> make it as one dataset
    dataset should have both original graph and picked edges classification

    Do the process one by one
    """
    self.G = nx.DiGraph()
    self.ori_G = nx.DiGraph()                  # fixed after cutting irrelevant roots
    self.budget = budget                       # fixed
    self.layer_sizes = layer_sizes
    self.DA = sum(layer_sizes)-1
    self.SR: decimal = sr                      # successful rate(SR), we suppose all edges have the same SR

    self.entries = []          # fixed
    self.taken = {}            # need to be updated
    self.not_taken = {}        # need to be updated
    self.blockable = []        # fixed
    self.blocked = []          # chosen blocked blockable edges
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
    print("\nNumber of nodes: ", self.G.number_of_nodes())
    print("\nNumber of edges: ", self.G.number_of_edges())
    print("\nNodes: ", self.G.nodes(data=True))
    print("\nEdges ", self.G.edges(data=True))
    print("\nGraph: ", self.G.graph)
    print(f'\n******* G total number of NOTTAKEN edges {len(self.edge_filter("class", EDGE_CLASS.NOTTAKEN.value))}')
    print(f'\n******* G total number of TAKEN edges {len(self.edge_filter("class", EDGE_CLASS.TAKEN.value))}')
    print(f'\n******* G total number of blocked_edges {len(self.blocked)}')
    print("\n----------------DEBUG END----------------")

  def torch_debug(self, data: Data) -> None:
    """
    debugger for torch data
    """
    # data structure
    print(f'Data structure:\n {data}')
    # node feature
    print(f'[torch_debug](Node feature)data.x:\n {data.x}')
    # edge classes
    print(f'[torch_debug](Edge Classes)data.y:\n {data.y}')
    # edges
    print(f'[torch_debug](Edges)data.edge_index:\n {data.edge_index}')
    # edge features
    print(f'[torch_debug](Edge Feature)data.edge_attr:\n {data.edge_attr}')
    # block features
    print(f'[torch_debug](Block Feature)data.new_x:\n {data.new_x}')

  def get_graph(self) -> nx.DiGraph():
    return self.G

  def get_blockable_edges(self) -> list:
    return self.blockable

  def get_entries(self) -> list:
    return self.entries

  def draw_graph(self, attr: str='blockable', edge_type: int = -1, G = None, edge_list: list=None) -> None:
    """
    Should only be called for testing purpose on small graph
    :parameter
      edge_type: 'all'(-1) / 'taken' / 'blocked' / 'not_taken'
      attr: 'blockable' / 'class' /
    """
    if G==None:
      G = self.G
    if attr == 'blockable':
      el = list(G.edges())
    elif attr == 'class' and edge_type != -1:
      el = list(self.edge_filter(attr, edge_type))
    elif edge_list != None:
      el = edge_list

    # configure drawing parameters
    edge_color = [G[u][v]['blockable']
                  for u, v in el]  # draw according to blockable or not
    # determine position as multipartite graph
    # 'layer' is fixed here!
    pos = nx.multipartite_layout(G, subset_key="layer")
    # draw graph
    nx.draw(G, pos,
            with_labels=True,
            node_size=5,
            connectionstyle="arc3,rad=-0.2",
            width=1,
            edge_color = edge_color,
            edgelist = el,
            # labels={k: k for k in range(sum(self.layer_sizes))},
            font_size=10)
    plt.show()

  def draw_from_list(self, edge_list: list):
    pass

  # utilities
  def __delete_graph(self) -> None:
    """
    Should be called every time we finish a complete data [Round]
    """
    self.G.clear()

  def __compare_graph(self) -> None:
    """
    if two graph is exactly the same
    it should give the output like 
    DiGraph with 0 nodes and 0 edges

    Here, we only compare ori_G and G
    """
    # check nodes
    R = self.ori_G.copy()
    R.remove_nodes_from(n for n in self.ori_G if n in self.G)
    # check edges
    # check_edges = len([(u,v,w) for (u,v,w) in self.ori_G.edges(data=True) if (u,v,w) in self.G.edges(data=True)])
    for (u,v,w) in self.ori_G.edges(data=True):
      if (u,v,w) not in self.G.edges(data=True):
        print(
            f'Here is the difference:\n ori({u},{v}): {self.ori_G[u][v]}; G({u},{v}): {self.G[u][v]}')
        exit(0)
    print(f'Edge test pass!')
        

  def read_graph(self, path: str):
    """
      Read graph from .gml, need to config nx.draw() to recover the original layout
      But all of the data could be normally read and create as a new nx.Graph
      Read all .gmls under the folder one time!

      need to config :     
        self.G
        self.layer_sizes
        self.DA
        self.blockable
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
    # print('\nTest in add_new_attributes.....')
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
    # print(f'After adding {attr}: {getattr(self.G, target)}')
    return

  def fetch_edges_from_ori_graph(self, edges: list) -> list:
    """
    return all edge with attributes from ori graph
    """
    ori_edge = []
    for edge in edges:
      ori_edge.append((edge[0], edge[1], self.ori_G[edge[0]][edge[1]]))
    return ori_edge

  def edge_status_change(self, edge: tuple, _from: str, _to: str) -> None:
    if _from == 'taken':
      self.taken.remove(edge)
      self.not_taken.append(edge)
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.NOTTAKEN.value
    elif _from == 'not_taken':
      self.not_taken.remove(edge)
      self.taken.append(edge)
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.TAKEN.value
    elif _from == 'blocked':
      self.blocked.remove(edge)
      self.blockable.append(edge)
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.NOTTAKEN.value
      self.G[edge[0]][edge[1]]['blockable'] = True
    elif _from == 'blockable':
      self.blockable.remove(edge)
      self.blocked.append(edge)
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.BLOCKED.value
      self.G[edge[0]][edge[1]]['blockable'] = False


  def edge_filter(self, *attr_pairs:list, G: nx.DiGraph = None, src: int = None, dst: int = None) -> list:
    """
    Fetch edge according to {attr:attr_val}
    
    [Parameters]
      G -- which graph
      attr -- attribtue name
      attr_val -- atttribute value you want to get
    [Return]
      filtered edges
    """
    # check the number of (att, attr_val)
    if len(attr_pairs) % 2 != 0:
      print("Bad filtering parameters!")
      return
    # check graph
    if G == None:
      G = self.G
    # if src != None and dst != None:
    #   return G[src][dst].get(attr, attr_val)
    # else
    edge_list = []
    for edge in G.edges:
      # check all condition pairs one by one
      checker = True
      for i in range(0, len(attr_pairs), 2):
        checker &= (G[edge[0]][edge[1]][attr_pairs[i]] == attr_pairs[i+1])
        if checker == False: # if there is only one False, break
          break
      # only if all conditions met
      if checker == True:
        edge_list.append((edge[0], edge[1]))
    return edge_list

  def edge_setter(self, edge_list: list, attr:str ,new_val: any):
    """
    set a new value for edge in the list
    the edges should be filtered by edge_filter
    """
    for edge in edge_list:
      self.G[edge[0]][edge[1]][attr] = new_val

  def struct_graph(self,
                   nonjump_percentage: float = None,
                   blockable_percentage: float = None,
                   outgoing_lower_bound: int = None,
                   outgoing_upper_bound: int = None) -> None:
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
    if nonjump_percentage != None:
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
    self.add_new_attributes('edges', 'valid', False)          # if the edge is relevent to our problem or not
    self.add_new_attributes('edges', 'connected_entries', 0)
    self.add_new_attributes('nodes', 'in_degree', 0)          # no need to set for now, will add to matrix directly
    self.add_new_attributes('nodes', 'out_degree', 0)         # no need to set for now, will add to matrix directly
    self.add_new_attributes('graph', 'layer_sizes', self.layer_sizes)
    self.add_new_attributes('edges', 'class', EDGE_CLASS.NOTTAKEN.value)
    self.add_new_attributes('edges', 'average_sr', 0)
    
    # self.__set_entries()
    # self.__set_blockable()
    # self.__set_connected_entries()
    # self.ori_G = self.G.copy()

  # config node embedding
  def __set_taken(self):
    self.taken = self.edge_filter('class', EDGE_CLASS.TAKEN.value)
  
  def __set_not_taken(self):
    self.not_taken = self.edge_filter('class', EDGE_CLASS.NOTTAKEN.value)

  def __set_entries(self):
    """
    set entry points
    """
    self.entries = self.__find_all_entries()
  
  def __set_blockable(self):
    """
    set blockable edges
    """
    self.blockable = self.edge_filter('blockable', True, G=self.ori_G)

  def __set_connected_entries(self):
    """
    set connected_entries
    set linkable edges (Here, we only calculate the blockable edges)
    """
    for edge in self.blockable:
      self.G[edge[0]][edge[1]]['connected_entries'] = len(self.linkable_entries(edge))

  def __set_to_da(self, node: int):
    stp = nx.shortest_path(self.G, source=node, target=self.DA)
    self.G.nodes[node]['']
    pass

  # utility algorithms
  def __find_all_entries(self) -> list:
    """
      Get all leaf nodes (should be randomized as any node in the graph)

      [Parameters]
        G: nx.Graph -- graph
      [Return]
        leaves as list
    """
    return [v for v, d in self.G.in_degree() if d == 0]

  def __find_all_roots(self) -> list:
    """
    return all roots including DA
    should only use for preprocessing
    """
    return [n for n, d in self.G.out_degree() if d == 0]

  def __delete_all_islands(self) -> None:
    self.G.remove_nodes_from(list(nx.isolates(self.G)))

  def __cut_from_root(self):
    """
    only keep the edges could reach DA
    all other edges don't make any sense to keep
    """
    # cut all other roots except DA
    roots_list = self.__find_all_roots()
    # print("\nTest from cut_from_root: All roots are: ", roots_list)
    for root in roots_list:
      if root != self.DA:
        tmp = list(nx.edge_dfs(self.G, root, orientation='reverse'))
        for edge in tmp:
          self.G[edge[0]][edge[1]]['valid'] = False
      else:
        # get all relevant edges and label as takenable edges
        tmp = list(nx.edge_dfs(self.G, root, orientation='reverse'))
        # print(f'\nTest from cut_from_root: the number of reachable edges: {len(tmp)}')
        # print(f'Test from cut_from_root: root is {root} ', tmp)
        for edge in tmp:
          self.G[edge[0]][edge[1]]['valid'] = True
        
        # cut all other edges
        invalid_edges = self.edge_filter('valid', False)
        self.G.remove_edges_from(invalid_edges)
        
        # delete all isolated nodes
        self.__delete_all_islands()

        # [DEBUG]
        print(self.__find_all_roots()) # should only be one root for now
        # self.draw_graph()
        return

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

  def graph_utility(self, sr_prob: float=None, G: nx.Graph=None) -> tuple:
    """
    Calculate the utility(successful rate) for the whole graph
    Used to evaluate the effect of randomization for each graph (used for building(build the label) loss function)
    we try to minimize the sr for each graph
    we assume the possibilities getting each entry are exactly the same, we improve
    """
    if G==None:
      G = self.G
    if sr_prob == None:
      sr_prob = self.SR
    total_successful_rate: float = 0.0
    stps = []
    successful_rate_list = self.__successful_rate(sr_prob)
    for entry in self.entries:
      try:
        stp = nx.shortest_path(G, source=entry, target=self.DA)
        stp = [[i,j] for i,j in zip(stp, stp[1:])]
      except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        # print(f'No path between {entry} and {self.DA}')
        # print('Or')
        # print(f'Node {entry} has been deleted!')
        continue
      # reserve current stp
      # print('\nTest in graph_utility: ==== ', stp)
      stps.append(stp)
      # calculate sr
      cur_path_successful_rate = successful_rate_list[len(stp)]
      # update total sr
      # print("--",cur_path_successful_rate)
      total_successful_rate += cur_path_successful_rate
    # return average successful rate and current stps
    return (total_successful_rate/len(self.entries), stps)

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
    if budget > len(self.blockable):
      print("[WARNING]: budget should never larger than the number of blockable edges!")
      exit(1)
    # store status
    G_tmp = self.G.copy()
    worst_sr = 1.0
    worst_block_choices = []
    worst_stps = []
    for i in range(epoch):
      blocked_edges = random.sample(self.blockable, budget)
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
    num_of_cutted_edges = (int)(len(self.blockable)/2)
    blocked, taken = self.__cut_strategy(num_of_cutted_edges, 1000)
    for edge in blocked:
      self.G[edge[0]][edge[1]]['class'] = EDGE_CLASS.BLOCKED.value
    for stp in taken:
      for i in range(len(stp)-1):
        self.G[stp[i]][stp[i+1]]['class'] = EDGE_CLASS.TAKEN.value

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
  def algorithm_1(self, budget: int, G: nx.DiGraph=None) -> None:
    """
    algorithm on simple tree O(blogn + n)
    using fib heap to choose the worthest blockable edges
    use the rest of the budget to block them

    Here, we use self.entries, because it has been updated and
    it is aligned with simple tree

    will update self.blocked
    :parameter
      budget: budget left from the Step 1(first classification)
      G: cutted simple tree
    """
    if G == None:
      G = self.G
    # print(f'simple tree debug: self.blockable {G.edges(data=True)}')
    for (node, value) in G.out_degree():
      if value > 1:
        print("this is not a simple tree.")
        return
    # get successful rate list for each layer
    # sr_list = self.__successful_rate(0.9)
    # print(sr_list)
    
    # [how to create a in-memory subgraph as GraphGenerator]
    # tmp = GraphGenerator(layer_sizes=self.layer_sizes)
    # tmp.G = G
    # tmp.struct_graph()
    # print(f'\nTesting in algorithm_1: {tmp.G.edges(data=True)}')
    # exit(0)
    
    # walk through every path from entry to DA
    for entry in self.entries:
      try:
        G.degree(entry)
      except:
        # print("[algorithm 1] This node does not exit in this graph, it has been deleted.")
        continue
      dst = entry
      # print(f'new entry: {dst}')
      distance = 0
      # DFS walk through the current path until meet stopping condition
      while dst != self.DA and dst != None:
        src = dst
        if (len(list(G.neighbors(dst))) == 0):
          break
        else:
          dst = list(G.neighbors(dst))[0] # jump to next node
          distance+=1
        # print(f'--walking on: {dst}')
        G[src][dst]['average_sr'] += distance
        G[src][dst]['connected_entries'] += 1
    # put all blockable edges into fibheap and sort them according to its benefit
    # choose which blockable edge to block
    maxh = MaxHeap(G, 'average_sr')
    for edge in self.edge_filter('blockable', True, G=G):
      G[edge[0]][edge[1]]['average_sr'] = decimal.Decimal(G[edge[0]][edge[1]]['connected_entries']) * \
          decimal.Decimal(1/self.SR) ** decimal.Decimal(
              G[edge[0]][edge[1]]['average_sr'] / G[edge[0]][edge[1]]['connected_entries']).ln() / decimal.Decimal(len(self.entries))
      maxh.heappush(edge)
    # using rest of the budget pick the worthiest blockable edges
    while maxh.__len__() != 0 and budget >= 0:
      budget-=1
      worthiest_edge = maxh.heappop()
      if G.has_edge(worthiest_edge[0], worthiest_edge[1]) == False:
        continue
      self.blockable.remove(worthiest_edge)
      self.blocked.append(worthiest_edge)
      # dfs delete on simple_tree
      cut_branch = list(nx.edge_dfs(
          G, worthiest_edge[0], orientation='reverse'))
      cut_branch.append(worthiest_edge)
      G.remove_edges_from(cut_branch)

  def algorithm_5(self) -> None:
    """
    0. heuristic: cut off all non-relevant nodes and edges
       for example, out_degree = 0, and the node is not DA
    1. first time classification for cutting the graph to simple tree
       •  for the first classification, we randomize the classification result as:
            1.one node could only TAKEN one out-degree, therefore, all other out-degrees are actually NOT_TAKEN
              -- one as TAKEN
              -- all other as NOT TAKEN
              -- leave blockable edges for now
            2. after finish linking all linkable entries to DA
               we randomly spend a random number of budget on those NOT TAKEN blockable edges
               Therefore, we might left a part of the budget for Step 2
               Or maybe, we got nothing left
          At the end of Step 1, there must be a simple tree, or branches out there, we can
          simply delete all of those branches, because they don't have out-degree from the
          beginning

          The number of entries might be changed.(zero out-degree)
        
    2. second time classification for cutting the rest edges
        • picked worthiest blockable edge to cut(frontiers) using the rest of the budget
    3. record the blocked edges from Step 1 & 2, reflect them back to the original graph
    4. run STP algorithm and calculate the performance

    Here, the performance is calculate using graph_utility

    Then we have a NN:
     input: one edge embedding
     output: possibility on classification, one-hot or just classification label
     loss function: the gap between the result from Step 4 and original classification
    """
    # print('\nTest in algorithm_5.....')
    # print('Before walking through...', nx.get_edge_attributes(self.G, "class"))
    # [Step 0:] preprocessing, 
    #           cut all untakenable edges
    #           cut all isolated nodes
    #           directly cut from self.G
    self.__cut_from_root()
    self.ori_G = self.G.copy()
    # [Step 1:] cut the graph to simple tree
    #           make a subgraph deep copy from self.G
    #           self.G not changed on this step
    self.__set_entries()         # update left entries
    self.__set_blockable()
    # print(f'blockable edges in original graph: {self.blockable}')
    # iter over all entry points
    # for each entry point, DFS to DA
    # randomly pick any forwarding edge from the current node, it should iter over all of them
    for entry in self.entries:
      dst = entry # setup entry node
      while dst != self.DA: # DFS reaching DA
        # randomly taken one from out_degree
        out_edges = list(self.G.edges(dst)) # get out edges
        taken = random.choice(out_edges)    # randomly pick one
        # make sure there is only one out-edge is taken for each node
        # label the taken edge as TAKEN
        # ! This for loop has to be kept to ensure no two or more TAKEN outedges from one node
        for o_e in out_edges:
          if self.G[o_e[0]][o_e[1]]['class'] == EDGE_CLASS.TAKEN.value:
            taken = o_e
            break
        self.G[taken[0]][taken[1]]['class']=EDGE_CLASS.TAKEN.value
        dst = taken[1]      # update dst
    self.__set_taken()      # update self.taken
    self.__set_not_taken()  # update self.taken
    simple_tree = self.G.edge_subgraph(self.taken).copy() # has to be deep copy
    
    # [DEBUG]
    print('\nOriginal graph after Step 1: ')
    print(f'All blockable: {self.blockable}')
    # self.draw_graph()
    print('Test utility on ori graph: ===== ', self.graph_utility()[0])

    print('\nSimple cutted tree after Step 1: ')
    # self.draw_graph(G=simple_tree)
    print('Test utility on simple tree: ===== ',
      self.graph_utility(G=simple_tree)[0])
    
    # get taken blockable edges from simple tree
    taken_blockable_condition = ['class', EDGE_CLASS.TAKEN.value, 'blockable', True]
    taken_blockable_edges = self.edge_filter(*taken_blockable_condition, G=simple_tree)
    
    # [Step 2]: simple tree, use rest of the budget to pick blockable edges (all in for algorithm_1)
    budget_cost_on_second_classification = min(len(taken_blockable_edges), self.budget)
    
    # run algorithm_1, self.blocked will be updated
    self.algorithm_1(
        budget=budget_cost_on_second_classification, G=simple_tree)
    self.edge_setter(self.taken, 'class',
                     EDGE_CLASS.NOTTAKEN.value)  # recover it
    self.__compare_graph()
    rest_budget = self.budget - len(self.blocked)
    if rest_budget > 0:
      for edge in random.sample(self.blockable, min(rest_budget,len(self.blockable))):
        self.blocked.append(edge)
        self.blockable.remove(edge)
    self.G.remove_edges_from(self.blocked) # remove all blocked blockable edges chosen by algorithm 1
    performance, stp_after_algorithm1 = self.graph_utility()  # check performance
    for stp in stp_after_algorithm1:  # label best performance as taken
      self.edge_setter(stp, 'class', EDGE_CLASS.TAKEN.value)
    self.__set_taken()
    self.__set_not_taken()
    print(f'\n//////// Final utility: {performance}')

    # recover status
    # 1. recover remove blockable edges
    # 2. recover blockable list & clear blocked list
    # 3. recover taken edges (after two classification we have a attack graph for attack (stp))
    self.G.add_edges_from(self.fetch_edges_from_ori_graph(
        self.blocked))
    self.blockable += self.blocked
    # self.blocked.clear()
    # self.__set_taken()
    # self.__set_not_taken()
    self.edge_setter(self.taken, 'class',
                     EDGE_CLASS.NOTTAKEN.value)
    # print(f'num of edges: {self.ori_G.number_of_edges()}')
    # print(len(self.taken) + len(self.not_taken))
    # print(self.taken)
    # print(self.not_taken)

    # all good till here
    self.draw_graph(G=self.ori_G)
    for i in range(30):
      self.__compare_graph()
      print(f'Taken edges: {self.taken}')
      print(f'Blocked edges: {self.blocked}')
      self.draw_graph(attr='class', edge_list=self.taken)
      self.edge_tremble()


  def edge_tremble(self) -> None:
    """
    update edge after every turn of algorithm_5
    1. pick any one fixed edge from simple tree (after the first classification)
    2. change it's classification, for example, taken -> not_taken / blocked(if blockable),
                                                not taken -> taken
                                                blocked -> unblocked
    """
    # new edge tremble round
    self.blocked.clear()

    simple_tree = self.G.edge_subgraph(self.taken).copy()
    taken_blockable_condition = [
        'class', EDGE_CLASS.TAKEN.value, 'blockable', True]
    taken_blockable_edges = self.edge_filter(
        *taken_blockable_condition, G=simple_tree)
    budget_cost_on_second_classification = min(
        len(taken_blockable_edges), self.budget)
    self.algorithm_1(
        budget=budget_cost_on_second_classification, G=simple_tree)
    self.edge_setter(self.taken, 'class',
                     EDGE_CLASS.NOTTAKEN.value)  # recover it
    rest_budget = self.budget - len(self.blocked)
    if rest_budget > 0:
      for edge in random.sample(self.blockable, min(rest_budget, len(self.blockable))):
        self.blocked.append(edge)
        self.blockable.remove(edge)
    # remove all blocked blockable edges chosen by algorithm 1
    self.G.remove_edges_from(self.blocked)
    performance, stp_after_algorithm1 = self.graph_utility()  # check performance
    for stp in stp_after_algorithm1:  # label best performance as taken
      self.edge_setter(stp, 'class', EDGE_CLASS.TAKEN.value)
    self.__set_taken()
    self.__set_not_taken()
    print(f'\n//////// Graph utility after new edge trumbling: {performance}')
    self.G.add_edges_from(self.fetch_edges_from_ori_graph(
        self.blocked))
    print(self.blocked)
    self.blockable += self.blocked
    # self.blocked.clear()
    self.edge_setter(self.taken, 'class',
                     EDGE_CLASS.NOTTAKEN.value)










