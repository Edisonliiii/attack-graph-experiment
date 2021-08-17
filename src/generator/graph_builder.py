# essential
from networkx.classes import graph
import sys
import time
from networkx.readwrite import graph6
sys.path.insert(1, '../utility')
sys.path.insert(1, './')
# networkx
import networkx as nx
import matplotlib.pyplot as plt
# customized
from graph_generator import GraphGenerator
import utility as ut

layer_sizes = [3,624,532,412,31,22,1]
graph_gen = GraphGenerator(layer_sizes)
graph_gen.struct_graph(nonjump_percentage=0.6, outgoing_lower_bound=5,
                       outgoing_upper_bound=6, blockable_percentage=0.3)
# graph_gen.draw_graph()
# graph_gen.graph_debug()

# print(f'All leaves are: {graph_gen.get_entries()}')
# linkable_entries = graph_gen.linkable_entries((8,9))
# print(f'All linkable entries are: {linkable_entries}')

# data = graph_gen.networkx_to_torch()
# print("\nTest networkx_to_torch......")
# print(f'First torch data: {data}')
# print(f'Edges after torch: {data.edge_index}')
# print(f'[Torch]: [num_nodes, num_node_features] {data.x}')

#graph_gen.graph_debug()

# print("\nTest blockable_edges")
# print(graph_gen.get_blockable_edges())

# graph_gen.cut_strategy(3, 100)
graph_gen.edge_classification_sample()
graph_gen.graph_debug()
graph_gen.networkx_to_torch()






