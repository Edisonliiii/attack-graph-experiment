# essential
from networkx.classes import graph
import sys
sys.path.insert(1, '../utility')
sys.path.insert(1, './')
# networkx
import networkx as nx
import matplotlib.pyplot as plt
# customized
from graph_generator import GraphGenerator
import utility as ut

layer_sizes = [4,3,2,1]
graph_gen = GraphGenerator(layer_sizes)
graph_gen.struct_graph(nonjump_percentage=0.8, outgoing_lower_bound=3,
                       outgoing_upper_bound=5, blockable_percentage=0.4)
graph_gen.draw_graph()
graph_gen.graph_debug()

leaves = graph_gen.find_all_leaves()
print(f'All leaves are: {leaves}')
linkable_entries = graph_gen.linkable_entries((8,9), leaves)
print(f'All linkable entries are: {linkable_entries}')

data = graph_gen.networkx_to_torch()
print("\nTest networkx_to_torch......")
print(f'First torch data: {data}')
print(f'Edges after torch: {data.edge_index}')
print(f'[Torch]: [num_nodes, num_node_features] {data.x}')

graph_gen.graph_debug()









