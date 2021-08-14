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

layer_sizes = [4,3,2,1] #[5,4,3,4,6,3,2,1]
ut.DA = sum(layer_sizes)-1
G = ut.struct_graph(*layer_sizes, nonjump_percentage=0.8, outgoing_lower_bound=3, 
                outgoing_upper_bound=5, blockable_percentage=0.4)

print("----DEBUG----")
# configure drawing parameters
edge_color = [G[u][v]['blockable'] for u,v in G.edges] # draw according to blockable or not
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

ut.graph_debug(G)
# print("\nTest simple tree algorithm......")
# ut.algorithm_tree(G, len(layer_sizes))

print("\nTest store_graph and read_graph......")
# store graph
plt.show()
# ut.store_graph(G)
# ut.read_graph()

print("\nTest linkable_entries......")
leaves = ut.find_all_leaves(G)
print("All linkable entries are: ", ut.linkable_entries(G, (8,9), leaves))

data = ut.networkx_to_torch(G)
print("\nTest networkx_to_torch......")
print("First torch data: ", data)
# print(data.__getitem__('blockable'))
#print("Nodes after torch: ", data.nodes)
print("Edges after torch: ", data.edge_index)
print(f'[Torch]: [num_nodes, num_node_features] {data.x}')
ut.graph_debug(G)


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









