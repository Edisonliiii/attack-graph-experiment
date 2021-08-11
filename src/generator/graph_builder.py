# essential
import sys
sys.path.insert(1, '../utility')
# networkx
import networkx as nx
import matplotlib.pyplot as plt
# customized
import utility as ut

layer_sizes = [4,3,2,1] #[5,4,3,4,6,3,2,1]
G = ut.struct_graph(*layer_sizes, nonjump_percentage=0.8, outgoing_lower_bound=2, 
                outgoing_upper_bound=3, blockable_percentage=0.2)

print("----DEBUG----")
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
print("\nTest Basic Information......")
print("Nodes: ", G.nodes(data=True))
print("Edges ", G.edges(data=True))
print("Node attributes: ", nx.get_node_attributes(G,"layer"))
print("Edge attributes: ", nx.get_edge_attributes(G,"blockable"))
print("Adjacency Matrix: ", nx.to_dict_of_dicts(G))
print("Graph info: ", nx.info(G))
ut.algorithm_tree(G, len(layer_sizes))

print("\nTest store_graph and read_graph......")
# store graph
plt.show()
#ut.store_graph(G)
#ut.read_graph()

print("\nTest add_new_attributes......")
ut.add_new_attributes(G, 'nodes', 'hello_node', 0)
print("Nodes after add_new_attributes: ", G.nodes(data=True))
ut.add_new_attributes(G, 'edges', 'hello_edge', 1)
print("Edges after add_new_attributes: ", G.edges(data=True))

print("\nTest linkable_entries......")
leaves = ut.find_all_leaves(G)
print("All linkable entries are: ", ut.linkable_entries(G, (8,9), leaves))

