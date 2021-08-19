# essential
from torch_geometric.nn import TopKPooling
import torch_geometric.nn as tg_nn
from networkx.classes import graph
import sys
import time
from networkx.readwrite import graph6
from numpy import dtype
sys.path.insert(1, '../utility')
sys.path.insert(1, './')
sys.path.insert(1, '../gnn')
# networkx
import networkx as nx
import matplotlib.pyplot as plt
# customized
from graph_generator import GraphGenerator
import utility as ut
# torch
import torch
import torch.nn.functional as F
from torch import autograd
import torch.nn as nn

# [HOW TO CREATE NEW DATASET]
layer_sizes = [7, 65, 33, 43, 22, 22, 1]  # [3,12,31,32,23,42,12,5,2,1]
graph_gen = GraphGenerator(layer_sizes)
graph_gen.struct_graph(nonjump_percentage=0.6, outgoing_lower_bound=5,
                       outgoing_upper_bound=6, blockable_percentage=0.3)
graph_gen.edge_classification_sample()
data = graph_gen.networkx_to_torch()
# print(f'data.x:\n {data.x}')                          # node feature
# print(f'data.y:\n {data.y}')                          # edge classes
# print(f'data.edge_index:\n {data.edge_index}')        # edges
# print(f'data.edge_attr:\n {data.edge_attr}')          # edge feature
# print(f'data.new_edge_attr:\n {data.new_edge_attr}')
ground_truth = F.one_hot(data.y, num_classes=3)
# print(f'classification after one-hot:\n {ground_truth}')
# graph_gen.draw_graph()
# print(f'data.edge_attr:\n {data.batch}')
# graph_gen.store_graph()
# graph_gen.graph_debug()

# [HOW TO READ PROCESSED DATA]
new_graph = GraphGenerator([])
new_graph.read_graph("")
new_data = new_graph.networkx_to_torch()
# print(f'new_data.x:\n {new_data.x}')
# print(f'new_data.y:\n {new_data.y}')
# print(f'new_data.edge_index:\n {new_data.edge_index}')
# print(f'new_data.edge_attr:\n {new_data.edge_attr}')
# new_graph.graph_debug()

# print(data)
# print(new_data)
# print("Compare .x: ", data.x==new_data.x)
# print("Compare .y: ", data.y == new_data.y)
# print("Compare .edge_index: ", data.edge_index == new_data.edge_index)
# print("Compare .edge_attr: ", data.edge_attr == new_data.edge_attr)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, models

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = tg_nn.SAGEConv(9, 128)
    #self.pool1 = TopKPooling(128, ratio=0.8)
    self.conv2 = tg_nn.SAGEConv(128, 128)
    #self.pool2 = TopKPooling(128, ratio=0.8)
    self.conv3 = tg_nn.SAGEConv(128, 128)
    #self.pool3 = TopKPooling(128, ratio=0.8)
    # self.item_embedding = torch.nn.Embedding(
    #     num_embeddings=df.item_id.max()+1, embedding_dim=9)
    self.lin1 = torch.nn.Linear(128, 128)
    self.lin2 = torch.nn.Linear(128, 64)
    self.lin3 = torch.nn.Linear(64, 3)
  def forward(self, data):
    x, edge_index = data.new_edge_attr, data.edge_index
    # print(f'x: {x.shape}')
    # print(f'edge_index: {edge_index.shape}')
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv3(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.lin1(x)
    x = F.relu(x)
    x = self.lin2(x)
    x = F.relu(x)
    x = self.lin3(x)
    # print(f'----x: {x.shape}')
    return F.log_softmax(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


model.train()
L = nn.CrossEntropyLoss()
for epoch in range(10000):
   optimizer.zero_grad()
   out=model(data)
   print(f'out: {out.shape}')
   print(f'data.y: {ground_truth.shape}')
   loss = L(out, data.y)
   print(f'loss: {loss}')
   loss.backward()
   optimizer.step()

y_pred = model(new_data)
print(y_pred.shape)
_,pred = torch.max(y_pred.data, 1)
print(pred)
print(new_data.y)
right = (new_data.y==2).sum()
acc = torch.sum((pred==new_data.y)*(pred==2))
print("--: ", acc/right)
