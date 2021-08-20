# essential
from networkx.algorithms.link_analysis.hits_alg import hub_matrix
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

#[HOW TO CREATE NEW DATASET]
layer_sizes = [43,12,31,32,23,42,12,5,2,1] #[7, 65, 333, 143, 212, 122, 1]  # [3,12,31,32,23,42,12,5,2,1]
graph_gen = GraphGenerator(layer_sizes)
graph_gen.struct_graph(nonjump_percentage=0.6, outgoing_lower_bound=5,
                       outgoing_upper_bound=6, blockable_percentage=0.3)
graph_gen.edge_classification_sample()
data = graph_gen.networkx_to_torch()
ground_truth = F.one_hot(data.y, num_classes=3)
# graph_gen.draw_graph()
# graph_gen.store_graph()
# graph_gen.graph_debug()

# [HOW TO READ PROCESSED DATA]
new_graph = GraphGenerator([])
new_graph.read_graph("")
new_data = new_graph.networkx_to_torch()

# TORCH------------------------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, models
# class Net(torch.nn.Module):
#   def __init__(self, p_drop=0):
#     super(Net, self).__init__()
#     self.conv1 = tg_nn.SAGEConv(9, 128, normalize=True)
#     #self.pool1 = TopKPooling(128, ratio=0.8)
#     self.conv2 = tg_nn.SAGEConv(128, 128)
#     #self.pool2 = TopKPooling(128, ratio=0.8)
#     self.conv3 = tg_nn.SAGEConv(128, 128)
#     #self.pool3 = TopKPooling(128, ratio=0.8)
#     # self.item_embedding = torch.nn.Embedding(
#     #     num_embeddings=df.item_id.max()+1, embedding_dim=9)
#     self.act1 = nn.Sequential(nn.ReLU(),
#                               nn.Dropout(p_drop))
#     self.act2 = nn.Sequential(nn.ReLU(),
#                               nn.Dropout(p_drop))
#     self.lin1 = torch.nn.Linear(128, 128)
#     self.lin2 = torch.nn.Linear(128, 64)
#     self.lin3 = torch.nn.Linear(64, 3)
#   # def apply_edges(self, edges):
#   #   h_u = edges[0]
#   #   h_v = edges[1]
#   #   score = self.W(torch.cat([h_u, h_v], 1))
#   #   return {'score': sc}
#   def forward(self, data):
#     x, edge_index = data.new_x, data.new_edge_index
#     # print(f'x: {x.shape}')
#     # print(f'edge_index: {edge_index.shape}')
#     x = self.act1(self.conv1(x, edge_index))
#     x = self.act2(self.conv2(x, edge_index))
#     x = self.conv3(x, edge_index)
#     x = self.lin1(x)
#     x = F.relu(x)
#     x = self.lin2(x)
#     x = F.relu(x)
#     x = self.lin3(x)
#     # print(f'----x: {x.shape}')
#     return F.log_softmax(x)

# # Train
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=0.0001, weight_decay=5e-4)

# def train():
#   """
#   build train process
#   """
#   model.train()
#   for data in train_loader:
#     data = data.to(device)
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out, data.y)
#     loss.backward()
#     optimizer.step()

    
# # --------------------------------------- old train ------------------
# for epoch in range(5000):
#    optimizer.zero_grad()
#    out=model(data)
#   #  print(f'out: {out.shape}')
#   #  print(f'data.y: {ground_truth.shape}')
#    loss = F.nll_loss(out, data.y)
#    print(f'loss: {loss}')
#    loss.backward()
#    optimizer.step()
# y_pred = model(new_data)
# # print(torch.exp(y_pred))
# # y_pred = torch.exp(y_pred)
# _,pred = torch.max(y_pred.data, 1)
# # print(pred)
# # print(new_data.y)
# right = (new_data.y==2).sum()
# acc = torch.sum((pred==new_data.y)*(pred==2))
# print("--: ", acc/right)
# DGL------------------------------------------------------------------------------------------------










