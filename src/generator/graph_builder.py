# essential
import os
import sys
import numpy as np
from random import randint
sys.path.insert(1, '../utility')
sys.path.insert(1, './')
sys.path.insert(1, '../gnn')
# networkx
# customized
from graph_generator import GraphGenerator
import utility as ut
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg_nn
from torch_geometric.data import DataLoader

#[HOW TO CREATE NEW DATASET]
def create_dataset(dataset_type: str, num_graph: int, num_layer: int, lower_bound: int, upper_bound: int) -> None:
  """
  create training dataset
  :parameter
    dataset_type -- 'train' / 'test' / 'validation'
    num_graph -- how many graphs you want to generate
    num_layer -- how many layers you prefer (should be also randmoized, but not for now)
    lower_bound -- lower bound of the number of nodes for each layer
    upper_bound -- .. value for each layer
  :return
    None
  """
  for i in range(num_graph):
    layer_sizes = np.random.randint(lower_bound, upper_bound, size=num_layer).tolist()
    layer_sizes.append(1) # always end with 1 which means AD
    graph_gen = GraphGenerator(layer_sizes)
    graph_gen.struct_graph(nonjump_percentage=0.6, outgoing_lower_bound=2,
                          outgoing_upper_bound=3, blockable_percentage=0.3)
    graph_gen.edge_classification_sample()
    # graph_gen.store_graph(dataset_type)
    graph_gen.draw_graph()
    # data = graph_gen.networkx_to_torch()
    # ground_truth = F.one_hot(data.y, num_classes=3)

# [HOW TO READ PROCESSED DATA]
def read_data_from_dataset(dataset_type: str) -> list:
  """
  read data from training set
  :parameter
    dataset_type -- 'train' / 'test' / 'validation'
  :return
    big list
  """
  path = f'./data/{dataset_type}/'
  dataset: list = []
  new_graph = GraphGenerator([])
  # read all training data one by one
  for filename in os.listdir(path):
    if filename.endswith(".gml"):  # read out graph
      new_graph.read_graph(os.path.join(path, filename))
      # new_graph.graph_debug()
      # new_graph.draw_graph()
      new_data = new_graph.networkx_to_torch()
      # new_graph.torch_debug(new_data)
      dataset.append(new_data)
  return dataset

# algorithm_5------------------------------------------------------------------------------------------
layer_sizes = np.random.randint(20000, 30000, size=5).tolist()
layer_sizes.append(1)  # always end with 1 which means AD
graph_gen = GraphGenerator(layer_sizes)
graph_gen.struct_graph(nonjump_percentage=0.9, outgoing_lower_bound=6,
                       outgoing_upper_bound=9, blockable_percentage=0.4)
# graph_gen.draw_graph()
graph_gen.algorithm_5()

# TORCH------------------------------------------------------------------------------------------------
# class Net(torch.nn.Module):
#   def __init__(self, p_drop=0.0):
#     super(Net, self).__init__()
#     self.conv1 = tg_nn.SAGEConv(9, 64, aggr='max')
#     #self.pool1 = TopKPooling(64, ratio=0.8)
#     self.conv2 = tg_nn.SAGEConv(64, 128, aggr='max')
#     #self.pool2 = TopKPooling(32, ratio=0.8)
#     self.conv3 = tg_nn.SAGEConv(128, 128, aggr='max')
#     #self.pool3 = TopKPooling(32, ratio=0.8)
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
#     x = self.act1(self.conv1(x, edge_index=edge_index))
#     x = self.act2(self.conv2(x, edge_index=edge_index))
#     x = self.conv3(x, edge_index=edge_index)
#     x = self.lin1(x)
#     x = F.relu(x)
#     x = self.lin2(x)
#     x = F.relu(x)
#     x = self.lin3(x)
#     # print(f'----x: {x.shape}')
#     return F.log_softmax(x)

# def train(dataset: list):
#   """
#   single train process
#   :parameter
#     dataset -- data list, one for one graph
#   :return
#   """
#   # _dataset = DataLoader(dataset, batch_size=1, shuffle=True)
#   # for batch in _dataset:
#   #   print(batch)
#   # exit(1)
#   # model.train()
#   dataset_length = len(dataset)
#   for i in range(40):
#     # index_for_train = randint(0, dataset_length-1)
#     data = dataset[i].to(device)
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out, data.y)
#     print(f'loss: {loss}')
#     loss.backward()
#     optimizer.step()

# Train
# create_dataset('train', 100, 5, 3, 10)
# create_dataset('test', 10, 5, 3, 10)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=0.001, weight_decay=5e-4)
# dataset_for_train = read_data_from_dataset('train')
# dataset_for_test = read_data_from_dataset('test')
# model.train()
# for epoch in range(10000):
#   print(f'-----------------------Round {epoch}')
#   train(dataset_for_train)
#   y_pred = model(dataset_for_test[11])
#   _,pred = torch.max(y_pred.data, 1)
#   right = (dataset_for_test[11].y==2).sum()
#   acc = torch.sum((pred==dataset_for_test[11].y)*(pred==2))
#   print("--: ", acc/right)

# # --------------------------------------- old train ------------------
# for epoch in range(5000):
#    optimizer.zero_grad()
#    out=model(dataset[0])
#   #  print(f'out: {out.shape}')
#   #  print(f'data.y: {ground_truth.shape}')
#    loss = F.nll_loss(out, dataset[0].y)
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










