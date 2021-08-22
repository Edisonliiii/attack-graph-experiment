import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg_nn


class Net(torch.nn.Module):
  def __init__(self, p_drop=0.75):
    super(Net, self).__init__()
    self.conv1 = tg_nn.SAGEConv(9, 64, normalize=True)
    #self.pool1 = TopKPooling(64, ratio=0.8)
    self.conv2 = tg_nn.SAGEConv(64, 64)
    #self.pool2 = TopKPooling(64, ratio=0.8)
    self.conv3 = tg_nn.SAGEConv(64, 64)
    #self.pool3 = TopKPooling(64, ratio=0.8)
    # self.item_embedding = torch.nn.Embedding(
    #     num_embeddings=df.item_id.max()+1, embedding_dim=9)
    self.act1 = nn.Sequential(nn.ReLU(),
                              nn.Dropout(p_drop))
    self.act2 = nn.Sequential(nn.ReLU(),
                              nn.Dropout(p_drop))
    self.lin1 = torch.nn.Linear(64, 64)
    self.lin2 = torch.nn.Linear(64, 64)
    self.lin3 = torch.nn.Linear(64, 3)
  # def apply_edges(self, edges):
  #   h_u = edges[0]
  #   h_v = edges[1]
  #   score = self.W(torch.cat([h_u, h_v], 1))
  #   return {'score': sc}
  def forward(self, data):
    x, edge_index = data.new_x, data.new_edge_index
    # print(f'x: {x.shape}')
    # print(f'edge_index: {edge_index.shape}')
    x = self.act1(self.conv1(x, edge_index))
    x = self.act2(self.conv2(x, edge_index))
    x = self.conv3(x, edge_index)
    x = self.lin1(x)
    x = F.relu(x)
    x = self.lin2(x)
    x = F.relu(x)
    x = self.lin3(x)
    # print(f'----x: {x.shape}')
    return F.log_softmax(x)
