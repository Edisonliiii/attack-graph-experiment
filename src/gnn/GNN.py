import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling
from torch_geometric.nn.pool import topk_pool
import SAGEConv
embed_dim = 9

class GNN(torch.nn.Module):
  def __init__(self):
    super(GNN, self).__init__()

    self.conv1 = SAGEConv(embed_dim, 128)
    self.pool1 = TopKPooling(128, ratio=0.8)
    self.conv2 = SAGEConv(128, 128)
    self.pool2 = TopKPooling(128, ratio=0.8)
    self.conv3 = SAGEConv(128, 128)
    self.pool3 = TopKPooling(128, ratio=0.8)
    self.item_embedding = torch.nn.Embedding(
        num_embeddings=df.item_id.max()+1, embedding_dim=embed_dim)
    self.lin1 = torch.nn.Linear(256, 128)
    self.lin2 = torch.nn.Linear(128, 64)
    self.lin3 = torch.nn.Linear(64, 1)
    self.bn1 = torch.nn.BatchNorm1d(128)
    self.bn2 = torch.nn.BatchNorm1d(64)
    self.act1 = torch.nn.ReLU()
    self.act2 = torch.nn.ReLU()

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.item_embedding(x)
    x = x.squeeze(1)

    x = F.relu(self.conv1(x, edge_index))

    x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
    x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

















