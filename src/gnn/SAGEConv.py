import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class SAGEConv(MessagePassing):
  def __init__(self, in_channels, out_channels):
    # "max" aggregation
    super(SAGEConv, self).__init__(aggr='max')
    # torch.nn.Liear is doing:
    # 1. each neighboring node embedding is multiplied by a weight matrix, added a bias
    # 2. pass an activation function
    self.lin = torch.nn.Linear(in_channels, out_channels)
    self.act = torch.nn.ReLU()
    # this part is for update
    # the aggregated message and the current node embedding is aggregated
    # then the same process as the previous round
    self.update_lin = torch.nn.Linear(
        in_channels + out_channels, in_channels, bias=False)
    self.update_act = torch.nn.ReLU()

  def forward(self, x, edge_index):
    # x: node [N, in_channels(features)]
    # edge_index: edge [2, E]
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

  def message(self, x_j):
    # x_j has shape [E, in_channels]
    x_j = self.lin(x_j)
    x_j = self.act(x_j)
    return x_j

  def update(self, aggr_out, x):
    # aggr_out has shape [N, out_channels]
    new_embedding = torch.cat([aggr_out, x], dim=1)
    new_embedding = self.update_lin(new_embedding)
    new_embedding = self.update_act(new_embedding)
    return new_embedding
