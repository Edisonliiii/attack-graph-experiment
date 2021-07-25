from __future__ import annotations
import abc
import uuid

class Nodes(metaclass=abc.ABCMeta):
  # static attribute
  node_idx : int = 10
  # define shared attributes
  def __init__(self, n_layer_number=-1):
    # concepts in networkx
    self.self_idx : int = 0
    self.self_layer_number : int = n_layer_number
    self.out_link : list  = []
    self.in_link : list = []
    self.node_props : list = []
    self.extra_props : list = []
    # concepts in neo4j
    self.node_label: list= []
    self.node_properties: dict = {}
    self.node_objectId: str = str(uuid.uuid4())
    self.node_name : str = ""
  def get_node_idx(self):
    return type(self).node_idx
  def set_node_idx(self, val: int):
    type(self).node_idx = val

# c = Computer(c_layer_number=100)
# u = User(u_layer_number=50)
# print("----oop test----")
# print("Computer C'tor test: ", c.self_layer_number)
# print("User C'tor test: ", u.self_layer_number)
# ch = hash(c)
# uh = hash(u)
