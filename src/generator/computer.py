from nodes import Nodes

class Computer(Nodes):
  def __init__(self, c_layer_number: int):
    super().__init__(n_layer_number=c_layer_number)
    ## concepts in neo4j
    # labels
    self.node_label.append(type(self).__name__)
    # properties
    self.owned: bool = False
    self.domain: str = ""
    self.operatingsystem: str = ""
    # name as node_name
    # objectid as node_objectId generated as created
    self.enabled: bool = True
