from nodes import Nodes

class User(Nodes):
  def __init__(self, u_layer_number: int):
    super().__init__(n_layer_number=u_layer_number)
    ## concepts in neo4j
    # labels
    self.node_label.append(type(self).__name__)
    # properties
    self.owned: bool = False
    self.domain: str = ""
    self.displayname: str = ""
    self.pwdlastset: int = -1
    self.lastlogon: int = -1
    self.enabled: bool = True
