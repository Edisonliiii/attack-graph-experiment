import networkx as nx

# ---------- graph utilities
def read_graph():
  """
    Read graph from .gml

    [Parameters]
      Read from where
    [Return]
      should be G(nx graph), but needs to fix impl
  """
  path = "./data/train/"
  for filename in os.listdir(path):
    if filename.endswith(".gml"): # read out graph
      G_tmp = nx.read_gml(os.path.join(path, filename), label="label")
      pos_tmp = nx.multipartite_layout(G_tmp, subset_key="layer")
      nx.draw(G_tmp, pos_tmp,
        with_labels=True,
        node_size=5,
        connectionstyle="arc3,rad=-0.2",
        width=0.1,
        font_size=10)
      print(os.path.join(path, filename))
      # plt.show()

def store_graph(G: nx.Graph):
  """
    Store graph as file

    [Parameters]
      G -- graph
    [Return]
      None
  """
  name = "./data/train/" + "Graph-" +  uuid.uuid4().hex + ".gml"
  nx.write_gml(G, name)
  pass

def build_graph():
  """
    Define graph nodes and edges(relationships) in the context of BloodHound
  """
  computers = []
  users = []
  groups = []
  gpos = []
  outs = []

  relationship = {
    "AdminTo": [
      ("Group", "Computer")
    ],
    "GenericAll": [
      ("Group", "Computer")
    ],
    "AddMember": [
      ("Group", "Group")
    ],
    "AllowedToDelegate": [
      ("User", "Computer"),
      ("Computer", "Computer")
    ],
    "CanRDP": [
      ("Group", "Computer"),
      ("User", "Computer")
    ],
    "Contains": [
      ("Domain", "OU"),
      ("OU", "Computer")
    ],
    "DCSync": [
      ("Group", "Domain")
    ],
    "ExecuteDCOM": [
      ("User", "Computer"),
      ("Group", "Computer")
    ],
    "ForceChangePassword": [
      ("Group", "User")
    ],
    "GenericWrite": [],
    "GetChanges": [],
    "GetChangesAll": [],
    "GpLink": [],
    "HasSession": [],
    "MemberOf": [],
    "Owns": [],
    "ReadLAPSPassword": [],
    "WriteDacl": [],
    "WriteOwner": []
  }
  pass

def neo4j_builder():
  """
    Build neo4j according to generated graph
  """
  pass

# ---------- graph algorithms
def find_all_leaves(G: nx.Graph) -> list:
  """
    Get all leaf nodes

    [Parameters]
      G: nx.Graph -- graph
    [Return]
      leaves as list
  """
  return [v for v, d in G.in_degree() if d == 0]
