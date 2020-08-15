import networkx as nx
from models import DeepWalk
class Embed:
    def __init__(self, data, walk_length=10, num_walks=80, workers=1):
        self.graph = nx.read_edgelist(data,create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])# Read graph
        self.walk_length = walk_length
        sekf.num_walks = num_walks
        self.workers = workers

    def get_embeddings(self):
        model = DeepWalk(self.graph, walk_length = self.walk_length, num_walks=self.num_walks,workers=self.workers)
        model.train(window_size=5,iter=3)
        return model.getembeddings()
