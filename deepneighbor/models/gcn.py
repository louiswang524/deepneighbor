import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *
from dgl.nn.pytorch import GraphConv
import time


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

#
# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#
#     def forward(self, g, feature):
#         # Creating a local scope so that all the stored ndata and edata
#         # (such as the `'h'` ndata below) are automatically popped out
#         # when the scope exits.
#         with g.local_scope():
#             g.ndata['h'] = feature
#             g.update_all(gcn_msg, gcn_reduce)
#             h = g.ndata['h']
#             return self.linear(h)

class Net(nn.Module):
    def __init__(self,in_feats, out_feats,num_classes):
        super(Net, self).__init__()
        self.layer1 = GraphConv(in_feats, out_feats)
        self.layer2 = GraphConv(out_feats, num_classes)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = torch.relu(h)
        h = self.layer2(g, h)
        return h




class GCN(object):
    def __init__(self, data, embed_size = 128, task = 'NodeClassification',train_test_ratio = 0.8):
        self.g,self.features,self.labels,self.train_mask,self.test_mask = build_graph(data, train_test_ratio,embed_size=128)
        self.embed_size = embed_size

    def evaluate(self,model, g, features, labels, mask):
        model.eval()
        with th.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def train(self,verbose = 1,epochs = 10):
        net = Net(self.g.number_of_nodes(), self.embed_size,num_classes=len(th.unique(self.labels)))
        print(net)
        optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
        dur = []
        for epoch in range(epochs):
            if epoch >=3:
                t0 = time.time()

            net.train()
            logits = net(self.g, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >=3:
                dur.append(time.time() - t0)

            acc = evaluate(net, self.g, self.features, self.labels, self.test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, loss.item(), acc, np.mean(dur)))
