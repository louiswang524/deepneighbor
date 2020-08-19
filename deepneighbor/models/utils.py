import numpy as np
import dgl
import torch.nn as nn
from sklearn import preprocessing
import torch as th

def build_graph(data, train_test_ratio,embed_size):
    data['item'] = data['item'].astype(str)
    words = data['user'].unique().tolist() + data['item'].unique().tolist()
    le = preprocessing.LabelEncoder()
    le.fit(words)
    src = np.array(le.transform(data.user))
    dst = np.array(le.transform(data.item))
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    g = dgl.DGLGraph((u, v))
    # print(g.number_of_nodes())
    # print(len(g.nodes()))
    # print(len(data.user.unique().tolist()+data.item.unique().tolist()))
    dict_node = {raw:node for raw,node in zip(data.user.unique().tolist()+data.item.unique().tolist(),g.nodes())}
    node_embedings = nn.Embedding(g.number_of_nodes(), embed_size)
    g.ndata['feat'] = node_embedings.weight
    features = g.ndata['feat']
    labels = th.LongTensor(g.nodes())
    train_mask = th.BoolTensor([True]* int(g.number_of_nodes()*train_test_ratio) +
                               [False]*(len(g.nodes())-int(g.number_of_nodes()*train_test_ratio)))
    test_mask = th.BoolTensor([False]* int(g.number_of_nodes()*train_test_ratio) +
                               [True]*(len(g.nodes())-int(g.number_of_nodes()*train_test_ratio)))
    return g, features, labels, train_mask, test_mask
import networkx as nx
import itertools
import random
import pandas as pd


def generate_sentences(data):
    '''
    input dataframe.
      user  item
    	1	a
    	1	b
    	1	c
    	2	d
    	2	a
    output sentences. [user_id1, all_item(1...k1),user_id2,all_item(1...k2),...]
    [1,a,b,c,2,d,a]
    '''
    out = []
    data = pd.read_csv(data,sep=' ',header=None)

    data.columns = ['user','item']
    print(data.head())
    data['item'] = data['item'].astype(str)
    data['user'] = data['user'].astype(str)
    for user in data.user.unique():
        temp = [user]
        temp.extend(data[data.user == user].item.unique().tolist())
        out.append(temp)
    return out

def convert_to(data,le):
    '''
    given txt or csv file
    transform with label encoder
    return dataframe with two columns [from] and [to]
    '''

    data = pd.read_csv(data)
    data.columns = ['user','item']
    le.fit(data.user.unique().tolist()+data.item.unique().tolist())
    for col in data.columns:
        data[col] = le.transform(data[col])
    return data,le




def generate_sentences_dw(data):
    return nx.read_edgelist(data,)
