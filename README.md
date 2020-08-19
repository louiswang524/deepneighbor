# DeepNeighbor
<p align="center">
  <img src="deepneighbor_logo.png"/>
</p>

[![Python Versions](https://img.shields.io/pypi/pyversions/deepneighbor.svg)](https://pypi.org/project/deepneighbor)
[![PyPI Version](https://img.shields.io/pypi/v/deepneighbor.svg)](https://pypi.org/project/deepneighbor)
[![license](https://img.shields.io/github/license/LouisBIGDATA/deepneighbor.svg?maxAge=2592000)](https://github.com/LouisBIGDATA/deepneighbor)
![GitHub repo size](https://img.shields.io/github/repo-size/Lou1sWang/deepneighbor)
[![Downloads](https://pepy.tech/badge/deepneighbor)](https://pepy.tech/project/deepneighbor)
[![GitHub Issues](https://img.shields.io/github/issues/Lou1sWang/deepneighbor.svg
)](https://github.com/Lou1sWang/deepneighbor/issues)
---

DeepNeighbor is a **High-level**,**Flexible** and **Extendible** package for embedding-based information retrieval from user-item interaction logs. Just as the name suggested, **'deep'** means deep learning models to get user/item embeddings, while **'neighbor'** means approximate nearest neighbor search in the embedding space.<br>
It mainly has two parts : Embed step and Search step by the following codes:<br>
<br>`model.train()`，which generates embeddings for users and items (Deep),
<br> `model.search()`, which looks for Approximate nearest neighbor for seed user/item (Neighbor) .
<br>

### Install
```python
pip install deepneighbor
```
### How To Use

```python
from deepneighbor.embed import Embed

model = Embed(data)
model.train(model='gat')
model.search(seed = 'Louis', k=10)
```
### Input format
The input data for the **Embed()** should be a (*.csv or *.txt ) file with two columns in order: 'user' and 'item'. For each user, the item are recommended to be ordered by time.
### Models
- [x]  word2vec
- [ ] Siamese Network with triple loss
- [ ]  deepwalk
- [x]  graph convolutional network
- [ ]  matrix factorization
- [x]  graph attention network

### Model Parameters
#### deepwalk
```python
model = Embed(data, model = 'deepwalk')
model.train(window_size=5,
            workers=1,
            iter=1
            dimensions=128)
```
- ```window_size``` Skip-gram window size.
- ```workers```Use these many worker threads to train the model (=faster training with multicore machines).
- ```iter``` Number of iterations (epochs) over the corpus.
- ```dimensions``` Dimensions for the node embeddings


#### graph attention network 
```python
model = Embed(data, model = 'gat')
model.train(window_size=5,
            learning_rate=0.01,
            epochs = 10,
            dimensions = 128,
            num_of_walks=80,
            beta=0.5,
            gamma=0.5,)
```
- ```window_size``` Skip-gram window size.
- ```learning_rate``` learning rate for optimizing graph attention network
- ```epochs``` Number of gradient descent iterations.
- ```dimensions``` Dimensions for the embeddings for each node (user/item)
- ```num_of_walks```Number of random walks.
- ```beta``` and ```gamma```Regularization parameter.


### Examples
TBD

### License
