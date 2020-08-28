# DeepNeighbor
<br />
<p align="center">
  <a >
    <img src="deepneighbor_logo.png" alt="Logo" width="150" height="150">
  </a>
  <p align="center">
    Embedding-based Retrieval for ANN Search and Recommendations!
    <br />
    <a href="https://colab.research.google.com/drive/1j6uWt_YYyHBQDK7EN3f5GTTZTmNn2Xc5?usp=sharing">View Demo</a>
    ·
    <a href="https://github.com/Lou1sWang/deepneighbor/issues">Report Bug</a>
    ·
    <a href="https://github.com/Lou1sWang/deepneighbor/issues">Request Feature</a>
  </p>
</p>

[![Python Versions](https://img.shields.io/pypi/pyversions/deepneighbor.svg)](https://pypi.org/project/deepneighbor)
[![PyPI Version](https://img.shields.io/pypi/v/deepneighbor.svg)](https://pypi.org/project/deepneighbor)
[![license](https://img.shields.io/github/license/LouisBIGDATA/deepneighbor.svg?maxAge=2592000)](https://github.com/LouisBIGDATA/deepneighbor)
![GitHub repo size](https://img.shields.io/github/repo-size/Lou1sWang/deepneighbor)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Lou1sWang/deepneighbor/)


[![Downloads](https://pepy.tech/badge/deepneighbor)](https://pepy.tech/project/deepneighbor)
[![GitHub Issues](https://img.shields.io/github/issues/Lou1sWang/deepneighbor.svg)](https://github.com/Lou1sWang/deepneighbor/issues)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Lou1sWang/deepneighbor/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](louiswang524@gmail.com)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


### Install
```python
pip install deepneighbor
```
### How To Use

```python
from deepneighbor import Embed

model = Embed(data_path, model='gat')
model.train() # see optional parameters below
model.search(seed = 'Louis', k=10) # ANN search
embedings = model.get_embeddings() # dictionary. key: node; value: n-dim node embedding
```
### Input format
The input data for the **Embed()** should be a (*.csv or *.txt ) file path (e.g. '\data\data.csv')with two columns in order: 'user' and 'item'. For each user, the item are recommended to be ordered by time.
### Models & parameters in Embed()
- [x] Word2Vec `w2v`
- [x] Graph attention network                    `gat`
- [ ] Factorization Machines `fm`
- [ ] Deep Semantic Similarity Model
- [ ] Siamese Network with triple loss
- [ ] Deepwalk
- [ ] Graph convolutional network
- [ ] Neural Graph Collaborative Filtering algorithm `ngcf`
- [ ] Matrix factorization `mf`


### Model Parameters
#### word2vec
```python
model = Embed(data, model = 'w2v')
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

### How To Search
#### ```model.search(seed, k)```
- ```seed``` The Driver for the algorithms
- ```k``` Number of Nearest Neighbors.

### Examples
Open [Colab](https://colab.research.google.com/drive/1j6uWt_YYyHBQDK7EN3f5GTTZTmNn2Xc5?usp=sharing) to run the example with facebook data.
### Contact
Please contact louiswang524@gmail.com for collaboration or providing feedbacks.
### License
This project is under MIT License, please see [here](LICENSE) for details.
