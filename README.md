# DeepNeighbor

[![Python Versions](https://img.shields.io/pypi/pyversions/deepneighbor.svg)](https://pypi.org/project/deepneighbor)
[![PyPI Version](https://img.shields.io/pypi/v/deepneighbor.svg)](https://pypi.org/project/deepneighbor)

---

DeepNeighbor is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based models along with lots of core components layers which can be used to easily build custom models.You can use any complex model with
<br>`model.train()`，which generates embeddings for users and items (Deep),
<br> and `model.search()`, which looks for Approximate nearest neighbor for seed user/item (Neighbor) .

```python
from deepneighbor.embed import Embed

model = Embed(data)
model.train()
model.search(seed = 'Louis', k=10)
```
