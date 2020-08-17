'''
input data:
a dataframe with two columns: user item

output:
a embedding lookup dictionary {'user_id/item_id':[vector]}

'''
from gensim.models import Word2Vec
#import deepneighbor.config as config
from deepneighbor.utils import generate_sentences
from annoy import AnnoyIndex
from sklearn import preprocessing


EMBED_SIZE = 128
WINDOW_SIZE = 10
ITER = 5
WORKERS = 3


class Embed(object):
    def __init__(self,data,model='w2v'):
        '''
        data: a dataframe: user, item

        model: 'w2v','deepwalk','gcn'
        '''

        self.data = data
        self.w2v_model = None
        self._annoy = None
        self._le = None
        self._embeddings = {}
        self.model_type = model

        if self.model_type == 'w2v':

            self.sentences = generate_sentences(data)
        if self.model_type == 'deepwalk':
            pass
        if self.model_type == 'gcn':
            pass

        self.dimension = 0




    def train(self,
            embed_size=128,
            window_size=5,
            workers=3,
            iter=5,
            task = 'NodeClassification',
            train_test_ratio = 0.8,
            **kwargs):

        if self.model_type == 'w2v':
            kwargs["sentences"] = self.sentences
            kwargs["min_count"] = kwargs.get("min_count", 0)
            kwargs["size"] = embed_size
            kwargs["sg"] = 1  # skip gram
            kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
            kwargs["workers"] = workers
            kwargs["window"] = window_size
            kwargs["iter"] = iter
            self.dimension = embed_size

            print(f"There are {self.data.user.nunique()} users")
            print(f"There are {self.data.item.nunique()} items")

            print("Learning embedding vectors...")
            model_w2v = Word2Vec(**kwargs)
            print("Learning embedding vectors done!")

            self.w2v_model = model_w2v

            self._annoy = AnnoyIndex(self.dimension, 'angular')

            words = self.data['user'].unique().tolist() + self.data['item'].unique().tolist()
            self._le = preprocessing.LabelEncoder()
            self._le.fit(words)
            for word in words:
                self._annoy.add_item(self._le.transform([word])[0],self.w2v_model.wv[word])

            self._annoy.build(-1)

        if self.model_type == 'gcn':
            pass


        #return model_w2v


    # def get_embeddings(self,):
    #     if self.w2v_model is None:
    #         print("model not train")
    #         return {}
    #
    #     self._embeddings = {}
    #     words = self.data['user'].unique().tolist() + self.data['item'].unique().tolist()
    #     for word in words:
    #         self._embeddings[word] = self.w2v_model.wv[word]
    #
    #     return self._embeddings

    def search(self, seed,k = 5):
        '''
        seed: seed item to find nearest neighbor
        k: number of cloest neighhbors
        '''

        a_return = self._annoy.get_nns_by_item(self._le.transform([seed])[0], k)
        return list(self._le.inverse_transform(a_return))
