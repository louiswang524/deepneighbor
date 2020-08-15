'''
input data:
a dataframe with two columns: user item

output:
a embedding lookup dictionary {'user_id/item_id':[vector]}

'''
from gensim.models import Word2Vec
import deepneighbor.config
from deepneighbor.utils import generate_sentences
from annoy import AnnoyIndex
from sklearn import preprocessing

class Embed(object):
    def __init__(self,data):
        '''
        data: a dataframe: user, item
        '''

        self.data = data
        self.w2v_model = None
        self._embeddings = {}


        self.sentences = generate_sentences(data)

    def sentences(self):
        return self.sentences


    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = config.EMBED_SIZE
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = config.WORKERS
        kwargs["window"] = config.WINDOW_SIZE
        kwargs["iter"] = config.ITER

        print(f"There are {self.data.user.nunique()} users")
        print(f"There are {self.data.item.nunique()} items")

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model


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

        a = AnnoyIndex(config.EMBED_SIZE, 'angular')

        words = self.data['user'].unique().tolist() + self.data['item'].unique().tolist()
        le = preprocessing.LabelEncoder()
        le.fit(words)
        for word in words:
            a.add_item(le.transform([word])[0],self.w2v_model.wv[word])

        a.build(-1)

        a_return = a.get_nns_by_item(le.transform([seed])[0], k)
        return le.inverse_transform(a_return)
