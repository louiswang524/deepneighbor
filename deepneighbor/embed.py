'''
input data:
a dataframe with two columns: user item

output:
a embedding lookup dictionary {'user_id/item_id':[vector]}

'''
from gensim.models import Word2Vec
#import deepneighbor.config as config
from deepneighbor.utils import generate_sentences,generate_sentences_dw,convert_to
from annoy import AnnoyIndex
from sklearn import preprocessing
from deepneighbor.models.gat import *



class Embed(object):
    def __init__(self,data_path,model='w2v',num_walks=100,walk_length=10):
        '''
        data: a dataframe: user, item

        model: 'w2v','deepwalk','gcn', 'gat'
        '''

        self.le = preprocessing.LabelEncoder()
        self.data ,self.le = convert_to(data_path,self.le)
        self.w2v_model = None
        self._annoy = None

        self._embeddings = {}
        self.model_type = model
        self.num_walks = num_walks
        self.walk_length = walk_length

        if self.model_type == 'w2v':

            self.sentences = generate_sentences(self.data)

        if self.model_type == 'deepwalk':
            self.sentences = generate_sentences_dw(data)
        if self.model_type == 'gat':
            pass




    def train(self,
            window_size=5,
            workers=3,
            iter=5,
            learning_rate=0.01,
            epochs = 10,
            dimensions = 128,
            num_of_walks=80,
            beta=0.5,
            gamma=0.5,
            **kwargs):
        self.workers=workers
        self.iter=iter
        self.window_size=window_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.dimensions=dimensions
        self.num_of_walks=num_of_walks
        self.beta=beta
        self.gamma=gamma
        self._annoy = AnnoyIndex(dimensions, 'angular')

        if self.model_type == 'w2v' or self.model_type=='deepwalk':
            kwargs["sentences"] = self.sentences
            kwargs["min_count"] = kwargs.get("min_count", 0)
            kwargs["size"] = self.dimensions
            kwargs["sg"] = 1  # skip gram
            kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
            kwargs["workers"] = self.workers
            kwargs["window"] = self.window_size
            kwargs["iter"] = self.iter


            print(f"There are {self.data.user.nunique()} users")
            print(f"There are {self.data.item.nunique()} items")

            print("Learning embedding vectors...")
            model_w2v = Word2Vec(**kwargs)
            print("Learning embedding vectors done!")

            self.w2v_model = model_w2v



            words = self.data['user'].unique().tolist() + self.data['item'].unique().tolist()

            for word in words:
                self._annoy.add_item(self.le.transform([word])[0],self.w2v_model.wv[word])

            self._annoy.build(-1)

        if self.model_type == 'gat':
            model = AttentionWalkTrainer(graph_path=self.data,
                                        dimensions=self.dimensions ,
                                        learning_rate=self.learning_rate,
                                        epochs=self.epochs ,
                                        window_size=self.window_size ,
                                        num_of_walks=self.num_of_walks,
                                        beta=self.beta,
                                        gamma=self.gamma )
            model.fit()
            emb = model.save_embedding()

            for id in emb.id:
                self._annoy.add_item(int(id),emb[emb.id==id].values.tolist()[0][1:])

            self._annoy.build(-1)




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

    def search(self, seed,k = 5, type=None):
        '''
        seed: seed item to find nearest neighbor
        k: number of cloest neighhbors
        '''

        a_return = self._annoy.get_nns_by_item(int(self.le.transform([seed])[0]), k)
        return list(self.le.inverse_transform(a_return))
