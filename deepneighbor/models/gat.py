import torch
import numpy as np
import pandas as pd
from tqdm import trange
from deepneighbor.models.utils import read_graph, feature_calculator, adjacency_opposite_calculator
from sklearn import preprocessing
class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see the paper.
    """
    def __init__(self,graph_path,window_size,learning_rate,epochs,dimensions,num_of_walks,beta,gamma,shapes):
        """
        Setting up the layer.
        :param args: Arguments object.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        self.graph_path = graph_path
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_of_walks = num_of_walks
        self.dimensions = dimensions
        self.shapes = shapes
        self.beta=beta
        self.gamma = gamma
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        half_dim = int(self.dimensions/2)
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], half_dim))
        self.right_factors = torch.nn.Parameter(torch.Tensor(half_dim, self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0], 1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.right_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim=0)
        probs = self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_tensor = weighted_target_tensor * probs
        weighted_tar_mat = torch.sum(weighted_target_tensor, dim=0)
        weighted_tar_mat = weighted_tar_mat.view(self.shapes[1], self.shapes[2])
        estimate = torch.mm(self.left_factors, self.right_factors)
        loss_on_target = - weighted_tar_mat* torch.log(torch.sigmoid(estimate))
        loss_opposite = -adjacency_opposite * torch.log(1-torch.sigmoid(estimate))
        loss_on_mat = self.num_of_walks*weighted_tar_mat.shape[0]*loss_on_target+loss_opposite
        abs_loss_on_mat = torch.abs(loss_on_mat)
        average_loss_on_mat = torch.mean(abs_loss_on_mat)
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.beta * (self.attention.norm(2)**2)
        loss = average_loss_on_mat + loss_on_regularization + self.gamma*norms
        return loss

class AttentionWalkTrainer(object):
    """
    Class for training the AttentionWalk model.
    """
    def __init__(self,
                graph_path,
                window_size,
                learning_rate,
                epochs ,
                dimensions ,
                num_of_walks,

                beta,
                gamma):
        """
        Initializing the training object.
        :param args: Arguments object.
        """
        self.graph_path = graph_path
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dimensions = dimensions
        self.beta = beta
        self.gamma = gamma
        self.num_of_walks=num_of_walks
        #self.le = preprocessing.LabelEncoder()

        self.graph = read_graph(self.graph_path)
        self.initialize_model_and_features()

    def initialize_model_and_features(self):
        """
        Creating data tensors and factroization model.
        """
        print("Initialing Graph data.")
        self.target_tensor = feature_calculator( self.graph,self.window_size)

        self.target_tensor = torch.FloatTensor(self.target_tensor)

        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        print(f"{self.target_tensor.shape} and {self.adjacency_opposite.shape}")
        self.model = AttentionWalkLayer(graph_path = self.graph_path,
                                        window_size = self.window_size,
                                        learning_rate = self.learning_rate,
                                        epochs = self.epochs,
                                        num_of_walks=self.num_of_walks,
                                        dimensions = self.dimensions,
                                        beta = self.beta,
                                        gamma = self.gamma,
                                        shapes = self.target_tensor.shape)




    def fit(self):
        """
        Fitting the model
        """
        print("\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epochs = trange(self.epochs, desc="Loss")
        for _ in self.epochs:
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()
            self.epochs.set_description("Attention Walk (Loss=%g)" % round(loss.item(), 4))
        print("\nModel Training Finished.\n")

    def save_model(self):
        """
        Saving the embedding and attention vector.
        """
        self.save_embedding()
        self.save_attention()

    def save_embedding(self):
        """
        Saving the embedding matrices as one unified embedding.
        """
        print("\nSaving the model.\n")
        left = self.model.left_factors.detach().numpy()
        right = self.model.right_factors.detach().numpy().T
        indices = np.array([range(len(self.graph))]).reshape(-1, 1)
        embedding = np.concatenate([indices, left, right], axis=1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.dimensions)]
        embedding = pd.DataFrame(embedding, columns=columns)
        #embedding.to_csv(self.args.embedding_path, index=None)
        print("\nEmbedding Saved!.\n")
        return embedding

    def save_attention(self):
        """
        Saving the attention vector.
        """
        attention = self.model.attention_probs.detach().numpy()
        indices = np.array([range(self.window_size)]).reshape(-1, 1)
        attention = np.concatenate([indices, attention], axis=1)
        attention = pd.DataFrame(attention, columns=["Order", "Weight"])
        #attention.to_csv(self.args.attention_path, index=None)
