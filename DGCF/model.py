import torch

from torch.nn import Module, ModuleList
from torch.nn import Embedding, Linear
from torch.nn import functional


class GCCF(Module):

    def __init__(self, num_user, num_movies, hparams):

        super(GCCF, self).__init__()

        self._hparams = hparams

        emb_size = self._hparams["emb_size"]
        num_layers = self._hparams["num_layers"]

        self._user_embedding = Embedding(num_user, emb_size)
        self._movie_embedding = Embedding(num_movies, emb_size)

        self._user_layers = ModuleList()
        self._movie_layers = ModuleList()

        for _ in range(num_layers):
            self._user_layers.append(Linear(emb_size, emb_size))
            self._movie_layers.append(Linear(emb_size, emb_size))

        self._output_layer = Linear((num_layers + 1) * emb_size, 1)

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self._user_embedding.weight)
        torch.nn.init.xavier_uniform_(self._movie_embedding.weight)

        for user_layer, movie_layer in zip(self._user_layers, self._movie_layers):
            torch.nn.init.xavier_uniform_(user_layer.weight)
            torch.nn.init.xavier_uniform_(movie_layer.weight)
            torch.nn.init.zeros_(user_layer.bias)
            torch.nn.init.zeros_(movie_layer.bias)

        torch.nn.init.xavier_uniform_(self._output_layer.weight)
        torch.nn.init.zeros_(self._output_layer.bias)

    def forward(self, user_adj, movie_adj, user_id, movie_id):

        dropout = self._hparams["dropout"]

        user_embeddings = []
        movie_embeddings = []

        user_embedding = self._user_embedding.weight
        movie_embedding = self._movie_embedding.weight

        user_embeddings.append(user_embedding)
        movie_embeddings.append(movie_embedding)

        for user_layer, movie_layer in zip(self._user_layers, self._movie_layers):

            user_embedding = user_layer(user_adj @ movie_embeddings[-1]) + user_layer(user_embeddings[-1])
            user_embedding = functional.leaky_relu(user_embedding)

            movie_embedding = movie_layer(movie_adj @ user_embeddings[-1]) + movie_layer(movie_embeddings[-1])
            movie_embedding = functional.leaky_relu(movie_embedding)

            user_embeddings.append(user_embedding)
            movie_embeddings.append(movie_embedding)

        user_item_interactions = []
        for user_embedding, movie_embedding in zip(user_embeddings, movie_embeddings):
            user_item_interactions.append(user_embedding[user_id] * movie_embedding[movie_id])
        user_item_interactions = torch.cat(user_item_interactions, dim=1)
        user_item_interactions = functional.dropout(user_item_interactions, p=dropout, training=self.training)
        output = self._output_layer(user_item_interactions)

        return output.view(-1)
