import torch

from torch.nn import Module, ModuleList
from torch.nn import Embedding, Linear
from torch.nn import functional
from torch import nn


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

        # Registered buffers that store the last propagated node states
        # from the previous snapshot. These are moved with the model and
        # are not trainable parameters. They are used as a carry-over
        # prior for the next snapshot's computation.
        self.register_buffer('prev_user_state', torch.zeros(num_user, emb_size))
        self.register_buffer('prev_movie_state', torch.zeros(num_movies, emb_size))

        
        self._use_agate = bool(self._hparams.get("use_agate", False))
        if self._use_agate:
            # Adaptive fusion combines previous state and current embedding per-node
            # using a learned gate. We'll create two instances for users
            # and movies.
            class AGate(nn.Module):
                def __init__(self, emb_size):
                    super(AGate, self).__init__()
                    # produces a per-dimension gate in (0,1)
                    self._proj = Linear(emb_size * 2, emb_size)
                    # initialize bias to favor previous state slightly for stability
                    try:
                        nn.init.constant_(self._proj.bias, -1.0)
                    except Exception:
                        # fallback if bias not present
                        pass

                def forward(self, prev, curr):
                    # prev, curr: (N, emb_size)
                    x = torch.cat([prev, curr], dim=1)
                    z = torch.sigmoid(self._proj(x))
                    # store last gate (detached) for monitoring
                    # shape: (N, emb_size)
                    self.last_gate = z.detach()
                    return z * curr + (1.0 - z) * prev

            self._agate_user = AGate(emb_size)
            self._agate_movie = AGate(emb_size)

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

        # carry_alpha controls how strongly previous snapshot states
        # influence the current forward pass. default 0.0 (no carry-over).
        alpha = float(self._hparams.get("carry_alpha", 0.0))

        user_embeddings = []
        movie_embeddings = []

        # Start from the current embedding weights and either fuse with
        # previous states via a learnable AGate or simple additive carry.
        if self._use_agate:
            # detach previous states to avoid backpropagating through long
            # snapshot histories (stability).
            user_embedding = self._agate_user(self.prev_user_state.detach(), self._user_embedding.weight)
            movie_embedding = self._agate_movie(self.prev_movie_state.detach(), self._movie_embedding.weight)
        else:
            user_embedding = self._user_embedding.weight + alpha * self.prev_user_state
            movie_embedding = self._movie_embedding.weight + alpha * self.prev_movie_state

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

        # Return output and the final propagated user/movie states so
        # the training loop can persist them between snapshots.
        return output.view(-1), user_embeddings[-1], movie_embeddings[-1]
