from typing import Any, List, Optional

import torch
import torch.nn as nn

from src.ncf.recommender import RecommenderSystem


class UserEmbedding(nn.Module):
    def __init__(self, num_users: int,
                 embedding_dim: int,
                 manual_feat_dim: Optional[int] = None,
                 manual_feat_combination_out_dim: Optional[int] = None):
        super().__init__()
        self.embedding_user = nn.Embedding(num_embeddings=num_users,
                                           embedding_dim=embedding_dim)

        self.leverage_user_feats = bool(manual_feat_dim)
        if manual_feat_dim and manual_feat_combination_out_dim:
            self.combination_layer = nn.Linear(embedding_dim + manual_feat_dim,
                                               manual_feat_combination_out_dim)

    def forward(self, user_indices, **kwargs):
        repr_user = self.embedding_user(user_indices)
        if self.leverage_user_feats:
            user_feats = kwargs['user_features']
            repr_user = torch.cat([repr_user, user_feats], dim=-1)
        if hasattr(self, 'combination_layer'):
            repr_user = self.combination_layer(repr_user)
        return repr_user


class GMF(RecommenderSystem):
    """Generalized Matrix Factorization"""

    def __init__(self, config: dict[str, Any]):
        super(GMF, self).__init__(config)

        self.embedding_user = UserEmbedding(self.hparams.num_repos,
                                            self.hparams.latent_dim_mf,
                                            self.hparams.get('manual_feat_dim'),
                                            self.hparams.get('manual_feat_combination_out_dim'))
        self.embedding_item = nn.Embedding(num_embeddings=self.hparams.num_libs,
                                           embedding_dim=self.hparams.latent_dim_mf)

        self.affine_output = nn.Linear(in_features=self.hparams.latent_dim_mf, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices, user_features=None):
        repr_vector = self.forward_repr(user_indices, item_indices, user_features=user_features)
        logits = self.affine_output(repr_vector)
        rating = self.logistic(logits)
        return rating

    def forward_repr(self, user_indices, item_indices, *args, **kwargs):
        user_embedding = self.embedding_user(user_indices, *args, **kwargs)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        return element_product


class MLP(RecommenderSystem):
    """Multilayer Perceptron model."""

    def __init__(self, config):
        super(MLP, self).__init__(config)

        self.embedding_user = UserEmbedding(self.hparams.num_repos,
                                            self.hparams.embedding_dim,
                                            self.hparams.get('manual_feat_dim'),
                                            self.hparams.get('manual_feat_combination_out_dim'))
        self.embedding_item = nn.Embedding(num_embeddings=self.hparams.num_libs,
                                           embedding_dim=self.hparams.embedding_dim)

        self.fc_layers = nn.Sequential(*self._build_hidden_layers())
        self.affine_output = nn.Linear(in_features=self.hparams.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

        print(self.fc_layers)

    def forward(self, user_indices, item_indices, user_features=None):
        repr_vector = self.forward_repr(user_indices, item_indices, user_features=user_features)
        logits = self.affine_output(repr_vector)
        ratings = self.logistic(logits)
        return ratings

    def forward_repr(self, user_indices, item_indices, *args, **kwargs):
        user_embedding = self.embedding_user(user_indices, *args, **kwargs)
        item_embedding = self.embedding_item(item_indices)

        out = torch.cat([user_embedding, item_embedding], dim=-1)
        out = self.fc_layers(out)
        return out

    def _build_hidden_layers(self) -> List[nn.Module]:
        layers = []
        for in_size, out_size in zip(self.hparams.layers, self.hparams.layers[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        return layers


class NeuMF(RecommenderSystem):
    def __init__(self, config: dict[str, Any]):
        super(NeuMF, self).__init__(config)
        self._gmf = GMF(config)
        self._mlp = MLP(config)

        predictive_factors = self.hparams.layers[-1] + self.hparams.latent_dim_mf
        self.affine_output = nn.Linear(in_features=predictive_factors, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        mf_vector = self._gmf.forward_repr(user_indices, item_indices)
        mlp_vector = self._mlp.forward_repr(user_indices, item_indices)

        vector = torch.cat([0.5 * mlp_vector, 0.5 * mf_vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
