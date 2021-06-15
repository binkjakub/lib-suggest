from abc import ABC
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import (
    MetricCollection, RetrievalNormalizedDCG, RetrievalPrecision, RetrievalRecall)


class RecommenderSystem(LightningModule, ABC):
    def __init__(self, config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)

        self.k_eval = config['k_eval']
        metrics = MetricCollection({
            f'Precision@{self.k_eval}': RetrievalPrecision(k=self.k_eval),
            f'Recall@{self.k_eval}': RetrievalRecall(k=self.k_eval),
            f'NDCG@{self.k_eval}': RetrievalNormalizedDCG(k=self.k_eval),
        })
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, train_batch, batch_id):
        repos, libs, ratings = train_batch
        ratings_pred = self.forward(repos, libs).squeeze(-1)

        loss = F.binary_cross_entropy(ratings_pred, ratings)
        self.log('train/loss', loss)

        self.train_metrics(ratings_pred, ratings.long(), indexes=repos.long())

        return loss

    def validation_step(self, val_batch, batch_id):
        repos, libs, ratings = val_batch
        ratings_pred = self.forward(repos, libs).squeeze(-1)

        loss = F.binary_cross_entropy(ratings_pred, ratings)
        self.log('val/loss', loss)

        self.val_metrics(ratings_pred, ratings.long(), indexes=repos.long())

    def training_epoch_end(self, outputs):
        self._aggregate_and_log_metrics(self.train_metrics)

    def validation_epoch_end(self, outputs):
        self._aggregate_and_log_metrics(self.val_metrics)

    def _aggregate_and_log_metrics(self,
                                   metrics: MetricCollection,
                                   progress_bar: bool = False) -> dict:
        metric_values = metrics.compute()
        metrics.reset()
        self.log_dict(metric_values, prog_bar=progress_bar)
        return metric_values
