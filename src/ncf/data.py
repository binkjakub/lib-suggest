import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.loader import load_train_test_interactions
from src.feature_extraction.features import get_features_tensor


class RepoLibDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self,
                 repo_tensor: Tensor,
                 lib_tensor: Tensor,
                 target_tensor: Tensor,
                 repo_feats: Optional[Tensor] = None,
                 ):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        assert (len(repo_tensor) == len(lib_tensor) == len(target_tensor))
        if repo_feats is not None:
            assert (len(repo_feats) == len(repo_tensor))

        self.repo_tensor = repo_tensor
        self.lib_tensor = lib_tensor
        self.target_tensor = target_tensor
        self.repo_feats = repo_feats

    def __getitem__(self, index):
        if self.repo_feats is None:
            return self.repo_tensor[index], self.lib_tensor[index], self.target_tensor[index]
        else:
            return (self.repo_tensor[index],
                    self.lib_tensor[index],
                    self.target_tensor[index],
                    self.repo_feats[index])

    def __len__(self):
        return self.repo_tensor.size(0)


class LibRecommenderDM(LightningDataModule):
    """Datamodule responsible for serving dataset to be used in training of NCF model."""

    VAL_BATCH_SIZE = 512

    def __init__(self, config: Dict):
        super().__init__()

        self._use_repo_features = bool(config['manual_feat_dim'])
        self._num_negatives = config['num_negatives']
        self._batch_size = config['batch_size']
        self._num_workers = config['num_workers']

        self.train_ratings: Optional[RepoLibDataset] = None
        self.test_ratings: Optional[RepoLibDataset] = None

        self.repo_names: Optional[np.ndarray] = None
        self.lib_names: Optional[np.ndarray] = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        repo_feats, train_interactions, test_interactions, self.repo_names, self.lib_names = \
            load_train_test_interactions()

        self._log_dataset_stats(train_interactions, "Train")
        self._log_dataset_stats(test_interactions, "Test")

        negatives = self._sample_negatives(train_interactions)

        self.train_ratings = self._build_dataset(train_interactions, negatives, repo_feats)
        self.test_ratings = self._build_dataset(test_interactions, negatives, repo_feats)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_ratings, batch_size=self._batch_size, shuffle=True,
                          num_workers=self._num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_ratings, batch_size=self.VAL_BATCH_SIZE, shuffle=False,
                          num_workers=self._num_workers)

    def _build_dataset(self,
                       ratings: pd.DataFrame,
                       negatives: pd.DataFrame,
                       repo_feats: pd.DataFrame) -> RepoLibDataset:
        """Creates training data."""
        train_ratings = pd.merge(ratings, negatives[['full_name', 'negative_items']],
                                 on='full_name')

        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(x, self._num_negatives))

        repos, libs, ratings = [], [], []
        for _, row in train_ratings.iterrows():
            repos.append(int(row['full_name']))
            libs.append(int(row['repo_requirements']))
            ratings.append(float(row['rating']))
            for i in range(self._num_negatives):
                repos.append(int(row['full_name']))
                libs.append(int(row['negatives'][i]))
                ratings.append(float(0))

        repos = torch.tensor(repos, dtype=torch.int)
        libs = torch.tensor(libs, dtype=torch.int)
        ratings = torch.tensor(ratings, dtype=torch.float)

        if self._use_repo_features:
            feats = get_features_tensor(self.repo_names, repo_feats, ratings)
        else:
            feats = None

        return RepoLibDataset(repos, libs, ratings, feats)

    def _sample_negatives(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Returns all negative items & 100 sampled negative items (for evaluation purposes)."""
        item_pool = set(ratings['repo_requirements'].unique())
        interact_status = ratings.groupby('full_name')['repo_requirements'].apply(
            set).reset_index().rename(
            columns={'repo_requirements': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(
            lambda x: item_pool - x)
        interact_status['test_negatives'] = interact_status['negative_items'].apply(
            lambda x: random.sample(x, 99))
        return interact_status[['full_name', 'negative_items', 'test_negatives']]

    @staticmethod
    def _log_dataset_stats(dataset: pd.DataFrame, name: str = ""):
        sparsity = (len(dataset)
                    / (dataset['full_name'].nunique() * dataset['repo_requirements'].nunique()))

        print(f"{name} dataset", "_".ljust(60, "_"))
        print(f"\t#repos: {dataset['full_name'].nunique()}")
        print(f"\t#libraries: {dataset['repo_requirements'].nunique()}")
        print(f"\t#interactions: {len(dataset)}")
        print(f"\tsparsity: {sparsity:0.3f}")
