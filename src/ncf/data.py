import random
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.loader import load_train_test_interactions


class RepoLibDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor: Tensor, item_tensor: Tensor, target_tensor: Tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.repo_tensor = user_tensor
        self.lib_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.repo_tensor[index], self.lib_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.repo_tensor.size(0)


class RepoLibEvaluationDataset(Dataset):
    """Wrapper for evaluation data, as recommendations are evaluated with 100 negative samples:
    Data should contain:
        - examples of positive interactions
        - subsample of negative interactions (preferably 100, as in NCF publication)
    """

    def __init__(self, test_users: Tensor, test_items: Tensor, negative_users: Tensor,
                 negative_items: Tensor):
        self._repos_idx = torch.unique(test_users)

        self.test_repos = test_users
        self.test_libs = test_items
        self.negative_repos = negative_users
        self.negative_libs = negative_items

    def __getitem__(self, repo_index):
        positive_items = self.test_libs[self.test_repos == repo_index]
        negative_items = self.negative_libs[self.negative_repos == repo_index]
        total_len = len(positive_items) + len(negative_items)

        repos = torch.cat([positive_items, negative_items])
        libs = torch.full((total_len,), repo_index, dtype=torch.int)
        ratings = torch.tensor(([1] * len(positive_items) + ([0] * len(negative_items))),
                               dtype=torch.int)
        return libs, repos, ratings

    def __len__(self):
        return self._repos_idx.size(0)


class LibRecommenderDM(LightningDataModule):
    """Datamodule responsible for preparation of dataset to be used in training of NCF model."""

    def __init__(self, config: Dict):
        super().__init__()

        self._num_negatives = config['num_negatives']
        self._batch_size = config['batch_size']

        self._num_workers = config['num_workers']
        # self._num_repos, self._num_libs = config['num_repos'], config['num_libs']

        self.train_ratings = None
        self.test_ratings = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        train_interactions, test_interactions = load_train_test_interactions()

        self._log_dataset_stats(train_interactions, "Train")
        self._log_dataset_stats(test_interactions, "Test")

        negatives = self._sample_negatives(train_interactions)
        self.train_ratings = self._build_train_dataset(train_interactions, negatives)
        self.test_ratings = self._build_test_dataset(test_interactions, negatives)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_ratings, batch_size=self._batch_size, shuffle=True,
                          num_workers=self._num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_ratings, batch_size=1, shuffle=False,
                          num_workers=self._num_workers)

    def _build_train_dataset(self, ratings: pd.DataFrame,
                             negatives: pd.DataFrame) -> RepoLibDataset:
        """Creates training data."""
        train_ratings = pd.merge(ratings, negatives[['full_name', 'negative_items']],
                                 on='full_name')

        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(x, self._num_negatives))

        users, items, ratings = [], [], []
        for _, row in train_ratings.iterrows():
            users.append(int(row['full_name']))
            items.append(int(row['repo_requirements']))
            ratings.append(float(row['rating']))
            for i in range(self._num_negatives):
                users.append(int(row['full_name']))
                items.append(int(row['negatives'][i]))
                ratings.append(float(0))

        return RepoLibDataset(
            torch.tensor(users, dtype=torch.int),
            torch.tensor(items, dtype=torch.int),
            torch.tensor(ratings, dtype=torch.float),
        )

    def _build_test_dataset(self,
                            test_ratings: pd.DataFrame,
                            negatives: pd.DataFrame) -> RepoLibEvaluationDataset:
        """Creates evaluation data."""
        test_ratings = pd.merge(test_ratings,
                                negatives[['full_name', 'test_negatives']],
                                on='full_name')

        test_users, test_items, negative_users, negative_items = [], [], [], []

        for _, row in test_ratings.iterrows():
            test_users.append(int(row['full_name']))
            test_items.append(int(row['repo_requirements']))
            for i in range(len(row['test_negatives'])):
                negative_users.append(int(row['full_name']))
                negative_items.append(int(row['test_negatives'][i]))

        return RepoLibEvaluationDataset(
            torch.tensor(test_users, dtype=torch.int),
            torch.tensor(test_items, dtype=torch.int),
            torch.tensor(negative_users, dtype=torch.int),
            torch.tensor(negative_items, dtype=torch.int),
        )

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
