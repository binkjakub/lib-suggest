from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import KBinsDiscretizer

from src.data.crawling.github_crawler import GithubCrawler
from src.defaults import REPO_DS, TRAIN_DS, TEST_DS

NUM_FEATURES = ['is_master_protected', 'n_all_issues', 'n_branches', 'n_closed_issues', 'n_forks',
                'n_milestones_all', 'n_milestones_closed', 'n_milestones_open',
                'n_open_issues', 'n_pr_all', 'n_pr_closed', 'n_pr_open', 'n_stars']


class LightFMModel:
    def __init__(self, crawler: Optional[GithubCrawler]):
        self.crawler = crawler
        self.dataset: Dataset
        self.num_users: int
        self.num_items: int
        self.model: LightFM
        self.train_interactions: scipy.sparse.coo_matrix
        self.train: pd.DataFrame
        self.test: pd.DataFrame
        self.user_features_type: str
        self.repo_features: pd.DataFrame
        self.user_features: Optional[scipy.sparse.csr.csr_matrix] = None

    @cached_property
    def item_mapping(self) -> dict:
        return {v: k for k, v in self.dataset.mapping()[2].items()}

    def prepare(self, train_path: str = TRAIN_DS, test_path: str = TEST_DS,
                user_features_path: str = REPO_DS):
        self.train = pd.read_csv(train_path).sort_values(by='full_name')
        self.test = pd.read_csv(test_path)
        self.repo_features = pd.read_csv(user_features_path)

    def fit(self,
            train: Optional[pd.DataFrame],
            user_features_type: Optional[str],
            num_epochs: int = 15, **kwargs):
        """Fit the model to predefined data."""
        assert user_features_type in [None, 'description', 'numerical']
        if train is None:
            train = self.train
        self.user_features_type = user_features_type
        self.dataset = Dataset()
        self.dataset.fit(users=iter(train['full_name'].unique()),
                         items=iter(train['repo_requirements'].unique()),
                         )
        self.user_features = self.get_user_features(self.user_features_type)

        self.num_users, self.num_items = self.dataset.interactions_shape()

        self.train_interactions, _ = self.dataset.build_interactions(
            [x for x in train.itertuples(index=False, name=None)])

        self.model = LightFM(**kwargs)
        self.model.fit(interactions=self.train_interactions, user_features=self.user_features,
                       epochs=num_epochs, num_threads=2)

    def refit_model(self, data: pd.DataFrame, repo_features: pd.DataFrame, num_epochs: int = 15):
        self.dataset.fit_partial(users=iter(data['full_name'].unique()),
                                 items=iter(data['repo_requirements'].unique()))
        self.train = pd.concat([self.train, data])
        self.repo_features = pd.concat([self.repo_features, repo_features])
        self.user_features = self.get_user_features(self.user_features_type)
        self.train_interactions, _ = self.dataset.build_interactions(
            [x for x in self.train.itertuples(index=False, name=None)])
        self.num_users, self.num_items = self.dataset.interactions_shape()
        self.model.fit(interactions=self.train_interactions, user_features=self.user_features,
                       epochs=num_epochs, num_threads=2)

    def predict_repo(self, repo_name: str, n: int = 5) -> list[str]:
        repo_id = self.dataset.mapping()[0][repo_name]
        scores = self.model.predict(
            user_ids=repo_id,
            item_ids=np.delete(np.arange(self.num_items),
                               self.train_interactions.tocsr()[repo_id].indices))
        top_items = np.argsort(-scores)
        return [self.item_mapping[i] for i in top_items[:n]]

    def get_user_features(self, user_features_type: Optional[str]
                          ) -> Optional[scipy.sparse.csr.csr_matrix]:
        if user_features_type == 'description':
            return self.build_user_text_features()
        elif user_features_type == 'numerical':
            return self.build_user_numerical_features()
        else:
            return None

    def build_user_text_features(self) -> scipy.sparse.csr.csr_matrix:
        """Build repository textual features, e.g. description."""
        self.dataset.fit_partial(items=(self.repo_features['full_name']),
                                 user_features=(self.repo_features['description']))
        user_features = self.dataset.build_user_features((
            (x.full_name, [x.description]) for x in self.repo_features.itertuples()))
        return user_features

    def build_user_numerical_features(self) -> scipy.sparse.csr.csr_matrix:
        """Build repository textual features, e.g. no. stars, no. forks etc."""
        repo_features = self.repo_features.copy()
        repo_features[NUM_FEATURES] = repo_features[NUM_FEATURES].astype(int)
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        repo_features[NUM_FEATURES] = pd.DataFrame(est.fit_transform(repo_features[NUM_FEATURES]),
                                                   columns=NUM_FEATURES)
        repo_features[NUM_FEATURES] = repo_features[NUM_FEATURES].astype(int)
        repo_features[NUM_FEATURES] = repo_features[NUM_FEATURES].astype(str)
        for f in NUM_FEATURES:
            repo_features[f] = f + '_' + repo_features[f]
        unique = np.unique(repo_features[NUM_FEATURES].values)
        self.dataset.fit_partial(items=repo_features['full_name'],
                                 user_features=unique)
        data = ((x.full_name, [getattr(x, col) for col in NUM_FEATURES])
                for x in repo_features.itertuples())

        user_features = self.dataset.build_user_features(data)
        return user_features

    def evaluate(self, test: Optional[pd.DataFrame], k: int = 5, k_ndcg: int = 5) -> dict:
        if test is None:
            test = self.test
        test_interactions, _ = self.dataset.build_interactions(
            [x for x in test.itertuples(index=False, name=None)])
        test_recall = recall_at_k(self.model,
                                  test_interactions,
                                  self.train_interactions,
                                  user_features=self.user_features,
                                  k=k).mean()
        test_precisions = precision_at_k(self.model,
                                         test_interactions,
                                         self.train_interactions,
                                         user_features=self.user_features,
                                         k=k).mean()
        preds = []
        for i in range(len(test)):
            preds.append(self.model.predict(i, np.arange(401), user_features=self.user_features))
        preds = np.vstack(preds)
        tr_ndcg = ndcg_score(y_true=test_interactions.toarray() + self.train_interactions.toarray(),
                             y_score=preds,
                             k=k_ndcg)
        ndcg = ndcg_score(y_true=test_interactions.toarray(),
                          y_score=preds,
                          k=k_ndcg)
        return {f'recall@{k}': test_recall, f'precision@{k}': test_precisions,
                f'train_ndcg@{k_ndcg}': tr_ndcg, f'test_ndcg@{k_ndcg}': ndcg}
