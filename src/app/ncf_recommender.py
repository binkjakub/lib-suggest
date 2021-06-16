from typing import Any, Optional

import numpy as np
import torch

from src.app.recommender import Recommender
from src.data.crawling.github_crawler import GithubCrawler
from src.data.loader import PathLike
from src.feature_extraction.features import FEATURE_NAMES
from src.ncf.models import MLP


class NCFRecommender(Recommender):
    """Recommendation based on Neural Collaborative Filtering (NCF) model."""

    N_RECOMMENDATIONS = 5

    def __init__(self,
                 checkpoint_path: PathLike,
                 n_recommendations: int = N_RECOMMENDATIONS,
                 crawler: Optional[GithubCrawler] = None,
                 ):
        super().__init__(crawler)
        self.model = MLP.load_from_checkpoint(checkpoint_path)

        self._input_libraries = torch.arange(self.model.hparams.num_libs)
        self._num_libraries = len(self._input_libraries)
        self._n_recommendation = n_recommendations

    def recommend(self, repository: dict[str, Any]) -> list[str]:
        if self.model.hparams.manual_feat_dim:
            feats = torch.tensor([repository.get(feat, 0) for feat in FEATURE_NAMES])
            feats = torch.repeat_interleave(feats[None, :], repeats=self._num_libraries, dim=0)
        else:
            feats = None
        repo_index = torch.full(size=(self._num_libraries,),
                                fill_value=self._get_repository_index(repository['full_name']))
        pred_ratings = self.model.forward(repo_index, self._input_libraries, feats)
        pred_ratings = pred_ratings.flatten()
        top_recommended = torch.argsort(pred_ratings)[:self._n_recommendation].tolist()
        top_recommended = self._get_lib_names(top_recommended)
        return top_recommended

    def _get_repository_index(self, repository_name: str) -> int:
        matching_repos = np.where(self.model.repo_names == repository_name)
        if matching_repos[0]:
            (repo_index, *_), *_ = matching_repos
            return repo_index
        else:
            return self.model.repo_oov_index

    def _get_lib_names(self, library_index: list[int]) -> list[str]:
        return [self.model.lib_names[lib_idx] for lib_idx in library_index]
