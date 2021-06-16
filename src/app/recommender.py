from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import pandas as pd
from numpy.random import PCG64
from surprise import Dataset, KNNBasic, Reader

from src.app.utils import get_top_n
from src.data.crawling.github_crawler import GithubCrawler
from src.dataset.prepare import prepare_interactions

T = TypeVar('T')


@dataclass
class RecommendationResult(Generic[T]):
    recommendation: T
    repository: dict[str, Any]


class Recommender(ABC, Generic[T]):
    def __init__(self, crawler: Optional[GithubCrawler] = None):
        self.crawler: Optional[GithubCrawler] = crawler

    @property
    def name(self) -> str:
        return type(self).__name__

    def crawl_and_recommend(self, repo_name: str) -> RecommendationResult:
        """Crawls repository and recommends libraries based on the current requirements."""
        assert self.crawler is not None
        repository = self.crawler.crawl_extract_repository(repo_name)
        recommendations = self.recommend(repository)
        return RecommendationResult(recommendations, repository)

    @abstractmethod
    def recommend(self, repository: dict[str, Any]) -> T:
        """Recommends libraries to use, based on the repository features."""


class RecommenderCollection(Recommender):
    def __init__(self, crawler: GithubCrawler, recommenders: list[Recommender[list[str]]]):
        super().__init__(crawler)
        self.recommenders: list[Recommender] = recommenders

    def recommend(self, repository: dict[str, Any]) -> dict[str, list[str]]:
        recommendations = {}
        for recommender_predictor in self.recommenders:
            recommendations[recommender_predictor.name] = \
                recommender_predictor.recommend(repository)

        return recommendations


class DummyRecommender(Recommender):
    """Mock recommendations."""

    def recommend(self, repository: dict[str, Any]) -> list[str]:
        return ['writeItYourself']


class KNNRecommender(Recommender):
    def __init__(self, crawler: GithubCrawler):
        super().__init__(crawler)
        # _, self.algo = surprise.dump.load(KNN_MODEL_PATH)
        self.train, _ = prepare_interactions()

    def sample_neg_packages(self, x: pd.DataFrame, n: int):
        generator = PCG64(12331)
        negatives = []
        while len(negatives) < n:
            [random_package] = self.train.sample(random_state=generator)['repo_requirements']
            if random_package not in x.repo_requirements.values:
                negatives.append({'full_name': x.full_name.values[0],
                                  'repo_requirements': random_package})

        negatives = pd.DataFrame(negatives)
        negatives['rating'] = 0
        return negatives

    def recommend(self, repository: dict[str, Any]) -> T:
        x = pd.DataFrame(repository).explode('repo_requirements')[
            ['full_name', 'repo_requirements']]
        x['rating'] = 1
        x_neg = self.sample_neg_packages(x, len(x))
        train = pd.concat([x, x_neg, self.train])
        trainset = Dataset.load_from_df(train, Reader(rating_scale=(0, 1))).build_full_trainset()

        algo = KNNBasic(sim_options={'name': 'pearson'})
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        top = get_top_n(predictions)[repository['full_name']]
        return [package for package, _ in top]
