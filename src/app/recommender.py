from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from src.data.crawling.github_crawler import GithubCrawler

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
