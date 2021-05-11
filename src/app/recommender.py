from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.data.crawling.github_crawler import GithubCrawler


@dataclass
class RecommendationResult:
    recommendation: list[str]
    repository: dict[str, Any]


class Recommender(ABC):
    def __init__(self, crawler: GithubCrawler):
        self.crawler = crawler

    @abstractmethod
    def recommend(self, repo_name: str) -> RecommendationResult:
        """Recommends libraries for repository based on its current requirements."""


class DummyRecommender(Recommender):
    """Mock recommendations."""

    def recommend(self, repo_name: str) -> RecommendationResult:
        repository = self.crawler.crawl_extract_repository(repo_name)
        return RecommendationResult(['writeItYourself'], repository)
