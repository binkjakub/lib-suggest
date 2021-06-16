import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Union

from github import Github
from github.GithubException import RateLimitExceededException, UnknownObjectException
from github.PaginatedList import PaginatedList
from github.Repository import Repository

from src.feature_extraction.repo import extract_repo


class GithubCrawler:
    """Wrapper class for a GitHub crawler."""

    def __init__(self, login_or_token: Optional[str] = None, password: Optional[str] = None):
        self.g = Github(login_or_token=login_or_token, password=password)

    def crawl_extract_repository(self, repo_name: str) -> Optional[dict[str, Any]]:
        repo = self.g.get_repo(repo_name)
        return extract_repo(repo)

    def search_repos(self, query, **kwargs) -> Union[Repository, PaginatedList]:
        return self.g.search_repositories(query, **kwargs)

    def crawl_between_dates(self,
                            query: str,
                            start_date: date,
                            end_date: date,
                            step_day: int,
                            **kwargs) -> Dict[str, PaginatedList]:
        """
        Crawls query within given days step_day argument as a window.

        This is to get around GitHub API limitations which let you retrieve only 1000 first search
        results.
        """
        crawled_dates = {}
        query += ' created:{}..{}'
        delta = end_date - start_date
        for i in range(0, delta.days + 1, step_day):
            from_day = start_date + timedelta(days=i)
            until_day = start_date + timedelta(days=i + step_day - 1)
            q = query.format(from_day, until_day)
            crawled_dates[str(from_day)] = self.search_repos(q, **kwargs)
        return crawled_dates

    @staticmethod
    def validate_repo(repo: Repository) -> bool:
        try:
            repo.get_contents('requirements.txt')
            return True
        except UnknownObjectException:
            return False

    def paginated_list_to_repos(self,
                                paginated_list: PaginatedList,
                                limit: Optional[int] = None) -> List[dict]:
        """Converts PaginatedList of repositories to actual Repositories list.

        This is where the GitHub API fetch each repository and the limits are often exceed.
        """
        retrieved_repos = []
        iter_paginated = iter(paginated_list)
        while True:
            if limit is not None and len(retrieved_repos) >= limit:
                break
            try:
                repo = next(iter_paginated)
                retrieved_repos.append(repo)
            except StopIteration:
                print(f"Fetched all {len(retrieved_repos)} repos!")
                break
            except RateLimitExceededException:
                search_rate_limit = self.g.get_rate_limit().search
                print(f"Exceeded rate limit!\nSearch remaining: {search_rate_limit.remaining}\n")
                print("Waiting 120 seconds...")
                time.sleep(120)
        return retrieved_repos


class MockedCrawler(GithubCrawler):
    """Mocked crawler."""

    def crawl_extract_repository(self, *args, **kwargs) -> dict[[str, Any]]:
        return {}
