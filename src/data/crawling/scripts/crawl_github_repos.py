"""
This script retrieves PaginatedList for the given query and dates, extract features from
the Repository object and saves it to json.
"""
from datetime import datetime
from typing import Optional

import click
from tqdm import tqdm

from src.data.crawling.github_crawler import GithubCrawler
from src.feature_extraction.repo import extract_batch


@click.command()
@click.option('--access-token', type=click.STRING,
              default='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',
              help="Access token generated with GitHub account or None in case of using public API"
                   "(limits are stricter in case of public API)")
@click.option('--query', type=click.STRING, default="language:python")
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), default="2020-01-01")
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), default="2020-01-07")
@click.option('--step-day', type=click.INT, default=1)
def crawl_github_repos(access_token: Optional[str], query: str, start_date: datetime,
                       end_date: datetime, step_day: int):
    github_crawler = GithubCrawler(access_token)
    crawled_pages = github_crawler.crawl_between_dates(query=query, start_date=start_date.date(),
                                                       end_date=end_date.date(), step_day=step_day)
    for date, paginated_list in tqdm(crawled_pages.items()):
        repos = github_crawler.paginated_list_to_repos(paginated_list)
        extract_batch(repos)


crawl_github_repos()
