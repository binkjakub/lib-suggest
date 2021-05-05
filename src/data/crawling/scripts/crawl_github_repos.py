"""
This script retrieves PaginatedList for the given query and dates, extract features from
the Repository object and saves it to json.
"""
import multiprocessing
import queue
import time
from datetime import datetime, timedelta
from multiprocessing import Process
from random import randrange
from typing import Optional

import click

from src.data.crawling.github_crawler import GithubCrawler
from src.feature_extraction.repo import extract_batch


@click.command()
@click.option('--access-token', '-a', type=click.STRING, required=True,
              help="Access token generated with GitHub account or None in case of using public API"
                   "(limits are stricter in case of public API)")
@click.option('--query', type=click.STRING, default="language:python")
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), default="2017-01-01")
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), default="2020-01-07")
@click.option('--step-day', type=click.INT, default=10)
@click.option('--daily-limit', type=click.INT, default=10)
@click.option('--n-jobs', type=click.INT, default=1)
@click.option('--repeat-span', type=click.INT, default=30)
def crawl_github_repos(access_token: str,
                       query: str,
                       start_date: datetime,
                       end_date: datetime,
                       step_day: int,
                       daily_limit: int,
                       n_jobs: int,
                       repeat_span: int):
    while True:
        random_start_date = random_date(start_date, end_date)
        random_end_date = random_date(random_start_date, end_date)

        print(random_start_date, '-', random_end_date)
        execute(access_token, query, start_date, end_date, step_day, daily_limit, n_jobs)
        time.sleep(repeat_span * 60)


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def execute(access_token: str,
            query: str,
            start_date: datetime,
            end_date: datetime,
            step_day: int,
            daily_limit: int,
            n_jobs: int):
    repos_queue = multiprocessing.Queue()
    args = (repos_queue, access_token, query, start_date, end_date, step_day, daily_limit)
    search_proc = Process(target=_worker_repo_list_scraper, args=args)
    search_proc.start()

    while search_proc.is_alive() or repos_queue.qsize() > 0:
        try:
            repos = repos_queue.get(timeout=30)
            extract_batch(repos, n_jobs=n_jobs)
        except queue.Empty:
            time.sleep(60)

    search_proc.join()


def _worker_repo_list_scraper(repos_queue: multiprocessing.Queue,
                              access_token: str,
                              query: str,
                              start_date: datetime,
                              end_date: datetime,
                              step_day: int,
                              daily_limit: Optional[int]):
    github_crawler = GithubCrawler(access_token)
    crawled_pages = github_crawler.crawl_between_dates(query=query,
                                                       start_date=start_date.date(),
                                                       end_date=end_date.date(),
                                                       step_day=step_day)
    for paginated_list in crawled_pages.values():
        repos = github_crawler.paginated_list_to_repos(paginated_list, daily_limit)
        repos_queue.put(repos)
    return


crawl_github_repos()
