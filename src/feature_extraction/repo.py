import multiprocessing
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import requirements
import srsly
from github import Repository, UnknownObjectException
from tqdm import tqdm

from src.defaults import RAW_DATA, REPOS


def extract_batch(batch: list[Repository], store_dir: Path = RAW_DATA, n_jobs: int = 1) -> None:
    store_dir.mkdir(exist_ok=True)

    with multiprocessing.Pool(n_jobs) as pool:
        p_bar = tqdm(
            pool.imap(safe_extract_repo, batch),
            desc="Extracting features",
            leave=True,
            total=len(batch),
        )
        status = {'dropped': 0, 'extracted': 0}
        with open(REPOS, 'a+') as file:

            for extracted in p_bar:
                if extracted is not None:
                    file.write(srsly.json_dumps(extracted) + '\n')
                    status['extracted'] += 1
                else:
                    status['dropped'] += 1
                    p_bar.set_postfix(status)


def safe_extract_repo(repo: Repository) -> Optional[dict[str, Any]]:
    try:
        return extract_repo(repo)
    except Exception as exc:
        print(f'Failed to extract {exc}. Continuing...')
        return None


def extract_repo(repo: Repository) -> Optional[dict[str, Any]]:
    repo_requirements = _get_requirements_names(repo)
    if repo_requirements is None or len(repo_requirements) == 0:
        return None
    else:
        repo_requirements = list(set(lib for lib in repo_requirements if lib is not None))
        if not repo_requirements:
            return None

    result = OrderedDict()
    result['full_name'] = repo.full_name
    result['created_at'] = repo.created_at
    result['last_modified'] = repo.last_modified
    result['description'] = repo.description
    result['n_subscribers'] = repo.subscribers_count
    result['n_stars'] = repo.stargazers_count
    result['n_forks'] = repo.forks_count
    result['n_open_issues'] = repo.get_issues(state='open').totalCount
    result['n_closed_issues'] = repo.get_issues(state='closed').totalCount
    result['n_all_issues'] = result['n_open_issues'] + result['n_closed_issues']
    # result['n_root_contents'] = len(repo.get_contents(''))
    # result['n_all_contents'] = _n_all_contents(repo)
    result['n_branches'] = repo.get_branches().totalCount
    result['is_master_protected'] = repo.get_branch("master").protected
    result['n_pr_open'] = repo.get_pulls(state='open', base='master').totalCount
    result['n_pr_closed'] = repo.get_pulls(state='closed', base='master').totalCount
    result['n_pr_all'] = result['n_pr_open'] + result['n_pr_closed']
    result['n_milestones_open'] = repo.get_milestones(state='open').totalCount
    result['n_milestones_closed'] = repo.get_milestones(state='closed').totalCount
    result['n_milestones_all'] = result['n_milestones_open'] + result['n_milestones_closed']
    result['readme_text'] = _content_text(repo, 'README.md')
    result['repo_requirements'] = repo_requirements
    return result


def _n_all_contents(repo: Repository) -> int:
    num = 0
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            num += 1

    return num


def _content_text(repo: Repository, content_name: str) -> Optional[str]:
    try:
        return repo.get_contents(content_name).decoded_content.decode("utf-8")
    except UnknownObjectException:
        return None


def _get_requirements_names(repo: Repository) -> Optional[list[str]]:
    if requirements_text := _content_text(repo, 'requirements.txt'):
        return [req.name for req in requirements.parse(requirements_text)]
    else:
        return None
