import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import requirements
import srsly
from github import Repository, UnknownObjectException
from tqdm import tqdm

from src.defaults import DATASET_PATH


def extract_batch(batch: list[Repository], store_dir: Path = DATASET_PATH) -> None:
    store_dir.mkdir(exist_ok=True)
    to_store = []
    for repo in tqdm(batch):
        try:
            to_store.append(extract_repo(repo))
        except Exception:
            print('Failed to extract. Continuing...')
    name = hashlib.sha1(
        ' '.join([features['full_name'] for features in to_store]).encode()).hexdigest()
    name += '.json'
    srsly.write_json(store_dir / name, to_store)


def extract_repo(repo: Repository) -> dict[str, Any]:
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
    result['n_all_issues'] = repo.get_issues(state='all').totalCount
    result['n_root_contents'] = len(repo.get_contents(''))
    result['n_all_contents'] = _n_all_contents(repo)
    result['n_branches'] = repo.get_branches().totalCount
    result['is_master_protected'] = repo.get_branch("master").protected
    result['n_pr_open'] = repo.get_pulls(state='open', base='master').totalCount
    result['n_pr_closed'] = repo.get_pulls(state='closed', base='master').totalCount
    result['n_pr_all'] = repo.get_pulls(state='all', base='master').totalCount
    result['n_milestones_open'] = repo.get_milestones(state='open').totalCount
    result['n_milestones_closed'] = repo.get_milestones(state='closed').totalCount
    result['n_milestones_all'] = repo.get_milestones(state='all').totalCount
    result['readme_text'] = _content_text(repo, 'README.md')
    result['requirements'] = _get_requirements_names(repo)
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
