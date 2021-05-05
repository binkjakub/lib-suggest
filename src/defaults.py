import os
from pathlib import Path

DVC_PATH = Path(os.path.dirname(__file__)).parent.joinpath('storage').absolute()
DATASET_PATH = DVC_PATH / 'dataset'

REPOS = DATASET_PATH / f"scraped_repos_{os.environ['USER']}.jsonl"
