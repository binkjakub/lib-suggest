import os
from pathlib import Path

DVC_PATH = Path(os.path.dirname(__file__)).parent.joinpath('storage').absolute()

RAW_DATA = DVC_PATH / 'raw_data'
REPOS = RAW_DATA / f"scraped_repos_{os.environ['USER']}.jsonl"

DATASET_DIR = DVC_PATH / 'dataset'

REPO_DS = DATASET_DIR / 'repo_features.csv'
TRAIN_DS = DATASET_DIR / 'train_interaction.csv'
TEST_DS = DATASET_DIR / 'test_interactions.csv'
