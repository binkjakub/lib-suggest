import os
from pathlib import Path

DVC_PATH = Path(os.path.dirname(__file__)).parent.joinpath('storage').absolute()
DATASET_PATH = DVC_PATH / 'dataset'