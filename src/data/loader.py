import os

import pandas as pd

from src.defaults import RAW_DATA


def load_all_datasets(path: str = RAW_DATA) -> pd.DataFrame:
    dataset_chunks = [file_entry
                      for file_entry in os.scandir(path) if file_entry.name.endswith('jsonl')]
    dataset_chunks = [pd.read_json(chunk.path, lines=True) for chunk in dataset_chunks]
    return pd.concat(dataset_chunks)
