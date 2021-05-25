import os
from typing import Optional

import pandas as pd

from src.defaults import RAW_DATA


def load_all_datasets(path: str = RAW_DATA,
                      drop_duplicates: bool = False,
                      drop_all_nan_cols: bool = False,
                      min_requirements_num: Optional[int] = None,
                      min_library_occurrences: Optional[int] = None) -> pd.DataFrame:
    """Loads all jsonl files from given directory to a single pandas.DataFrame."""
    dataset_chunks = [file_entry
                      for file_entry in os.scandir(path) if file_entry.name.endswith('jsonl')]
    dataset_chunks = [pd.read_json(chunk.path, lines=True) for chunk in dataset_chunks]
    dataset = pd.concat(dataset_chunks)

    if drop_duplicates:
        dataset = dataset.drop_duplicates('full_name')
    if drop_all_nan_cols:
        dataset = dataset.dropna(axis=1, how='all')
    if min_library_occurrences is not None:
        lib_counts = dataset['repo_requirements'].explode().value_counts()
        filtered_libs = set(lib_counts[lib_counts > min_library_occurrences].index.tolist())
        dataset['repo_requirements'] = dataset['repo_requirements'].apply(
            lambda libs: [lib for lib in libs if lib in filtered_libs]
        )
    if min_requirements_num is not None:
        dataset = dataset[dataset['repo_requirements'].apply(len) > min_requirements_num]

    dataset = dataset.reset_index(drop=True)

    return dataset
