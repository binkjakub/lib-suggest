import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas import CategoricalDtype

from src.defaults import RAW_DATA, TEST_DS, TRAIN_DS

PathLike = Union[str, Path]


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


def load_train_test_interactions(train_path: PathLike = TRAIN_DS,
                                 test_path: PathLike = TEST_DS
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_interactions, test_interactions = pd.read_csv(train_path), pd.read_csv(test_path)

    train_interactions['rating'] = 1
    test_interactions['rating'] = 1

    lib_dtype = CategoricalDtype(categories=test_interactions['full_name'].unique())
    repo_dtype = CategoricalDtype(categories=train_interactions['repo_requirements'].unique())

    def names_to_codes(df: pd.DataFrame):
        df['full_name'] = df['full_name'].astype(lib_dtype).cat.codes
        df['repo_requirements'] = df['repo_requirements'].astype(repo_dtype).cat.codes
        return df

    train_interactions = names_to_codes(train_interactions)
    test_interactions = names_to_codes(test_interactions)

    assert train_interactions.columns.tolist() == ['full_name', 'repo_requirements', 'rating']
    assert test_interactions.columns.tolist() == ['full_name', 'repo_requirements', 'rating']

    return train_interactions, test_interactions
