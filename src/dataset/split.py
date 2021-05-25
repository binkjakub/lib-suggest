from typing import Optional

import numpy as np
import pandas as pd


def split_leave_one_out(interactions: pd.DataFrame,
                        repo_col: str,
                        num_test_examples: int = 1,
                        random_state: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits ratings into train/test set delegating last interaction to the test set."""
    interactions_grouped = interactions.groupby([repo_col])
    assert (interactions_grouped.size() >= (num_test_examples + 1)).all()

    test_indices = interactions_grouped[repo_col].sample(n=num_test_examples,
                                                         random_state=random_state).index
    train_indices = interactions.index.difference(test_indices)

    train = interactions.loc[train_indices]
    test = interactions.loc[test_indices]

    # check repositories in 'dev' and 'test' set overlaps with training set
    assert np.in1d(train[repo_col].unique(), test[repo_col].unique()).all()

    return train, test
