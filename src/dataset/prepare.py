import pandas as pd

from src.dataset.negatives import sample_negatives
from src.defaults import TEST_DS, TRAIN_DS


def prepare_interactions(prepare_test=True):
    train_interactions = pd.read_csv(TRAIN_DS)
    train_negatives = sample_negatives(train_interactions, len(train_interactions)*4, seed=1293)

    train_interactions['rating'] = 1
    train_negatives['rating'] = 0

    test = None
    train = pd.concat([train_interactions, train_negatives])

    if prepare_test:
        test_interactions = pd.read_csv(TEST_DS)
        test_negatives = sample_negatives(test_interactions, len(test_interactions)*4, seed=1293)
        test_interactions['rating'] = 1
        test_negatives['rating'] = 0
        test = pd.concat([test_interactions, test_negatives])

    return train, test
