import pandas as pd

from src.dataset.negatives import sample_negatives
from src.defaults import TEST_DS, TRAIN_DS


def prepare_interactions():
    train_interactions = pd.read_csv(TRAIN_DS)
    test_interactions = pd.read_csv(TEST_DS)

    train_negatives = sample_negatives(train_interactions, len(train_interactions)*4, seed=1293)
    test_negatives = sample_negatives(test_interactions, len(test_interactions)*4, seed=1293)

    train_interactions['rating'] = 1
    train_negatives['rating'] = 0

    test_interactions['rating'] = 1
    test_negatives['rating'] = 0

    train = pd.concat([train_interactions, train_negatives])
    test = pd.concat([test_interactions, test_negatives])

    return train, test
