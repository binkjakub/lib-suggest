import pandas as pd
from numpy.random import PCG64
from numpy.random.bit_generator import BitGenerator


def sample_negative(train_interactions: pd.DataFrame, generator: BitGenerator) -> tuple[str, str]:
    [random_repo] = train_interactions.sample(random_state=generator)['full_name']
    [random_package] = train_interactions.sample(random_state=generator)['repo_requirements']
    return random_repo, random_package


def sample_negatives(train_interactions: pd.DataFrame, n: int, seed=17) -> pd.DataFrame:
    generator = PCG64(seed)

    negatives = []
    while len(negatives) < n:
        random_repo, random_package = sample_negative(train_interactions, generator)
        if ((train_interactions.full_name == random_repo) & (
                train_interactions.repo_requirements == random_package)).sum() == 0:
            negatives.append({'full_name': random_repo, 'repo_requirements': random_package})

    negatives = pd.DataFrame(negatives)
    negatives['rating'] = 0
    return negatives
