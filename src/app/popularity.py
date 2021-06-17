import pandas as pd

from src.defaults import REPO_DS, TRAIN_DS


class Popularity:
    def __init__(self):
        repo_features = pd.read_csv(REPO_DS)[['full_name', 'n_stars']]
        repo_features = repo_features.drop_duplicates()
        train_interactions = pd.read_csv(TRAIN_DS)
        self.train_interactions = train_interactions.merge(repo_features, on='full_name')

    def get_repos_that_use(self, requirement: str, n: int = 5) -> list:
        r = self.train_interactions[self.train_interactions.repo_requirements == requirement]
        r = r.sort_values('n_stars', ascending=False)
        return list(r.full_name.iloc[:n])
