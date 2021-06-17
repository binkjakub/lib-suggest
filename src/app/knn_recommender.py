from typing import Optional, Any

import pandas as pd
from numpy.random import PCG64
from surprise import Dataset, Reader, KNNBasic

from src.app.recommender import Recommender, T
from src.app.utils import get_top_n
from src.data.crawling.github_crawler import GithubCrawler
from src.defaults import KNN_TRAIN


class KNNRecommender(Recommender):
    N_RECOMMENDATIONS = 5

    def __init__(self,
                 n_recommendations: int = N_RECOMMENDATIONS,
                 crawler: Optional[GithubCrawler] = None):
        super().__init__(crawler)
        self.train = pd.read_csv(KNN_TRAIN)
        self.n_recommendations = n_recommendations

    def sample_neg_packages(self, x: pd.DataFrame, n: int):
        generator = PCG64(12331)
        negatives = []
        while len(negatives) < n:
            [random_package] = self.train.sample(random_state=generator)['repo_requirements']
            if random_package not in x.repo_requirements.values:
                negatives.append({'full_name': x.full_name.values[0],
                                  'repo_requirements': random_package})

        negatives = pd.DataFrame(negatives)
        negatives['rating'] = 0
        return negatives

    def recommend(self, repository: dict[str, Any]) -> T:
        x = pd.DataFrame(repository).explode('repo_requirements')[
            ['full_name', 'repo_requirements']]
        x['rating'] = 1
        x_neg = self.sample_neg_packages(x, len(x) * 4)
        train = pd.concat([x, x_neg, self.train])
        trainset = Dataset.load_from_df(train, Reader(rating_scale=(0, 1))).build_full_trainset()

        algo = KNNBasic(sim_options={'name': 'pearson'})
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        top = get_top_n(predictions, self.n_recommendations)[repository['full_name']]
        return [package for package, _ in top]
