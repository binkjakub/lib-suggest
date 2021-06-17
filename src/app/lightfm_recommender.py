from typing import Any, Optional

import pandas as pd

from src.app.recommender import Recommender
from src.data.crawling.github_crawler import GithubCrawler
from src.lightfm.recommender import LightFMModel


class LightFMRecommender(Recommender):
    """Recommendation based on LightFM model.

    https://making.lyst.com/lightfm/docs/home.html
    """

    N_RECOMMENDATIONS = 5

    def __init__(self,
                 n_recommendations: int = N_RECOMMENDATIONS,
                 crawler: Optional[GithubCrawler] = None,
                 user_features_type: Optional[str] = None,
                 num_epochs: int = 15,
                 **kwargs
                 ):
        super().__init__(crawler)
        self.model = LightFMModel(crawler=crawler)
        self.model.prepare()
        self.model.fit(train=None, user_features_type=user_features_type,
                       num_epochs=num_epochs, **kwargs)
        self._n_recommendation = n_recommendations

    def recommend(self, repository: dict[str, Any]) -> list[str]:
        repo_name = repository['full_name']
        this_repo_features = pd.DataFrame(repository).drop(
            columns=['repo_requirements']).drop_duplicates()
        repository = pd.DataFrame(repository)
        this_test = repository[['full_name', 'repo_requirements']]
        self.model.refit_model(this_test, this_repo_features)
        return self.model.predict_repo(repo_name, n=self._n_recommendation)
