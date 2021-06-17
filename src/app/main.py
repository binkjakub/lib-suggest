import os

import pandas as pd
import streamlit as st

from src.app.knn_recommender import KNNRecommender
from src.app.lightfm_recommender import LightFMRecommender

from src.app.ncf_recommender import NCFRecommender
from src.app.popularity import Popularity
from src.app.recommender import DummyRecommender, RecommenderCollection
from src.data.crawling.github_crawler import GithubCrawler

lightfm_params = {'user_features_type': None, 'num_epochs': 15, 'no_components': 50,
                  'loss': 'warp', 'learning_schedule': 'adagrad'}

login_or_token = os.environ.get('GITHUB_TOKEN')
password = os.environ.get('GITHUB_TOKEN')
checkpoint_path = os.environ.get('NCF_CHECKPOINT_PATH')
n_recommendations = os.environ.get('N_RECOMMENDATIONS', 5)

crawler = GithubCrawler(login_or_token=login_or_token, password=password)
recommender = RecommenderCollection(crawler,
                                    [KNNRecommender(),
                                     LightFMRecommender(n_recommendations=n_recommendations,
                                                        **lightfm_params),
                                     NCFRecommender(checkpoint_path, n_recommendations),
                                     DummyRecommender()])

popularity = Popularity()

st.title("Lib-suggest")
st.markdown("Put repository name in the field below to acquire library recommendations.")

repo_name = st.text_input(label="Repository Name")
if repo_name:
    try:
        recommendation = recommender.crawl_and_recommend(repo_name)
    except Exception as exc:
        st.error(exc)
    else:
        st.header(recommendation.repository['full_name'])

        with st.beta_expander("Requirements"):
            st.code('\n'.join(recommendation.repository['repo_requirements']))

        st.subheader("Recommended libraries (and other repos that use them)")

        for r_name, r_result in recommendation.recommendation.items():
            st.markdown(f"`{r_name}`")

            package_rows = []
            for req in r_result:
                other_users = popularity.get_repos_that_use(req, 3)
                other_users = ", ".join(other_users)
                package_row = {'Recommendation': req, 'Most popular usages': other_users}
                package_rows.append(package_row)

            st.table(pd.DataFrame(package_rows))
