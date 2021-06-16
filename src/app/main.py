import os

import streamlit as st

from src.app.ncf_recommender import NCFEmbeddingRecommender
from src.app.recommender import DummyRecommender, RecommenderCollection
from src.data.crawling.github_crawler import GithubCrawler

login_or_token = os.environ.get('GITHUB_TOKEN')
password = os.environ.get('GITHUB_TOKEN')
checkpoint_path = os.environ.get('NCF_CHECKPOINT_PATH')
n_recommendations = os.environ.get('N_RECOMMENDATIONS', 5)

crawler = GithubCrawler(login_or_token=login_or_token, password=password)
recommender = RecommenderCollection(crawler,
                                    [NCFEmbeddingRecommender(checkpoint_path, n_recommendations),
                                     DummyRecommender()])

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

        st.subheader("Recommended libraries")

        for r_name, r_result in recommendation.recommendation.items():
            st.markdown(f"`{r_name}`")
            st.code('\n'.join(r_result))
