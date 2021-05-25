import os

import streamlit as st

from src.app.recommender import DummyRecommender
from src.data.crawling.github_crawler import GithubCrawler

login_or_token = os.environ.get('GITHUB_TOKEN')
password = os.environ.get('GITHUB_TOKEN')

crawler = GithubCrawler(login_or_token=login_or_token, password=password)
recommender = DummyRecommender(crawler)

st.title("Lib-suggest")
st.markdown("Put repository name in the field below to acquire library recommendations.")

repo_name = st.text_input(label="Repository Name")
if repo_name:
    try:
        recommendation = recommender.recommend(repo_name)
    except Exception as exc:
        st.error(exc)
    else:
        st.header(recommendation.repository['full_name'])

        with st.beta_expander("Requirements"):
            st.code('\n'.join(recommendation.repository['repo_requirements']))

        st.subheader("Recommended libraries")
        st.code('\n'.join(recommendation.recommendation))
