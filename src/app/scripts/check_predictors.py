from src.app.ncf_recommender import NCFRecommender
from src.data.crawling.github_crawler import MockedCrawler

checkpoint_path = ('/home/jakub/semester_3_ds/recommender_systems/logs/lib_suggest_ncf/3rqk3vgq/'
                   'checkpoints/MLP-epoch=01-train_ndcg=1.00-val_ndcg=1.00.ckpt')
recommender = NCFRecommender(MockedCrawler(), checkpoint_path, n_recommendations=5)
recommendations = recommender.crawl_and_recommend('binkjakub/lib-suggest')
print(recommendations)
