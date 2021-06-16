from src.app.ncf_recommender import NCFEmbeddingRecommender

checkpoint_path = ''

recommender = NCFEmbeddingRecommender(checkpoint_path, n_recommendations=5)
recommendations = recommender.recommend({'repo_requirements': ['scipy']})
print(recommendations)
