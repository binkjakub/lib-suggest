from github import Github

from src.feature_extraction.repo import extract_batch

g = Github()
repos = [g.get_repo("pytorch/pytorch"), g.get_repo("flairNLP/flair"),
         g.get_repo("davidfischer/requirements-parser")]
extract_batch(repos)
# pprint(extract_repo(repo))
