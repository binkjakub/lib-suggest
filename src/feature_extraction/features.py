from typing import Iterable

import numpy as np
import pandas as pd
import torch

FEATURE_NAMES = ('is_master_protected', 'n_all_issues', 'n_branches', 'n_closed_issues',
                 'n_forks', 'n_milestones_all', 'n_milestones_closed', 'n_milestones_open',
                 'n_open_issues', 'n_pr_all', 'n_pr_closed', 'n_pr_open', 'n_stars')


def get_features_tensor(repo_names: np.ndarray,
                        repo_feats: pd.DataFrame,
                        repos: torch.Tensor,
                        feature_names: Iterable[str] = FEATURE_NAMES) -> torch.Tensor:
    repo_feats = repo_feats.set_index('full_name')
    repo_names = repo_names[repos.long()]
    repo_feats = repo_feats.loc[repo_names, feature_names].values.astype(float)
    repo_feats = torch.tensor(repo_feats, dtype=torch.float)
    assert len(repo_feats) == len(repos)
    return repo_feats
