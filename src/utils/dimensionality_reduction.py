from typing import Collection, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


def projection_plot(
        vectors: np.ndarray,
        labels: Optional[Collection] = None,
        labels_name: Optional[str] = None,
        title: Optional[str] = None,
        method: str = 't-sne',
        ret_projections: bool = False
) -> Union[plt.Axes, tuple[plt.Axes, pd.DataFrame]]:
    """Returns axis containing datapoints projected by TSNE."""
    projected_vectors = get_projection(vectors, labels, labels_name, method)

    ax = sns.scatterplot(data=projected_vectors, x='x', y='y', hue=labels_name, palette="deep")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=labels_name)

    title = title or f"Projected embeddings (colored by {labels_name})"
    ax.set_title(title)

    if ret_projections:
        return ax, projected_vectors
    else:
        return ax


def get_projection(vectors: np.ndarray,
                   labels: Optional[Collection] = None,
                   labels_name: Optional[str] = None,
                   method: str = 't-sne',
                   metric: str = 'cosine',
                   **kwargs) -> pd.DataFrame:
    """Returns dataframe with TSNE points."""
    if method == 't-sne':
        projected_vectors = TSNE(n_components=2,
                                 metric=metric,
                                 square_distances=True, **kwargs).fit_transform(vectors)
    elif method == 'umap':
        projected_vectors = UMAP(n_components=2, metric=metric, **kwargs).fit_transform(vectors)
    else:
        raise ValueError(f"Invalid projection method name: {method}")
    projected_vectors = pd.DataFrame(projected_vectors, columns=['x', 'y'])

    labels_name = labels_name or 'Labels'
    projected_vectors[labels_name] = labels
    projected_vectors[labels_name] = projected_vectors[labels_name].astype('category')

    return projected_vectors


def pairwise_similarities_plot(vectors):
    """Returns plot with pairwise similarity distribution (takes lower triangular of sim matrix)."""
    similarities = cosine_similarity(vectors)
    tril_mask = np.mask_indices(similarities.shape[0], np.tril, k=-1)
    tril_similarities = similarities[tril_mask]

    ax = sns.histplot(x=tril_similarities, kde=True)
    ax.set_title("Pairwise similarities distribution")
    return ax
