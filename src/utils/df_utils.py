import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def series_of_list_to_one_hot(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Binarization of Series of lists."""

    binarizer = MultiLabelBinarizer(sparse_output=True)
    return pd.DataFrame.sparse.from_spmatrix(
        binarizer.fit_transform(data[col_name]),
        index=data.index,
        columns=binarizer.classes_)
