"""
Dataset loading utilities.
"""

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from src.config import DATASETS


def load_dataset(dataset_name: str):
    """
    Load a dataset and return:
        X -> pandas DataFrame
        y -> pandas Series (encoded as 0/1)
    """

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_id = DATASETS[dataset_name]

    X, y = fetch_openml(
        data_id=dataset_id,
        as_frame=True,
        return_X_y=True
    )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Preserve original indices
    y = pd.Series(
        y,
        index=X.index,
        name="target"
    )

    return X, y