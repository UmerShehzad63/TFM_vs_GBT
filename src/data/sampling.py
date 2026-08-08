"""
Sampling utilities.
"""

import pandas as pd

from sklearn.model_selection import train_test_split


def train_test_split_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def generate_training_subsets(
    X_train,
    y_train,
    training_sizes,
    random_state=42,
):

    subsets = {}

    class_counts = y_train.value_counts(normalize=True)

    for size in training_sizes:

        if size == "ALL":

            subsets["ALL"] = (
                X_train.copy(),
                y_train.copy(),
            )

            continue

        if size > len(X_train):
            continue

        selected_indices = []

        for label, proportion in class_counts.items():

            n = max(1, round(size * proportion))

            idx = (
                y_train[y_train == label]
                .sample(
                    n=min(n, (y_train == label).sum()),
                    random_state=random_state,
                    replace=False,
                )
                .index
            )

            selected_indices.extend(idx)

        # If rounding gives too many samples
        selected_indices = selected_indices[:size]

        # If rounding gives too few samples
        if len(selected_indices) < size:

            remaining = X_train.index.difference(selected_indices)

            extra = (
                pd.Series(remaining)
                .sample(
                    size - len(selected_indices),
                    random_state=random_state,
                )
                .tolist()
            )

            selected_indices.extend(extra)

        X_subset = X_train.loc[selected_indices]
        y_subset = y_train.loc[selected_indices]

        subsets[size] = (
            X_subset,
            y_subset,
        )

    return subsets