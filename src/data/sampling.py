"""
Sampling utilities.
"""

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

    for size in training_sizes:

        if size == "ALL":

            subsets["ALL"] = (
                X_train.copy(),
                y_train.copy(),
            )

            continue

        if size > len(X_train):
            continue

        X_subset = X_train.sample(
            n=size,
            random_state=random_state,
            replace=False,
        )

        y_subset = y_train.loc[X_subset.index]

        subsets[size] = (
            X_subset,
            y_subset,
        )

    return subsets