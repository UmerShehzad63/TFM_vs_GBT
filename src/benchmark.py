"""
Benchmark runner.
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.data.preprocessing import (
    build_tree_preprocessor,
    build_tabpfn_preprocessor,
)

from src.metrics import evaluate
from src.models.tree_models import get_tree_models
from src.models.tabpfn_model import get_tabpfn

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lightgbm",
)


def benchmark_models(
    X_train,
    y_train,
    X_test,
    y_test,
    sample_size,
    seed,
):

    results = []

    tree_models = get_tree_models(seed)

    tree_preprocessor = build_tree_preprocessor(X_train)

    for name, model in tree_models.items():

        values, counts = np.unique(y_train, return_counts=True)

        print(
            f"    {name} target distribution: "
            f"{dict(zip(values.tolist(), counts.tolist()))}"
        )

        pipeline = Pipeline([
            ("prep", tree_preprocessor),
            ("model", model),
        ])

        start = time.perf_counter()

        pipeline.fit(X_train, y_train)

        training_time = time.perf_counter() - start

        metrics = evaluate(
            pipeline,
            X_test,
            y_test,
        )

        metrics.update({
            "Model": name,
            "Samples": sample_size,
            "Seed": seed,
            "Training_Time": training_time,
        })

        results.append(metrics)

    if sample_size >= 10:

        tab_preprocessor = build_tabpfn_preprocessor(X_train)

        X_train_tab = tab_preprocessor.fit_transform(X_train)
        X_test_tab = tab_preprocessor.transform(X_test)

        model = get_tabpfn()

        start = time.perf_counter()

        model.fit(X_train_tab, y_train)

        training_time = time.perf_counter() - start

        metrics = evaluate(
            model,
            X_test_tab,
            y_test,
        )

        metrics.update({
            "Model": "TabPFN",
            "Samples": sample_size,
            "Seed": seed,
            "Training_Time": training_time,
        })

        results.append(metrics)

    return pd.DataFrame(results)