"""
Benchmark runner.
"""

import time
import pandas as pd

from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_tree_preprocessor
from src.metrics import evaluate
from src.models.tree_models import get_tree_models


def benchmark_tree_models(

    X_train,

    y_train,

    X_test,

    y_test,

    sample_size,

    seed,

):

    results = []

    models = get_tree_models(seed)

    preprocessor = build_tree_preprocessor(X_train)

    for name, model in models.items():

        pipeline = Pipeline([

            ("prep", preprocessor),

            ("model", model)

        ])

        start = time.perf_counter()

        pipeline.fit(X_train, y_train)

        training_time = time.perf_counter() - start

        metrics = evaluate(

            pipeline,

            X_test,

            y_test

        )

        metrics.update({

            "Model": name,

            "Samples": sample_size,

            "Seed": seed,

            "Training_Time": training_time

        })

        results.append(metrics)

    return pd.DataFrame(results)