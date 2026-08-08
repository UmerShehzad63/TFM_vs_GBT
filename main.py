from src.config import RANDOM_SEEDS, TRAINING_SIZES
from src.data.loaders import load_dataset
from src.data.sampling import (
    train_test_split_data,
    generate_training_subsets,
)
from src.benchmark import benchmark_models

import pandas as pd

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

X, y = load_dataset("adult")

X_train, X_test, y_train, y_test = train_test_split_data(
    X,
    y,
)

all_results = []

for seed in RANDOM_SEEDS:

    print(f"\nSeed: {seed}")

    subsets = generate_training_subsets(
        X_train,
        y_train,
        TRAINING_SIZES,
        random_state=seed,
    )

    for size, (Xs, ys) in subsets.items():

        print(f"  Samples: {size}")

        df = benchmark_models(
            Xs,
            ys,
            X_test,
            y_test,
            size,
            seed,
        )

        all_results.append(df)

results = pd.concat(
    all_results,
    ignore_index=True,
)

results.to_csv(
    "results/csv/tree_results.csv",
    index=False,
)

print(f"\nSaved {len(results)} rows to results/csv/tree_results.csv")
print("\nBenchmark completed.")