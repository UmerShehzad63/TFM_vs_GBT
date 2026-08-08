from pathlib import Path

import pandas as pd

from src.config import (
    RANDOM_SEEDS,
    TRAINING_SIZES,
    DATASETS,
)

from src.data.loaders import load_dataset

from src.data.sampling import (
    train_test_split_data,
    generate_training_subsets,
)

from src.benchmark import benchmark_models


RESULTS_DIR = Path("results/csv")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


for dataset_name in DATASETS:

    output_file = RESULTS_DIR / f"{dataset_name}.csv"

    if output_file.exists():

        print("=" * 60)
        print(f"Skipping {dataset_name} (already completed)")
        print("=" * 60)
        continue

    print()
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    X, y = load_dataset(dataset_name)

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
        output_file,
        index=False,
    )

    print()
    print(f"Saved {len(results)} rows to {output_file}")

print()
print("=" * 60)
print("ALL DATASETS COMPLETED")
print("=" * 60)