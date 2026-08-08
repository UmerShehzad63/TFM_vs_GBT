"""
Statistical analysis of final benchmark results.

This script reads the raw benchmark CSV files and produces
summary tables for the research analysis.

Raw result files are never modified.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "csv"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "Adult": RESULTS_DIR / "adult.csv",
    "Bank Marketing": RESULTS_DIR / "bank-marketing.csv",
    "Credit-G": RESULTS_DIR / "credit-g.csv",
}


MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFN",
]


PERFORMANCE_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]


TIME_METRICS = [
    "Training_Time",
    "Prediction_Time",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_results():

    frames = []

    for dataset_name, path in DATASETS.items():

        df = pd.read_csv(path)

        df["Dataset"] = dataset_name

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

def create_overall_performance(df):

    rows = []

    for (dataset, model), group in df.groupby(
        ["Dataset", "Model"]
    ):

        result = {
            "Dataset": dataset,
            "Model": model,
        }

        for metric in PERFORMANCE_METRICS:

            result[f"{metric}_Mean"] = group[
                metric
            ].mean()

            result[f"{metric}_Std"] = group[
                metric
            ].std()

        rows.append(result)

    return pd.DataFrame(rows)

    grouped = (
        df.groupby(
            ["Dataset", "Model"]
        )[PERFORMANCE_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )

    rows = []

    for _, row in grouped.iterrows():

        result = {
            "Dataset": row["Dataset"],
            "Model": row["Model"],
        }

        for metric in PERFORMANCE_METRICS:

            result[f"{metric}_Mean"] = row[
                (metric, "mean")
            ]

            result[f"{metric}_Std"] = row[
                (metric, "std")
            ]

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# OVERALL COMPUTATIONAL COST
# ============================================================

def create_time_summary(df):

    grouped = (
        df.groupby(
            ["Dataset", "Model"]
        )[TIME_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )

    rows = []

    for _, row in grouped.iterrows():

        result = {
            "Dataset": row["Dataset"],
            "Model": row["Model"],
        }

        for metric in TIME_METRICS:

            result[f"{metric}_Mean"] = row[
                (metric, "mean")
            ]

            result[f"{metric}_Std"] = row[
                (metric, "std")
            ]

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# PERFORMANCE BY TRAINING SIZE
# ============================================================

def create_learning_curve_summary(df):

    grouped = (
        df.groupby(
            ["Dataset", "Samples", "Model"]
        )[PERFORMANCE_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )

    rows = []

    for _, row in grouped.iterrows():

        result = {
            "Dataset": row["Dataset"],
            "Samples": row["Samples"],
            "Model": row["Model"],
        }

        for metric in PERFORMANCE_METRICS:

            result[f"{metric}_Mean"] = row[
                (metric, "mean")
            ]

            result[f"{metric}_Std"] = row[
                (metric, "std")
            ]

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# BEST MODEL
# ============================================================

def create_best_model_table(overall):

    rows = []

    for dataset in overall["Dataset"].unique():

        subset = overall[
            overall["Dataset"] == dataset
        ]

        for metric in PERFORMANCE_METRICS:

            mean_column = f"{metric}_Mean"

            best_index = subset[
                mean_column
            ].idxmax()

            best = subset.loc[best_index]

            rows.append({
                "Dataset": dataset,
                "Metric": metric,
                "Best_Model": best["Model"],
                "Mean": best[mean_column],
                "Std": best[f"{metric}_Std"],
            })

    return pd.DataFrame(rows)


# ============================================================
# TABPFN COMPARISON
# ============================================================

def create_tabpfn_comparison(overall):

    rows = []

    for dataset in overall["Dataset"].unique():

        subset = overall[
            overall["Dataset"] == dataset
        ]

        tabpfn = subset[
            subset["Model"] == "TabPFN"
        ].iloc[0]

        for baseline in [
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ]:

            baseline_row = subset[
                subset["Model"] == baseline
            ].iloc[0]

            for metric in PERFORMANCE_METRICS:

                metric_column = f"{metric}_Mean"

                tabpfn_value = tabpfn[
                    metric_column
                ]

                baseline_value = baseline_row[
                    metric_column
                ]

                rows.append({
                    "Dataset": dataset,
                    "Baseline": baseline,
                    "Metric": metric,
                    "TabPFN_Mean": tabpfn_value,
                    "Baseline_Mean": baseline_value,
                    "Difference_TabPFN_minus_Baseline":
                        tabpfn_value - baseline_value,
                })

    return pd.DataFrame(rows)


# ============================================================
# TABPFN BY TRAINING SIZE
# ============================================================

def create_tabpfn_learning_comparison(df):

    grouped = (
        df.groupby(
            ["Dataset", "Samples", "Model"]
        )[PERFORMANCE_METRICS]
        .mean()
        .reset_index()
    )

    rows = []

    for dataset in grouped["Dataset"].unique():

        dataset_df = grouped[
            grouped["Dataset"] == dataset
        ]

        for sample_size in sorted(
            dataset_df["Samples"].unique()
        ):

            size_df = dataset_df[
                dataset_df["Samples"] == sample_size
            ]

            tabpfn = size_df[
                size_df["Model"] == "TabPFN"
            ]

            if tabpfn.empty:
                continue

            tabpfn = tabpfn.iloc[0]

            for baseline in [
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]:

                baseline_row = size_df[
                    size_df["Model"] == baseline
                ]

                if baseline_row.empty:
                    continue

                baseline_row = baseline_row.iloc[0]

                for metric in PERFORMANCE_METRICS:

                    rows.append({
                        "Dataset": dataset,
                        "Samples": sample_size,
                        "Baseline": baseline,
                        "Metric": metric,
                        "TabPFN_Mean":
                            tabpfn[metric],
                        "Baseline_Mean":
                            baseline_row[metric],
                        "Difference":
                            tabpfn[metric]
                            - baseline_row[metric],
                    })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL RESEARCH ANALYSIS")
    print("=" * 70)

    df = load_results()

    print()
    print(f"Combined observations: {len(df)}")
    print(
        f"Datasets: {df['Dataset'].nunique()}"
    )
    print(
        f"Models: {df['Model'].nunique()}"
    )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    overall = create_overall_performance(df)

    overall_path = (
        TABLES_DIR / "overall_performance.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print()
    print(
        f"Saved: {overall_path}"
    )

    # --------------------------------------------------------
    # Computational performance
    # --------------------------------------------------------

    times = create_time_summary(df)

    times_path = (
        TABLES_DIR / "computational_performance.csv"
    )

    times.to_csv(
        times_path,
        index=False,
    )

    print(
        f"Saved: {times_path}"
    )

    # --------------------------------------------------------
    # Learning curves
    # --------------------------------------------------------

    learning = create_learning_curve_summary(df)

    learning_path = (
        TABLES_DIR / "learning_curve_summary.csv"
    )

    learning.to_csv(
        learning_path,
        index=False,
    )

    print(
        f"Saved: {learning_path}"
    )

    # --------------------------------------------------------
    # Best models
    # --------------------------------------------------------

    best = create_best_model_table(overall)

    best_path = (
        TABLES_DIR / "best_models.csv"
    )

    best.to_csv(
        best_path,
        index=False,
    )

    print(
        f"Saved: {best_path}"
    )

    # --------------------------------------------------------
    # TabPFN comparison
    # --------------------------------------------------------

    comparison = create_tabpfn_comparison(
        overall
    )

    comparison_path = (
        TABLES_DIR / "tabpfn_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        f"Saved: {comparison_path}"
    )

    # --------------------------------------------------------
    # TabPFN learning comparison
    # --------------------------------------------------------

    tabpfn_learning = (
        create_tabpfn_learning_comparison(df)
    )

    tabpfn_learning_path = (
        TABLES_DIR
        / "tabpfn_learning_comparison.csv"
    )

    tabpfn_learning.to_csv(
        tabpfn_learning_path,
        index=False,
    )

    print(
        f"Saved: {tabpfn_learning_path}"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST MODEL BY DATASET AND METRIC")
    print("=" * 70)

    print(best.to_string(index=False))

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()