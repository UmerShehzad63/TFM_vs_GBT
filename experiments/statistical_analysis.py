"""
Statistical analysis of benchmark results.

Reads the original seed-level benchmark CSV files and produces:
    - paired model comparisons
    - mean differences
    - standard deviations
    - Wilcoxon signed-rank tests
    - effect sizes
    - confidence intervals
    - small/medium/large training-regime comparisons

No model training is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
TABLE_DIR = RESULTS_DIR / "tables" / "statistics"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "Adult": CSV_DIR / "adult.csv",
    "Bank Marketing": CSV_DIR / "bank-marketing.csv",
    "Credit-G": CSV_DIR / "credit-g.csv",
}


MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFN",
]


METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_results():

    frames = []

    for dataset_name, path in DATASETS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file:\n{path}"
            )

        df = pd.read_csv(path)

        df["Dataset"] = dataset_name

        frames.append(df)

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    return results


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_results(df):

    required_columns = {
        "Dataset",
        "Model",
        "Samples",
        "Seed",
        *METRICS,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[METRICS].isna().any().any():
        raise ValueError(
            "Metric columns contain missing values."
        )

    print("=" * 70)
    print("STATISTICAL ANALYSIS VALIDATION")
    print("=" * 70)

    print(f"Total observations: {len(df)}")
    print(f"Datasets: {df['Dataset'].nunique()}")
    print(f"Models: {df['Model'].nunique()}")
    print(f"Seeds: {df['Seed'].nunique()}")

    print("\nObservations by dataset:")
    print(df.groupby("Dataset").size())

    print("\nObservations by model:")
    print(df.groupby("Model").size())

    print()


# ============================================================
# PAIRED MODEL COMPARISON
# ============================================================

def paired_comparison(
    df,
    dataset,
    metric,
    model_a,
    model_b,
):
    """
    Compare model_a against model_b using matched
    dataset / sample-size / seed observations.
    """

    subset = df[
        (df["Dataset"] == dataset)
        & (df["Model"].isin([model_a, model_b]))
    ][
        [
            "Dataset",
            "Samples",
            "Seed",
            "Model",
            metric,
        ]
    ]

    pivot = subset.pivot_table(
        index=["Dataset", "Samples", "Seed"],
        columns="Model",
        values=metric,
        aggfunc="mean",
    )

    if model_a not in pivot.columns:
        return None

    if model_b not in pivot.columns:
        return None

    pivot = pivot.dropna(
        subset=[model_a, model_b]
    )

    if len(pivot) < 2:
        return None

    a = pivot[model_a].to_numpy()
    b = pivot[model_b].to_numpy()

    differences = a - b

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    mean_difference = float(np.mean(differences))
    std_difference = float(
        np.std(differences, ddof=1)
    )

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    nonzero = differences[differences != 0]

    if len(nonzero) >= 2:

        try:
            statistic, p_value = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
            )

            statistic = float(statistic)
            p_value = float(p_value)

        except ValueError:

            statistic = np.nan
            p_value = np.nan

    else:

        statistic = np.nan
        p_value = np.nan

    # --------------------------------------------------------
    # Rank-biserial effect size
    # --------------------------------------------------------

    if len(nonzero) >= 2:

        absolute = np.abs(nonzero)

        order = np.argsort(absolute)

        ranks = np.empty(len(absolute))
        ranks[order] = np.arange(
            1,
            len(absolute) + 1,
        )

        positive_rank_sum = np.sum(
            ranks[nonzero > 0]
        )

        negative_rank_sum = np.sum(
            ranks[nonzero < 0]
        )

        denominator = (
            positive_rank_sum
            + negative_rank_sum
        )

        if denominator > 0:

            rank_biserial = (
                positive_rank_sum
                - negative_rank_sum
            ) / denominator

        else:

            rank_biserial = np.nan

    else:

        rank_biserial = np.nan

    # --------------------------------------------------------
    # Approximate 95% CI for paired mean difference
    # --------------------------------------------------------

    n = len(differences)

    if n >= 2 and std_difference > 0:

        standard_error = (
            std_difference / np.sqrt(n)
        )

        margin = (
            1.96 * standard_error
        )

        ci_low = mean_difference - margin
        ci_high = mean_difference + margin

    else:

        ci_low = np.nan
        ci_high = np.nan

    return {
        "Dataset": dataset,
        "Metric": metric,
        "Model_A": model_a,
        "Model_B": model_b,
        "N": n,
        "Mean_A": mean_a,
        "Mean_B": mean_b,
        "Mean_Difference_A_minus_B": mean_difference,
        "Difference_SD": std_difference,
        "CI95_Low": ci_low,
        "CI95_High": ci_high,
        "Wilcoxon_Statistic": statistic,
        "Wilcoxon_p": p_value,
        "Rank_Biserial_Effect": rank_biserial,
    }


# ============================================================
# ALL TABPFN COMPARISONS
# ============================================================

def create_pairwise_table(df):

    rows = []

    tree_models = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    for dataset in DATASETS:

        for metric in METRICS:

            for tree_model in tree_models:

                result = paired_comparison(
                    df=df,
                    dataset=dataset,
                    metric=metric,
                    model_a="TabPFN",
                    model_b=tree_model,
                )

                if result is not None:
                    rows.append(result)

    return pd.DataFrame(rows)


def holm_adjust(p_values):

    """Return Holm-adjusted p-values for a family of paired tests."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = ~np.isnan(values)
    valid_values = values[valid]

    if len(valid_values) == 0:
        return adjusted

    order = np.argsort(valid_values)
    running_maximum = 0.0
    family_size = len(valid_values)

    for rank, position in enumerate(order):
        candidate = min(1.0, (family_size - rank) * valid_values[position])
        running_maximum = max(running_maximum, candidate)
        adjusted[np.flatnonzero(valid)[position]] = running_maximum

    return adjusted


# ============================================================
# TRAINING REGIMES
# ============================================================

def assign_training_regime(sample_size):

    if sample_size <= 100:
        return "Small"

    if sample_size <= 1000:
        return "Medium"

    return "Large"


def create_regime_comparison(df):

    work = df.copy()

    work["Training_Regime"] = (
        work["Samples"]
        .apply(assign_training_regime)
    )

    rows = []

    tree_models = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    for dataset in DATASETS:

        for regime in [
            "Small",
            "Medium",
            "Large",
        ]:

            for metric in METRICS:

                for tree_model in tree_models:

                    subset = work[
                        (work["Dataset"] == dataset)
                        & (
                            work["Training_Regime"]
                            == regime
                        )
                        & (
                            work["Model"]
                            .isin(
                                [
                                    "TabPFN",
                                    tree_model,
                                ]
                            )
                        )
                    ]

                    pivot = subset.pivot_table(
                        index=[
                            "Samples",
                            "Seed",
                        ],
                        columns="Model",
                        values=metric,
                        aggfunc="mean",
                    )

                    if (
                        "TabPFN" not in pivot.columns
                        or tree_model not in pivot.columns
                    ):
                        continue

                    pivot = pivot.dropna(
                        subset=[
                            "TabPFN",
                            tree_model,
                        ]
                    )

                    if len(pivot) == 0:
                        continue

                    difference = (
                        pivot["TabPFN"]
                        - pivot[tree_model]
                    )

                    rows.append({
                        "Dataset": dataset,
                        "Training_Regime": regime,
                        "Metric": metric,
                        "Comparison": (
                            f"TabPFN vs {tree_model}"
                        ),
                        "N": len(difference),
                        "TabPFN_Mean": (
                            pivot["TabPFN"].mean()
                        ),
                        "Tree_Mean": (
                            pivot[tree_model].mean()
                        ),
                        "Mean_Difference": (
                            difference.mean()
                        ),
                        "Difference_SD": (
                            difference.std(ddof=1)
                            if len(difference) > 1
                            else np.nan
                        ),
                    })

    return pd.DataFrame(rows)


# ============================================================
# OVERALL MODEL SUMMARY
# ============================================================

def create_model_summary(df):

    rows = []

    grouped = (
        df.groupby(
            [
                "Dataset",
                "Model",
            ]
        )
    )

    for (
        dataset,
        model,
    ), group in grouped:

        for metric in METRICS:

            values = group[metric]

            rows.append({
                "Dataset": dataset,
                "Model": model,
                "Metric": metric,
                "N": len(values),
                "Mean": values.mean(),
                "Std": values.std(ddof=1),
                "Min": values.min(),
                "Max": values.max(),
            })

    return pd.DataFrame(rows)


# ============================================================
# SAMPLE-SIZE COMPARISON
# ============================================================

def create_sample_size_summary(df):

    rows = []

    for dataset in DATASETS:

        for sample_size in sorted(
            df.loc[
                df["Dataset"] == dataset,
                "Samples",
            ].unique()
        ):

            for metric in METRICS:

                subset = df[
                    (df["Dataset"] == dataset)
                    & (
                        df["Samples"]
                        == sample_size
                    )
                ]

                for model in MODELS:

                    values = subset.loc[
                        subset["Model"] == model,
                        metric,
                    ]

                    if len(values) == 0:
                        continue

                    rows.append({
                        "Dataset": dataset,
                        "Samples": sample_size,
                        "Metric": metric,
                        "Model": model,
                        "Mean": values.mean(),
                        "Std": values.std(ddof=1),
                        "N": len(values),
                    })

    return pd.DataFrame(rows)


# ============================================================
# SIGNIFICANCE SUMMARY
# ============================================================

def create_significance_summary(pairwise):

    if pairwise.empty:
        return pairwise.copy()

    summary = pairwise.copy()

    summary["Significant_p05"] = (
        summary["Wilcoxon_p"] < 0.05
    )

    summary["Holm_Adjusted_p"] = holm_adjust(
        summary["Wilcoxon_p"]
    )

    summary["Significant_Holm_p05"] = (
        summary["Holm_Adjusted_p"] < 0.05
    )

    summary["Direction"] = np.where(
        summary[
            "Mean_Difference_A_minus_B"
        ] > 0,
        "TabPFN higher",
        np.where(
            summary[
                "Mean_Difference_A_minus_B"
            ] < 0,
            "TabPFN lower",
            "Equal",
        ),
    )

    summary["Effect_Strength"] = pd.cut(
        summary["Rank_Biserial_Effect"].abs(),
        bins=[
            -np.inf,
            0.1,
            0.3,
            0.5,
            np.inf,
        ],
        labels=[
            "Negligible",
            "Small",
            "Medium",
            "Large",
        ],
    )

    return summary


# ============================================================
# PRINT KEY FINDINGS
# ============================================================

def print_key_findings(pairwise):

    print()
    print("=" * 70)
    print("KEY STATISTICAL FINDINGS")
    print("=" * 70)

    if pairwise.empty:
        print("No pairwise comparisons available.")
        return

    significant = pairwise[
        pairwise["Wilcoxon_p"] < 0.05
    ]

    holm_adjusted = holm_adjust(pairwise["Wilcoxon_p"])
    holm_significant = np.sum(holm_adjusted < 0.05)

    print(
        f"\nSignificant comparisons (p < 0.05): "
        f"{len(significant)} / {len(pairwise)}"
    )

    print(
        f"Significant comparisons after Holm correction: "
        f"{holm_significant} / {len(pairwise)}"
    )

    print(
        "\nPositive difference means "
        "TabPFN performed better."
    )

    for dataset in DATASETS:

        print()
        print("-" * 70)
        print(dataset)

        subset = pairwise[
            pairwise["Dataset"] == dataset
        ]

        for metric in METRICS:

            metric_data = subset[
                subset["Metric"] == metric
            ]

            if metric_data.empty:
                continue

            print(f"\n{metric}:")

            for _, row in metric_data.iterrows():

                p = row["Wilcoxon_p"]

                if pd.isna(p):
                    p_text = "NA"
                else:
                    p_text = f"{p:.4g}"

                print(
                    f"  TabPFN vs "
                    f"{row['Model_B']}: "
                    f"difference="
                    f"{row['Mean_Difference_A_minus_B']:.4f}, "
                    f"p={p_text}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL STATISTICAL ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_results()

    validate_results(df)

    # --------------------------------------------------------
    # Overall model summary
    # --------------------------------------------------------

    model_summary = create_model_summary(df)

    output = (
        TABLE_DIR
        / "model_summary.csv"
    )

    model_summary.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Pairwise TabPFN comparisons
    # --------------------------------------------------------

    pairwise = create_pairwise_table(df)

    output = (
        TABLE_DIR
        / "tabpfn_pairwise_statistics.csv"
    )

    pairwise.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Significance summary
    # --------------------------------------------------------

    significance = (
        create_significance_summary(pairwise)
    )

    output = (
        TABLE_DIR
        / "significance_summary.csv"
    )

    significance.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Training-regime analysis
    # --------------------------------------------------------

    regime = create_regime_comparison(df)

    output = (
        TABLE_DIR
        / "training_regime_comparison.csv"
    )

    regime.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Sample-size summary
    # --------------------------------------------------------

    sample_summary = (
        create_sample_size_summary(df)
    )

    output = (
        TABLE_DIR
        / "sample_size_summary.csv"
    )

    sample_summary.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    print_key_findings(pairwise)

    print()
    print("=" * 70)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)
    print()
    print(
        f"Output directory:\n{TABLE_DIR}"
    )


if __name__ == "__main__":
    main()
