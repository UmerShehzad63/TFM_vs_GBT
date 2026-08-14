"""
Create additional statistical and research figures from existing results.

This script does NOT retrain any model.
It reads the already-generated benchmark CSV files and statistical tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

RAW_RESULTS_DIR = ROOT / "results" / "csv"
TABLES_DIR = ROOT / "results" / "tables"
PAPER_TABLES_DIR = TABLES_DIR / "paper"

OUTPUT_DIR = ROOT / "results" / "figures" / "statistical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = [
    ("Adult", "adult.csv"),
    ("Bank Marketing", "bank-marketing.csv"),
    ("Credit-G", "credit-g.csv"),
]

MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFN",
]


# ============================================================
# Load data
# ============================================================

def load_raw_results() -> pd.DataFrame:
    """Load and combine all raw benchmark result CSV files."""

    frames = []

    for dataset_name, filename in DATASETS:
        path = RAW_RESULTS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")

        df = pd.read_csv(path)
        df["Dataset"] = dataset_name

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    required = {
        "Dataset",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Prediction_Time",
        "Training_Time",
        "Model",
        "Samples",
        "Seed",
    }

    missing = required - set(combined.columns)

    if missing:
        raise ValueError(
            f"Raw results are missing columns: {sorted(missing)}"
        )

    return combined


def load_statistical_results() -> pd.DataFrame:
    """Load paired TabPFN-vs-tree statistical results."""

    path = PAPER_TABLES_DIR / "table_tabpfn_vs_trees.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing statistical table: {path}")

    df = pd.read_csv(path)

    required = {
        "Dataset",
        "Metric",
        "Tree_Model",
        "N",
        "TabPFN_Mean",
        "Tree_Mean",
        "TabPFN_Minus_Tree",
        "CI95_Low",
        "CI95_High",
        "Wilcoxon_p",
        "Rank_Biserial_Effect",
        "Significant_p05",
        "Holm_Adjusted_p",
        "Significant_Holm_p05",
        "Direction",
        "Effect_Strength",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Statistical table is missing columns: {sorted(missing)}"
        )

    return df


def load_model_summary() -> pd.DataFrame:
    """Load model-level summary table."""

    path = PAPER_TABLES_DIR / "table_model_summary.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing model summary table: {path}")

    df = pd.read_csv(path)

    required = {
        "Dataset",
        "Model",
        "Metric",
        "N",
        "Mean",
        "Std",
        "Min",
        "Max",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Model summary is missing columns: {sorted(missing)}"
        )

    return df


def load_computational_summary() -> pd.DataFrame:
    """Load computational summary."""

    path = TABLES_DIR / "computational_performance.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing computational table: {path}")

    df = pd.read_csv(path)

    required = {
        "Dataset",
        "Model",
        "Training_Time_Mean",
        "Training_Time_Std",
        "Prediction_Time_Mean",
        "Prediction_Time_Std",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Computational table is missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# Figure 1 + 2:
# Distribution plots
# ============================================================

def create_distribution_plot(
    raw: pd.DataFrame,
    metric: str,
    filename: str,
    title: str,
) -> None:
    """Create violin + box plot for a metric across datasets/models."""

    plot_df = raw[
        ["Dataset", "Model", metric]
    ].dropna().copy()

    plot_df["Model"] = pd.Categorical(
        plot_df["Model"],
        categories=MODELS,
        ordered=True,
    )

    plt.figure(figsize=(13, 8))

    sns.violinplot(
        data=plot_df,
        x="Dataset",
        y=metric,
        hue="Model",
        order=[d[0] for d in DATASETS],
        hue_order=MODELS,
        inner=None,
        cut=0,
        density_norm="width",
    )

    sns.boxplot(
        data=plot_df,
        x="Dataset",
        y=metric,
        hue="Model",
        order=[d[0] for d in DATASETS],
        hue_order=MODELS,
        width=0.18,
        fliersize=2,
        linewidth=1,
        dodge=True,
    )

    plt.xlabel("Dataset")
    plt.ylabel(metric)
    plt.title(title)

    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    output = OUTPUT_DIR / filename
    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Figure 3:
# Forest plot
# ============================================================

def create_forest_plot(stats: pd.DataFrame) -> None:
    """Create forest plot using paired differences and 95% CIs."""

    plot_df = stats.copy()

    metric_order = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    baseline_order = [
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]

    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"],
        categories=metric_order,
        ordered=True,
    )

    plot_df["Tree_Model"] = pd.Categorical(
        plot_df["Tree_Model"],
        categories=baseline_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(
        ["Dataset", "Metric", "Tree_Model"]
    ).reset_index(drop=True)

    labels = [
        f"{row.Dataset} | {row.Metric} | {row.Tree_Model}"
        for row in plot_df.itertuples()
    ]

    y = np.arange(len(plot_df))

    plt.figure(figsize=(12, max(10, len(plot_df) * 0.24)))

    x = plot_df["TabPFN_Minus_Tree"].to_numpy()
    low = plot_df["CI95_Low"].to_numpy()
    high = plot_df["CI95_High"].to_numpy()

    lower_error = x - low
    upper_error = high - x

    plt.errorbar(
        x,
        y,
        xerr=[lower_error, upper_error],
        fmt="o",
        capsize=3,
        markersize=4,
        linewidth=1,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    # Mark Holm-significant comparisons.
    significant = (
        plot_df["Significant_Holm_p05"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    for index, is_significant in enumerate(significant):
        if is_significant:
            plt.text(
                x[index],
                y[index],
                " *",
                va="center",
                fontsize=8,
            )

    plt.yticks(
        y,
        labels,
        fontsize=8,
    )

    plt.xlabel(
        "Paired difference (TabPFN − baseline)"
    )

    plt.ylabel(
        "Dataset | Metric | Baseline"
    )

    plt.title(
        "TabPFN vs Gradient-Boosted Trees: Paired Differences with 95% CIs"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "tabpfn_vs_tree_forest_plot.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Figure 4:
# Effect-size heatmap
# ============================================================

def create_effect_size_heatmap(stats: pd.DataFrame) -> None:
    """Create rank-biserial effect-size heatmap."""

    heatmap_df = stats.copy()

    heatmap_df["Row"] = (
        heatmap_df["Dataset"]
        + " | "
        + heatmap_df["Metric"]
    )

    pivot = heatmap_df.pivot_table(
        index="Row",
        columns="Tree_Model",
        values="Rank_Biserial_Effect",
        aggfunc="first",
    )

    # Preserve logical ordering.
    row_order = []

    for dataset, _ in DATASETS:
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ]:
            row_order.append(
                f"{dataset} | {metric}"
            )

    row_order = [
        row for row in row_order
        if row in pivot.index
    ]

    pivot = pivot.reindex(row_order)

    plt.figure(
        figsize=(10, max(10, len(pivot) * 0.25))
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={
            "label": "Rank-biserial effect size"
        },
    )

    # Add Holm-significance markers.
    significance_lookup = {}

    for row in stats.itertuples():
        key = (
            f"{row.Dataset} | {row.Metric}",
            row.Tree_Model,
        )

        significance_lookup[key] = bool(
            str(row.Significant_Holm_p05).lower()
            in {"true", "1", "yes"}
        )

    for row_index, row_name in enumerate(pivot.index):
        for col_index, model in enumerate(pivot.columns):
            key = (row_name, model)

            if significance_lookup.get(key, False):
                value = pivot.iloc[
                    row_index,
                    col_index,
                ]

                if not pd.isna(value):
                    plt.text(
                        col_index + 0.86,
                        row_index + 0.18,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    plt.xlabel("Baseline Model")
    plt.ylabel("Dataset | Metric")

    plt.title(
        "TabPFN Rank-Biserial Effect Sizes\n"
        "* Holm-adjusted p < 0.05"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "tabpfn_effect_size_heatmap.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Figure 5–7:
# Performance-difference curves
# ============================================================

def create_performance_difference_plot(
    raw: pd.DataFrame,
    dataset: str,
    metric: str,
) -> None:
    """
    Plot TabPFN - baseline mean performance by training size.
    """

    subset = raw[
        raw["Dataset"] == dataset
    ].copy()

    grouped = (
        subset.groupby(
            ["Samples", "Model"],
            as_index=False,
        )[metric]
        .mean()
    )

    pivot = grouped.pivot(
        index="Samples",
        columns="Model",
        values=metric,
    )

    plt.figure(figsize=(11, 7))

    for baseline in [
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]:
        if (
            "TabPFN" not in pivot.columns
            or baseline not in pivot.columns
        ):
            continue

        difference = (
            pivot["TabPFN"]
            - pivot[baseline]
        )

        plt.plot(
            difference.index,
            difference.values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"TabPFN − {baseline}",
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Training Size")
    plt.ylabel(
        f"TabPFN − baseline {metric}"
    )

    plt.xscale("log")

    plt.legend()

    plt.title(
        f"{dataset}: {metric} Performance Difference by Training Size"
    )

    plt.tight_layout()

    safe_dataset = dataset.lower().replace(
        " ",
        "_",
    )

    safe_metric = metric.lower().replace(
        "_",
        "-",
    )

    output = (
        OUTPUT_DIR
        / f"{safe_dataset}_{safe_metric}_difference.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Figure 8:
# ROC-AUC vs prediction-time Pareto scatter
# ============================================================

def create_pareto_plot(
    model_summary: pd.DataFrame,
    computational: pd.DataFrame,
) -> None:
    """Create ROC-AUC vs prediction-time scatter plot."""

    roc = model_summary[
        model_summary["Metric"] == "ROC_AUC"
    ][
        [
            "Dataset",
            "Model",
            "Mean",
        ]
    ].copy()

    comp = computational[
        [
            "Dataset",
            "Model",
            "Prediction_Time_Mean",
        ]
    ].copy()

    merged = roc.merge(
        comp,
        on=["Dataset", "Model"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No matching rows found for ROC-AUC "
            "and computational summary."
        )

    plt.figure(figsize=(11, 8))

    for model in MODELS:
        subset = merged[
            merged["Model"] == model
        ]

        plt.scatter(
            subset["Prediction_Time_Mean"],
            subset["Mean"],
            s=70,
            label=model,
        )

        for row in subset.itertuples():
            plt.annotate(
                f"{row.Dataset}",
                (
                    row.Prediction_Time_Mean,
                    row.Mean,
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xscale("log")

    plt.xlabel(
        "Mean Prediction Time (seconds, log scale)"
    )

    plt.ylabel("Mean ROC-AUC")

    plt.title(
        "ROC-AUC vs Prediction Time"
    )

    plt.legend(
        title="Model"
    )

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.5,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "roc_auc_prediction_time_pareto.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CREATING ADDITIONAL STATISTICAL FIGURES")
    print("=" * 70)

    raw = load_raw_results()
    stats = load_statistical_results()
    model_summary = load_model_summary()
    computational = load_computational_summary()

    print(f"\nRaw observations: {len(raw)}")
    print(f"Statistical comparisons: {len(stats)}")

    # --------------------------------------------------------
    # Distribution plots
    # --------------------------------------------------------

    create_distribution_plot(
        raw,
        "F1",
        "f1_distribution_by_dataset.png",
        "F1-score Distribution Across Models and Datasets",
    )

    create_distribution_plot(
        raw,
        "ROC_AUC",
        "roc_auc_distribution_by_dataset.png",
        "ROC-AUC Distribution Across Models and Datasets",
    )

    # --------------------------------------------------------
    # Forest plot
    # --------------------------------------------------------

    create_forest_plot(stats)

    # --------------------------------------------------------
    # Effect-size heatmap
    # --------------------------------------------------------

    create_effect_size_heatmap(stats)

    # --------------------------------------------------------
    # Performance-difference curves
    # --------------------------------------------------------

    for dataset_name, _ in DATASETS:

        create_performance_difference_plot(
            raw,
            dataset_name,
            "ROC_AUC",
        )

        create_performance_difference_plot(
            raw,
            dataset_name,
            "F1",
        )

    # --------------------------------------------------------
    # Pareto scatter
    # --------------------------------------------------------

    create_pareto_plot(
        model_summary,
        computational,
    )

    print("\n" + "=" * 70)
    print("STATISTICAL FIGURE CREATION COMPLETED")
    print("=" * 70)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()