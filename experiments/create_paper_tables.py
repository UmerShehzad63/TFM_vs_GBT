"""
Create publication-ready tables from the completed benchmark analysis.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = TABLE_DIR / "paper"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # ========================================================
    # 1. BEST MODEL BY DATASET AND METRIC
    # ========================================================

    best_models = pd.read_csv(
        TABLE_DIR / "best_models.csv"
    )

    best_models = best_models[
        [
            "Dataset",
            "Metric",
            "Best_Model",
            "Mean",
            "Std",
        ]
    ].copy()

    best_models["Mean"] = best_models["Mean"].round(4)
    best_models["Std"] = best_models["Std"].round(4)

    output = OUTPUT_DIR / "table_best_models.csv"

    best_models.to_csv(
        output,
        index=False,
    )

    print(f"Saved: {output}")

    # ========================================================
    # 2. TABPFN VS TREE MODELS
    # ========================================================

    statistics = pd.read_csv(
        TABLE_DIR
        / "statistics"
        / "significance_summary.csv"
    )

    statistics = statistics[
        [
            "Dataset",
            "Metric",
            "Model_B",
            "N",
            "Mean_A",
            "Mean_B",
            "Mean_Difference_A_minus_B",
            "CI95_Low",
            "CI95_High",
            "Wilcoxon_p",
            "Rank_Biserial_Effect",
            "Significant_p05",
            "Direction",
            "Effect_Strength",
        ]
    ].copy()

    statistics = statistics.rename(
        columns={
            "Model_B": "Tree_Model",
            "Mean_A": "TabPFN_Mean",
            "Mean_B": "Tree_Mean",
            "Mean_Difference_A_minus_B":
                "TabPFN_Minus_Tree",
        }
    )

    numeric_columns = [
        "TabPFN_Mean",
        "Tree_Mean",
        "TabPFN_Minus_Tree",
        "CI95_Low",
        "CI95_High",
        "Wilcoxon_p",
        "Rank_Biserial_Effect",
    ]

    for column in numeric_columns:
        statistics[column] = statistics[column].round(5)

    output = (
        OUTPUT_DIR
        / "table_tabpfn_vs_trees.csv"
    )

    statistics.to_csv(
        output,
        index=False,
    )

    print(f"Saved: {output}")

    # ========================================================
    # 3. OVERALL MODEL SUMMARY
    # ========================================================

    model_summary = pd.read_csv(
        TABLE_DIR
        / "statistics"
        / "model_summary.csv"
    )

    model_summary[
        ["Mean", "Std", "Min", "Max"]
    ] = model_summary[
        ["Mean", "Std", "Min", "Max"]
    ].round(4)

    output = (
        OUTPUT_DIR
        / "table_model_summary.csv"
    )

    model_summary.to_csv(
        output,
        index=False,
    )

    print(f"Saved: {output}")

    # ========================================================
    # 4. SMALL / MEDIUM / LARGE TRAINING REGIMES
    # ========================================================

    regimes = pd.read_csv(
        TABLE_DIR
        / "statistics"
        / "training_regime_comparison.csv"
    )

    regimes[
        [
            "TabPFN_Mean",
            "Tree_Mean",
            "Mean_Difference",
            "Difference_SD",
        ]
    ] = regimes[
        [
            "TabPFN_Mean",
            "Tree_Mean",
            "Mean_Difference",
            "Difference_SD",
        ]
    ].round(4)

    output = (
        OUTPUT_DIR
        / "table_training_regimes.csv"
    )

    regimes.to_csv(
        output,
        index=False,
    )

    print(f"Saved: {output}")

    # ========================================================
    # 5. HUMAN-READABLE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PAPER TABLE SUMMARY")
    print("=" * 70)

    print()
    print("BEST MODELS")
    print(best_models.to_string(index=False))

    print()
    print("SIGNIFICANT TABPFN COMPARISONS")

    significant = statistics[
        statistics["Significant_p05"]
    ]

    print(
        f"{len(significant)} / "
        f"{len(statistics)} comparisons significant."
    )

    print()
    print("TabPFN wins:")
    print(
        (
            statistics["Direction"]
            == "TabPFN higher"
        ).sum()
    )

    print()
    print("TabPFN loses:")
    print(
        (
            statistics["Direction"]
            == "TabPFN lower"
        ).sum()
    )

    print()
    print("=" * 70)
    print("PAPER TABLE CREATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()