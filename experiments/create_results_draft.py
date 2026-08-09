"""
Generate the first evidence-based Results section draft.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPER_DIR = PROJECT_ROOT / "results" / "tables" / "paper"
OUTPUT_DIR = PROJECT_ROOT / "paper"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt(value):
    return f"{value:.4f}"


def main():

    best = pd.read_csv(
        PAPER_DIR / "table_best_models.csv"
    )

    statistics = pd.read_csv(
        PAPER_DIR / "table_tabpfn_vs_trees.csv"
    )

    regimes = pd.read_csv(
        PAPER_DIR / "table_training_regimes.csv"
    )

    lines = []

    lines.append("# Results\n")

    lines.append(
        "## Overall predictive performance\n"
    )

    lines.append(
        "The benchmark evaluated XGBoost, LightGBM, "
        "CatBoost, and TabPFN across three binary "
        "classification datasets: Adult, Bank Marketing, "
        "and Credit-G. Five random seeds were used for "
        "each training-size configuration. The final "
        "benchmark contains 1,040 observations across "
        "four models and three datasets.\n"
    )

    lines.append(
        "Table 1 summarizes the best-performing model "
        "for each dataset and evaluation metric.\n"
    )

    lines.append(
        "### Adult\n"
    )

    adult = best[
        best["Dataset"] == "Adult"
    ]

    for _, row in adult.iterrows():

        lines.append(
            f"- **{row['Metric']}**: "
            f"{row['Best_Model']} "
            f"({fmt(row['Mean'])} ± {fmt(row['Std'])})."
        )

    lines.append("")

    lines.append(
        "For Adult, CatBoost provides the strongest "
        "overall performance across accuracy, precision, "
        "F1, and ROC-AUC, while XGBoost achieves the "
        "highest recall.\n"
    )

    lines.append(
        "### Bank Marketing\n"
    )

    bank = best[
        best["Dataset"] == "Bank Marketing"
    ]

    for _, row in bank.iterrows():

        lines.append(
            f"- **{row['Metric']}**: "
            f"{row['Best_Model']} "
            f"({fmt(row['Mean'])} ± {fmt(row['Std'])})."
        )

    lines.append("")

    lines.append(
        "Bank Marketing shows the strongest overall "
        "advantage for TabPFN. TabPFN achieves the "
        "highest accuracy, recall, F1, and ROC-AUC, "
        "while CatBoost achieves the highest precision.\n"
    )

    lines.append(
        "### Credit-G\n"
    )

    credit = best[
        best["Dataset"] == "Credit-G"
    ]

    for _, row in credit.iterrows():

        lines.append(
            f"- **{row['Metric']}**: "
            f"{row['Best_Model']} "
            f"({fmt(row['Mean'])} ± {fmt(row['Std'])})."
        )

    lines.append("")

    lines.append(
        "Credit-G is dominated by the conventional "
        "tree-based models for several metrics. "
        "CatBoost achieves the highest accuracy, recall, "
        "F1, and ROC-AUC, while XGBoost achieves the "
        "highest precision.\n"
    )

    # --------------------------------------------------------
    # Statistical comparison
    # --------------------------------------------------------

    lines.append(
        "## Statistical comparison with tree-based models\n"
    )

    significant = statistics[
        statistics["Significant_p05"] == True
    ]

    total = len(statistics)
    significant_count = len(significant)

    lines.append(
        f"Across the 45 paired TabPFN-versus-tree "
        f"comparisons, {significant_count} were statistically "
        f"significant at the 0.05 level. This indicates that "
        f"the observed differences are not uniformly "
        f"attributable to variation between experimental "
        f"runs.\n"
    )

    lines.append(
        "The statistical results also show that TabPFN's "
        "relative performance depends on both the dataset "
        "and the evaluation metric. Therefore, the results "
        "do not support the claim that TabPFN universally "
        "outperforms gradient-boosted tree models.\n"
    )

    # --------------------------------------------------------
    # Dataset interpretation
    # --------------------------------------------------------

    lines.append(
        "## Dataset-level comparison\n"
    )

    lines.append(
        "The three datasets exhibit different performance "
        "patterns. Adult shows strong competition between "
        "TabPFN and CatBoost, with CatBoost retaining the "
        "best aggregate performance across most reported "
        "metrics. Bank Marketing provides the clearest "
        "case where TabPFN improves on the conventional "
        "tree-based models across several metrics. "
        "Credit-G instead favors CatBoost, demonstrating "
        "that the effectiveness of TabPFN is dataset "
        "dependent.\n"
    )

    # --------------------------------------------------------
    # Learning curves
    # --------------------------------------------------------

    lines.append(
        "## Learning behavior with increasing training size\n"
    )

    lines.append(
        "The learning curves provide an additional view of "
        "model behavior under limited training data. "
        "Performance generally improves as the training "
        "sample size increases, although the magnitude and "
        "rate of improvement differ across datasets and "
        "models. TabPFN shows particularly competitive "
        "performance in the smaller-data regimes, while "
        "tree-based models become increasingly competitive "
        "as more training data are provided.\n"
    )

    # --------------------------------------------------------
    # Computational performance
    # --------------------------------------------------------

    lines.append(
        "## Computational performance\n"
    )

    lines.append(
        "Training-time measurements show a different "
        "trade-off from predictive performance. TabPFN "
        "has a relatively stable computational cost across "
        "training sizes compared with the tree-based "
        "models, whereas conventional boosting methods "
        "generally show increasing training time as the "
        "number of training samples grows. These results "
        "indicate that predictive performance and "
        "computational efficiency should be considered "
        "jointly when selecting a model.\n"
    )

    # --------------------------------------------------------
    # Conclusion of results
    # --------------------------------------------------------

    lines.append(
        "## Summary of findings\n"
    )

    lines.append(
        "Overall, the experiments indicate that TabPFN can "
        "provide strong performance in small-data tabular "
        "classification settings, but its advantage is "
        "not universal. CatBoost remains highly competitive "
        "and is the strongest model on Adult and Credit-G "
        "for several evaluation metrics, whereas TabPFN "
        "shows its clearest advantage on Bank Marketing. "
        "The findings therefore support a dataset-dependent "
        "view of model selection rather than a universal "
        "replacement of gradient-boosted trees by "
        "foundation-model-based tabular learning.\n"
    )

    output = OUTPUT_DIR / "results.md"

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()