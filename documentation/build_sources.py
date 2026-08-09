"""Generate LaTeX sections and tables from the project's validated sources."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

CITATIONS = {
    "Chen and Guestrin": r"\textcite{chen2016xgboost}",
    "Ke et al.": r"\textcite{ke2017lightgbm}",
    "Prokhorenkova et al.": r"\textcite{prokhorenkova2018catboost}",
    "Hollmann et al.": r"\textcite{hollmann2025tabpfn}",
}


def tex_inline(text: str) -> str:
    text = text.replace("$", r"\$")
    for source, replacement in CITATIONS.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)
    text = text.replace("_", r"\_")
    return text


def title(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text).strip()


def markdown_to_tex(source: Path, destination: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    list_mode: str | None = None
    quote_mode = False

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            output.append(r"\end{" + list_mode + "}")
            list_mode = None

    def close_quote() -> None:
        nonlocal quote_mode
        if quote_mode:
            output.append(r"\end{quote}")
            quote_mode = False

    heading_commands = {1: "section", 2: "subsection", 3: "subsubsection"}
    for raw in lines:
        line = raw.strip()
        if not line or line == "---":
            close_list(); close_quote()
            if output and output[-1] != "": output.append("")
            continue
        if line == "[":
            close_list(); close_quote(); output.append(r"\[")
            continue
        if line == "]":
            output.append(r"\]")
            continue
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            close_list(); close_quote()
            command = heading_commands[len(match.group(1))]
            output.append(rf"\{command}{{{title(tex_inline(match.group(2)))}}}")
            continue
        if line.startswith("> "):
            close_list()
            if not quote_mode:
                output.append(r"\begin{quote}"); quote_mode = True
            output.append(tex_inline(line[2:]))
            continue
        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        bullet = re.match(r"^-\s+(.+)$", line)
        if numbered or bullet:
            close_quote()
            wanted = "enumerate" if numbered else "itemize"
            if list_mode and list_mode != wanted: close_list()
            if not list_mode:
                output.append(r"\begin{" + wanted + "}"); list_mode = wanted
            output.append(r"\item " + tex_inline((numbered or bullet).group(1)))
            continue
        close_list(); close_quote()
        output.append(tex_inline(line))
    close_list(); close_quote()
    rendered = "\n\n".join(output) + "\n"
    rendered = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda match: "\\[\n" + re.sub(r"\n\s*\n", "\n", match.group(1).strip()) + "\n\\]",
        rendered,
        flags=re.DOTALL,
    )
    if source.stem == "methodology":
        rendered = rendered.replace("The Adult dataset is", "The Adult dataset \\cite{becker1996adult} is", 1)
        rendered = rendered.replace("The Bank Marketing dataset represents", "The Bank Marketing dataset \\cite{moro2014bank} represents", 1)
        rendered = rendered.replace("Credit-G is a binary", "Credit-G \\cite{openmlcreditg} is a binary", 1)
        rendered = rendered.replace("The Wilcoxon signed-rank test was used", "The Wilcoxon signed-rank test \\cite{wilcoxon1945} was used", 1)
    destination.write_text(rendered, encoding="utf-8")


def fmt(value: str, places: int = 4) -> str:
    try:
        number = float(value)
    except ValueError:
        return value.replace("_", r"\_")
    if number == 0:
        return "<0.0001"
    if abs(number) < 0.0001:
        return f"{number:.2e}"
    return f"{number:.{places}f}"


def write_table(name: str, caption: str, label: str, columns: list[str], rows: list[list[str]], spec: str, landscape: bool = False) -> None:
    opening = [r"\begin{landscape}", r"\small"] if landscape else []
    closing = [r"\end{longtable}", r"\end{landscape}"] if landscape else [r"\end{longtable}"]
    body = opening + [r"\begin{longtable}{" + spec + "}", rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule",
            " & ".join(columns) + r" \\", r"\midrule", r"\endfirsthead", r"\toprule",
            " & ".join(columns) + r" \\", r"\midrule", r"\endhead"]
    body += [" & ".join(row) + " \\\\" for row in rows]
    body += [r"\bottomrule"] + closing
    (OUT / "tables" / name).write_text("\n".join(body) + "\n", encoding="utf-8")


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_tables() -> None:
    best = read_csv("results/tables/paper/table_best_models.csv")
    write_table("best_models.tex", "Best mean performance by dataset and metric.", "tab:best-models",
                ["Dataset", "Metric", "Best model", "Mean", "Std"],
                [[r["Dataset"], r["Metric"].replace("_", r"\_"), r["Best_Model"], fmt(r["Mean"]), fmt(r["Std"])] for r in best], "llLNN")

    paired = read_csv("results/tables/paper/table_tabpfn_vs_trees.csv")
    write_table("paired_statistics.tex", "Paired statistical comparison of TabPFN against gradient-boosted tree models. Difference is TabPFN minus the tree-model mean.", "tab:paired-statistics",
                ["Dataset", "Metric", "Tree", "$\\Delta$", "$p$ (unadjusted)", "$p$ (Holm)", "Direction", "Holm significant"],
                [[r["Dataset"], r["Metric"].replace("_", r"\_"), r["Tree_Model"], fmt(r["TabPFN_Minus_Tree"]), fmt(r["Wilcoxon_p"]), fmt(r["Holm_Adjusted_p"]), r["Direction"], r["Significant_Holm_p05"].replace("True", "Yes").replace("False", "No")] for r in paired], "p{2.5cm}p{1.8cm}p{1.9cm}R{1.6cm}R{2.1cm}R{1.8cm}p{3.0cm}p{2.1cm}", landscape=True)

    summary = read_csv("results/tables/paper/table_model_summary.csv")
    selected = [r for r in summary if r["Metric"] in {"F1", "ROC_AUC"}]
    write_table("model_summary.tex", "Model-level aggregate performance for F1 and ROC-AUC (mean $\\pm$ standard deviation).", "tab:model-summary",
                ["Dataset", "Model", "Metric", "$n$", "Mean", "Std"],
                [[r["Dataset"], r["Model"], r["Metric"].replace("ROC_AUC", "ROC-AUC"), r["N"], fmt(r["Mean"]), fmt(r["Std"])] for r in selected], "llcNNN")

    regimes = read_csv("results/tables/paper/table_training_regimes.csv")
    selected = [r for r in regimes if r["Metric"] == "ROC_AUC"]
    write_table("training_regimes.tex", "Training-regime comparisons for ROC-AUC. Difference is TabPFN minus the tree-model mean.", "tab:training-regimes",
                ["Dataset", "Training regime", "Model comparison", "$n$", "TabPFN", "Tree", "$\\Delta$"],
                [[r["Dataset"], r["Training_Regime"], r["Comparison"], r["N"], fmt(r["TabPFN_Mean"]), fmt(r["Tree_Mean"]), fmt(r["Mean_Difference"])] for r in selected], "llLNNNN")

    computation = read_csv("results/tables/computational_performance.csv")
    write_table("computational_performance.tex", "Mean computational measurements in seconds, reported separately for training and prediction.", "tab:computational-performance",
                ["Dataset", "Model", "Training Mean", "Training Std", "Prediction Mean", "Prediction Std"],
                [[r["Dataset"], r["Model"], fmt(r["Training_Time_Mean"], 3), fmt(r["Training_Time_Std"], 3), fmt(r["Prediction_Time_Mean"], 3), fmt(r["Prediction_Time_Std"], 3)] for r in computation], "llNNNN")


def main() -> None:
    sections = ["abstract", "introduction", "methodology", "related_work", "results", "discussion", "conclusion"]
    for section in sections:
        markdown_to_tex(ROOT / "paper" / f"{section}.md", OUT / "sections" / f"{section}.tex")
    build_tables()


if __name__ == "__main__":
    main()
