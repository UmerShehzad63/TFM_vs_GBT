from pathlib import Path
import re
import sys

import pypandoc


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAPER_DIR = ROOT / "paper"
DOCUMENTATION_DIR = ROOT / "documentation"
SECTIONS_DIR = DOCUMENTATION_DIR / "sections"


# ============================================================
# Authoritative manuscript files
# ============================================================

MANUSCRIPT_FILES = [
    "abstract.md",
    "Introduction.md",
    "Research Questions and Hypotheses.md",
    "methodology.md",
    "related_work.md",
    "Discussion and Summary of Results.md",
    "conclusion.md",
    "limitations.md",
    "references.md",
]


# ============================================================
# Remove manually written section numbers
# ============================================================

def remove_manual_heading_numbers(text: str) -> str:
    """
    Remove section numbers that are already present in the
    Markdown headings.

    Example:

        # 1. Introduction

    becomes:

        # Introduction

    and:

        ## 3.2 Datasets

    becomes:

        ## Datasets

    This allows LaTeX to generate section numbering itself.
    """

    pattern = re.compile(
        r"^(#{1,6})\s+"
        r"\d+(?:\.\d+)*\.?\s+"
        r"(.*)$"
    )

    cleaned_lines = []

    for line in text.splitlines():

        match = pattern.match(line)

        if match:
            hashes = match.group(1)
            heading = match.group(2)

            cleaned_lines.append(
                f"{hashes} {heading}"
            )

        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# Convert one Markdown file directly to LaTeX
# ============================================================

def convert_file(source: Path, target: Path) -> None:

    text = source.read_text(
        encoding="utf-8"
    )

    cleaned_text = remove_manual_heading_numbers(
        text
    )

    pypandoc.convert_text(
        cleaned_text,
        to="latex",
        format="markdown",
        outputfile=str(target),
        extra_args=[
            "--wrap=none",
        ],
    )


# ============================================================
# Main
# ============================================================

def main() -> int:

    print("=" * 70)
    print("DIRECT MARKDOWN -> LATEX CONVERSION")
    print("=" * 70)

    # Create documentation/sections if necessary.
    SECTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Convert directly from paper/*.md
    # --------------------------------------------------------

    for filename in MANUSCRIPT_FILES:

        source = PAPER_DIR / filename

        target = SECTIONS_DIR / filename.replace(
            ".md",
            ".tex",
        )

        if not source.exists():

            print(
                f"\nERROR: source Markdown file not found:"
            )
            print(source)

            return 1

        print(f"\nConverting:")
        print(f"  SOURCE : {source}")
        print(f"  TARGET : {target}")

        try:

            convert_file(
                source,
                target,
            )

        except Exception as exc:

            print(
                f"\nERROR while converting {filename}:"
            )
            print(exc)

            return 1

        print(
            f"  OK     : {target.name}"
        )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFYING LATEX OUTPUT")
    print("=" * 70)

    missing = []

    for filename in MANUSCRIPT_FILES:

        tex_name = filename.replace(
            ".md",
            ".tex",
        )

        target = SECTIONS_DIR / tex_name

        if target.exists() and target.stat().st_size > 0:

            print(
                f"  OK  {tex_name}"
            )

        else:

            print(
                f"  MISSING  {tex_name}"
            )

            missing.append(tex_name)

    # --------------------------------------------------------
    # Check that no Markdown files exist in documentation
    # sections.
    # --------------------------------------------------------

    documentation_md = list(
        SECTIONS_DIR.glob("*.md")
    )

    if documentation_md:

        print(
            "\nWARNING: Markdown files found in "
            "documentation/sections:"
        )

        for path in documentation_md:
            print(f"  {path.name}")

    else:

        print(
            "\n  OK  documentation/sections contains "
            "only generated LaTeX sections."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if missing:

        print("\n" + "=" * 70)
        print("CONVERSION FAILED")
        print("=" * 70)

        return 1

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nSource manuscript:")
    print(PAPER_DIR)

    print("\nGenerated LaTeX:")
    print(SECTIONS_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())