# LaTeX manuscript

This directory is a self-contained LaTeX rendering of the existing manuscript and validated benchmark outputs. It does not modify the research sources in `paper/` or `results/`.

## Contents

- `main.tex` is the manuscript entry point.
- `sections/` contains LaTeX converted from `paper/*.md` plus the table/figure placements.
- `tables/` contains formatted LaTeX tables generated from validated CSV results.
- `figures/` contains copies of the existing benchmark figures.
- `references.bib` is the bibliography.
- `build/` is the suggested output directory.

## Regenerate source-derived files

From the repository root, run:

```powershell
python documentation/build_sources.py
```

## Compile

From `documentation/`, use a TeX installation with Biber:

```powershell
pdflatex -output-directory=build main.tex
biber build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

The PDF will be at `build/main.pdf`.
