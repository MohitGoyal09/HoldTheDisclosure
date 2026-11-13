#!/bin/bash
# Build docs/REPORT.pdf from docs/REPORT.md and docs/figures/. Needs pandoc and xelatex.
set -euo pipefail
cd "$(dirname "$0")/.."
H=$(mktemp)
trap 'rm -f "$H"' EXIT
printf '\\usepackage{caption}\n\\captionsetup{labelformat=empty,font=small}\n' > "$H"
pandoc docs/REPORT.md --resource-path=docs --pdf-engine=xelatex --shift-heading-level-by=-1 \
  --toc -V toc-title=Contents -H "$H" \
  -V geometry:margin=2.2cm -V fontsize=10pt -V mainfont="Helvetica Neue" -V monofont=Menlo \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -M subtitle="Lexsi AI Safety Hackathon 2026, Track 1" -M date="November 2026" \
  -o docs/REPORT.pdf
echo "wrote docs/REPORT.pdf"
