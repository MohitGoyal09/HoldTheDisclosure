#!/usr/bin/env bash
# One command to re-check this repo: unit tests, then the headline test2 numbers from the published HF data.
# No GPU, no API key. Usage: scripts/verify.sh [existing snapshot dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" -m pytest -q
D=${1:-${TMPDIR:-/tmp}/htd-data-snapshot}
if [ ! -f "$D/followup/eval/base_test2.jsonl" ]; then
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Rockerleo/hold-the-disclosure', repo_type='dataset', local_dir=sys.argv[1], allow_patterns=['followup/*', 'robustness/roi_strict/*'])" "$D"
fi
"$PY" scripts/verify_headline.py --data-dir "$D"
