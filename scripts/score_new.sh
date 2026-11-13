#!/usr/bin/env bash
# Score new test2 transcripts with both panels (helpfulness and disclosure; turn 5), then re-render the results.
# Usage: HF_TOKEN=... scripts/score_new.sh outputs/kaggle-eval-rule/eval/sft_rule_s1_test2.jsonl [...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
for pair in llama:meta-llama/Llama-3.3-70B-Instruct deepseek:deepseek-ai/DeepSeek-V3-0324 gptoss:openai/gpt-oss-120b; do
  n=${pair%%:*}; m=huggingface/${pair#*:}
  "$PY" scripts/helpful_run.py "$@" --turn 5 --model "$m" --out "outputs/judge_test2/helpful_$n.jsonl" --resume &
  "$PY" scripts/judge_run.py "$@" --turn 5 --rubric panel --model "$m" --out "outputs/judge_disc_test2/panel_$n.jsonl" --workers 6 --resume &
done
wait
"$PY" scripts/render_robustness.py
