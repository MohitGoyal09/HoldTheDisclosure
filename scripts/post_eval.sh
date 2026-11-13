#!/bin/bash
# After an eval kernel finishes: download its outputs, print failure lines, run the local judges
# (v2 + ablations), the LLM panel and the helpfulness panel on every test/control transcript
# (--resume skips rows already done), and stage transcripts + adapter sidecars for reuse.
#
# usage: scripts/post_eval.sh <kernel-slug | owner/kernel-slug> <local-out-dir> <eval-subdir: outputs/eval | outputs/eval_llama>
# needs: kaggle CLI, a Hugging Face token (huggingface_hub.get_token) for the hosted judges.
set -uo pipefail
K=$1; OUT=$2; SUB=$3
cd "$(dirname "$0")/.."
rm -rf "$OUT" && mkdir -p "$OUT"
case "$K" in */*) REF=$K;; *) REF=rockerleo/$K;; esac
kaggle kernels output "$REF" -p "$OUT" >/dev/null 2>&1
E="$OUT/htd/$SUB"
echo "== $K"
python3 - "$OUT" <<'EOF'
import glob, json, re, sys
for f in glob.glob(sys.argv[1] + "/*.log"):
    t = "".join(x.get("data", "") for x in json.load(open(f)))
    for pat in (r"FAILED: \[[^\]]*\]", r"ARM FAILED[^\n]*", r"judge rows \d+ error rate [0-9.]+", r"RUN \d+ \| \d/6 \| \w+", r"Traceback"):
        for m in re.findall(pat, t):
            print(" ", m)
EOF
ls "$E" | tr '\n' ' '; echo

export HF_TOKEN=$(.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export HUGGINGFACE_API_KEY=$HF_TOKEN
O=outputs/judge_local
mkdir -p $O
TEST=$(ls "$E"/*_test*.jsonl | grep -v -e strict -e third)
CTRL=$(ls "$E"/*_control*.jsonl)
Q=huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507
G=huggingface/google/gemma-3-27b-it
judge() { .venv/bin/python scripts/judge_run.py $TEST --turn 5 --rubric "$1" --model "$2" --out "$O/$3.jsonl" --workers 4 --resume > "$O/$3.log" 2>&1; echo "  $3 exit $?"; }
judge v2 $Q v2_qwen & judge v1 $Q v1_qwen & judge v2 $G v2_gemma &
judge panel huggingface/meta-llama/Llama-3.3-70B-Instruct panel_llama &
judge panel huggingface/deepseek-ai/DeepSeek-V3-0324 panel_deepseek &
judge panel huggingface/openai/gpt-oss-120b panel_gptoss &
for pair in llama=meta-llama/Llama-3.3-70B-Instruct deepseek=deepseek-ai/DeepSeek-V3-0324 gptoss=openai/gpt-oss-120b; do
  n=${pair%%=*}; m=${pair#*=}
  ( .venv/bin/python scripts/helpful_run.py $TEST $CTRL --turn 5 --model huggingface/$m --out $O/helpful_$n.jsonl --workers 4 --resume > $O/helpful_$n.log 2>&1; echo "  helpful_$n exit $?" ) &
done
wait
.venv/bin/python scripts/panel_vote.py $O/panel_llama.jsonl $O/panel_deepseek.jsonl $O/panel_gptoss.jsonl --out $O/panel_reference.jsonl | tr -d '\n '; echo

if [ "$SUB" = "outputs/eval" ]; then
  STAGE=outputs/staging/base_eval
  mkdir -p $STAGE
  cp "$E"/*.jsonl "$E"/*.adapter.json $STAGE/ 2>/dev/null
  echo "staged $(ls $STAGE/*.adapter.json | wc -l) sidecars in $STAGE"
fi
echo "POST_EVAL_DONE $K"
