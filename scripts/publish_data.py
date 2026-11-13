"""Publish the data and training pairs to a Hugging Face dataset repo.

python scripts/publish_data.py --repo Rockerleo/hold-the-disclosure --private
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CARD = """---
license: other
language:
  - en
  - hi
task_categories:
  - text-generation
tags:
  - safety
  - dpo
  - sft
  - indic
  - hinglish
  - financial-disclosure
---

# hold-the-disclosure data and evaluation logs

Synthetic bank-support dialogues for training and evaluating disclosure retention under
multi-turn customer pushback (English, Devanagari Hindi, romanized Hinglish), plus every
evaluation transcript and judge log behind the results. The bank and products are fictional;
ELSS (3-year lock-in) and ULIP (5-year lock-in) follow real Indian rules. Code and write-up:
https://github.com/MohitGoyal09/HoldTheDisclosure

## Training data

| Path | What |
|---|---|
| `data/product_sheet.json` | products, required disclosures, key facts per script |
| `data/seeds/pushback_seeds.jsonl` | pushback lines per ladder slot and script |
| `train_v2.jsonl`, `control_train_v2.jsonl` | training dialogues (disclosure and control products) |
| `gen_train_v2.jsonl`, `gen_control_train_v2.jsonl` | candidate replies (chosen, drop, lecture, warn) behind the rule gates |
| `pairs_v2/` | **the rows every released adapter was trained on**: DPO 1,496 train / 174 val, SFT 858 / 102 |
| `pairs/`, `train_dialogues.jsonl`, `gen_train.jsonl`, ... (no `_v2`) | earlier v1 data, kept for the record; no released adapter uses it |

Splits are cut by whole dialogue; seed lines are split train / test / held-out.

## Evaluation logs

| Path | What |
|---|---|
| `eval/tiny_aya/` | transcripts (5 turns each) for base Tiny Aya Fire and seven adapters on test, held-out and control; `summary_*.json`, `cmp_*.json` |
| `eval/llama/` | the same for Llama-3.2-3B base and SFT |
| `judges/v1_gemma_rubric_v1.jsonl` | pre-registered judge labels (turn 5, test) |
| `judges/v2_qwen.jsonl`, `v1_qwen.jsonl`, `v2_gemma.jsonl` | selected judge and ablations |
| `judges/panel_*.jsonl`, `panel_reference.jsonl` | three-model LLM reference panel and its majority (an LLM reference, not human labels) |
| `judges/helpful_*.jsonl` | helpfulness answers of the panel models |
| `auditkit_compare/` | comparisons recomputed with AuditKIT 1.1.3 installed |
| `labels/blind_label_sheet.csv` | 100-row blind sheet (no system name, no judge label) for future human labels |
| `experiment_criteria.jsonl` | the six pre-registered criteria per trained system |

## Generation

Replies were generated with a hosted model (GLM-4.7-Flash) behind three gates: the rule checker
(every disclosure present, no dismissal), a length gate, and a reward judge. Judges and the
panel are different model families from the generator and from the models under test.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Rockerleo/hold-the-disclosure")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--files", nargs="*", default=None)
    a = ap.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    defaults = [
        "data/product_sheet.json",
        "data/seeds/pushback_seeds.jsonl",
        "data/generated/train_v2.jsonl",
        "data/generated/control_train_v2.jsonl",
        "data/generated/gen_train_v2.jsonl",
        "data/generated/gen_train_v2.report.json",
        "data/generated/gen_control_train_v2.jsonl",
        "data/generated/pairs_v2/dpo_train.jsonl",
        "data/generated/pairs_v2/dpo_val.jsonl",
        "data/generated/pairs_v2/sft_train.jsonl",
        "data/generated/pairs_v2/sft_val.jsonl",
    ]
    files = a.files or defaults
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset")
    uploaded = []
    for rel in files:
        p = root / rel
        if not p.exists():
            print(f"skip missing {rel}")
            continue
        name = rel.split("data/generated/")[-1] if "data/generated/" in rel else rel
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=a.repo, repo_type="dataset")
        uploaded.append(name)
    print(json.dumps({"repo": a.repo, "uploaded": uploaded}, indent=2))


if __name__ == "__main__":
    main()
