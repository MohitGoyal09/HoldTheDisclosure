"""Write Hugging Face model cards for the released adapters, with numbers read from the artifacts.

python scripts/model_cards.py --out outputs/staging/cards

Each card gets base model, recipe, data, DR@5 on test and held-out (from cmp_*.json), helpfulness
(LLM panel majority), the training run and the licence. No number is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.helpful import helpful_rates  # noqa: E402

GITHUB = "https://github.com/MohitGoyal09/HoldTheDisclosure"
DATA = "Rockerleo/hold-the-disclosure"
TINY = "CohereLabs/tiny-aya-fire"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
SFT = "LoRA SFT (TRL 0.29.1), rank 16, alpha 32, lr 1e-4, 1 epoch, 4-bit base, 858 rows of `pairs_v2/sft_train.jsonl`"


def dpo(lr: str, mix: str, rows: int) -> str:
    return f"LoRA DPO through AlignTune (TRL backend), beta 0.1, lr {lr}, rank 16, 1 epoch, 4-bit base, {rows} rows ({mix})"


ADAPTERS = {
    "htd-tiny-aya-fire-sft-lora": ("sft", TINY, SFT + ", training run `htd-train` v6"),
    "htd-tiny-aya-fire-sft-lora-seed1": ("sft_s1", TINY, SFT + ", seed 1 (seeded LoRA init)"),
    "htd-tiny-aya-fire-sft-lora-v5": ("sft_v5", TINY, SFT + ", training run `htd-train` v5"),
    "htd-tiny-aya-fire-aligntune-dpo-lora": ("at", TINY, dpo("5e-6", "drop, lecture and warn pairs", 1496)),
    "htd-tiny-aya-fire-aligntune-dpo-dropwarn-lora": ("at_drop", TINY, dpo("5e-6", "drop and warn pairs", 858)),
    "htd-tiny-aya-fire-aligntune-dpo-dropwarn-lr5e5-lora": ("at_drop_lr5e5", TINY, dpo("5e-5", "drop and warn pairs", 858)),
    "htd-tiny-aya-fire-aligntune-dpo-lr5e5-lora": ("at_lr5e5", TINY, dpo("5e-5", "drop, lecture and warn pairs", 1496)),
    "htd-llama-3.2-3b-sft-lora": ("llama_sft", LLAMA, SFT + ", seed 1, on Llama-3.2-3B-Instruct"),
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def card(repo: str, label: str, base: str, recipe: str, cmps: dict, helpful: dict) -> str:
    licence = "llama3.2" if base == LLAMA else "cc-by-nc-4.0"
    rows = []
    for split, name in (("test", "test (unseen pushback lines)"), ("heldout", "held-out (unseen products and tactics)")):
        c = cmps.get(split)
        if c:
            s = c["significance"]
            rows.append(f"| {name} | {pct(c['mean_a'])} | {pct(c['mean_b'])} | {100 * c['diff']:+.1f} "
                        f"[{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}] | {s['mcnemar_p']:.2g} |")
    help_rows = []
    for kind, rates in helpful.items():
        if label in rates:
            help_rows.append(f"| {kind} | {100 * rates[label]['yes'] / rates[label]['n']:.0f}% (n={rates[label]['n']}) |")
    return f"""---
base_model: {base}
library_name: peft
license: {licence}
language: [en, hi]
tags: [lora, safety, financial-disclosure, multi-turn, hinglish, lexsi-hackathon-2026]
datasets: [{DATA}]
---

# {repo}

LoRA adapter for `{base}` from the *Hold the disclosure* project (Lexsi AI Safety Hackathon 2026,
Track 1). Goal: a bank support assistant keeps required risk disclosures across five turns of
customer pushback in English, Devanagari Hindi and romanized Hinglish, without refusing or
lecturing. The bank and products are fictional.

**Training.** {recipe}. Data: [`{DATA}`](https://huggingface.co/datasets/{DATA}).

## Evaluation

DR@5 is the share of 60 dialogues (20 per script) whose turn-5 reply states every required
disclosure, by the project's rule checker, against the same base model with a plain prompt.
Interval: 95% percentile bootstrap of the paired difference; p: exact McNemar test.

| split | base | this adapter | change, points [95% CI] | McNemar p |
|---|---|---|---|---|
{chr(10).join(rows)}

Helpfulness at turn 5 (majority of three LLMs answering whether the reply gives the help the
customer asked for, regardless of warnings; an LLM measure, not human labels):

| products | helpful |
|---|---|
{chr(10).join(help_rows)}

Full results, methods, limitations and deviations: [{GITHUB}]({GITHUB}) (`docs/REPORT.md`,
`docs/RESULTS_FULL.md`).

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "Rockerleo/{repo}")
```

## Limitations

Synthetic dialogues about one fictional bank; 20 test dialogues per script; a scripted pushback
ladder; a rule checker that rewards key facts, not full legal wording. Not for real financial
advice or deployment. The adapter inherits the base model's licence (`{licence}`); Tiny Aya
adapters are for non-commercial research only.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=Path("outputs/kaggle-eval-2x2/htd/outputs/eval"))
    ap.add_argument("--llama", type=Path, default=Path("outputs/kaggle-second-model/htd/outputs/eval_llama"))
    ap.add_argument("--judges", type=Path, default=Path("outputs/judge_local"))
    ap.add_argument("--out", type=Path, default=Path("outputs/staging/cards"))
    a = ap.parse_args()
    panels = [[json.loads(x) for x in (a.judges / f"helpful_{n}.jsonl").read_text().splitlines() if x.strip()]
              for n in ("llama", "deepseek", "gptoss")]
    helpful = {"disclosure products": helpful_rates([[r for r in p if r["requires_disclosure"]] for p in panels]),
               "control products": helpful_rates([[r for r in p if not r["requires_disclosure"]] for p in panels])}
    for repo, (label, base, recipe) in ADAPTERS.items():
        d = a.llama if label.startswith("llama") else a.eval
        cmps = {s: json.loads((d / f"cmp_{label}_{s}.json").read_text()) for s in ("test", "heldout")
                if (d / f"cmp_{label}_{s}.json").exists()}
        if len(cmps) != 2:
            raise SystemExit(f"{repo}: missing comparison files in {d}")
        (a.out / repo).mkdir(parents=True, exist_ok=True)
        (a.out / repo / "README.md").write_text(card(repo, label, base, recipe, cmps, helpful), encoding="utf-8")
    print(f"wrote {len(ADAPTERS)} cards to {a.out}")


if __name__ == "__main__":
    main()
