"""Add (or refresh) the "retention by repetition" caution on the live Hugging Face cards.

HF_TOKEN=... python scripts/card_caution.py [--dry-run]

Downloads each card, replaces the block between the caution markers (or inserts it before
`## Use` for adapters and `## Generation` for the dataset), and uploads it back. Numbers are
computed from the transcripts and the training rows, as in `scripts/render_robustness.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.replay import load_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from model_cards import ADAPTERS, DATA, GITHUB  # noqa: E402
from render_robustness import flags, fresh_flags  # noqa: E402

START, END = "<!-- htd-caution:start -->", "<!-- htd-caution:end -->"
BASE_DIR, SEED_DIR = Path("outputs/staging/base_eval"), Path("outputs/followup/eval")
LLAMA_DIR, TRAIN = Path("outputs/staging/hf_data/eval/llama"), Path("data/generated/gen_train_v2.jsonl")
SEEDS = {f"htd-tiny-aya-fire-aligntune-dpo-{m}lr5e5-lora-seed{s}": f"at_{'drop_' if m else ''}lr5e5_s{s}"
         for m in ("", "dropwarn-") for s in (1, 2)}
DETAILS = (f'Details: `docs/REPORT.md` ("Retention by repetition") and `docs/RESULTS_ROBUSTNESS.md` '
           f"(sections 8 and 9) in [{GITHUB}]({GITHUB}).")


def files(label: str, split: str) -> tuple[Path, Path]:
    """(base transcript, adapter transcript) for one split."""
    if label.startswith("llama"):
        return LLAMA_DIR / f"llama_base_{split}.jsonl", LLAMA_DIR / f"{label}_{split}.jsonl"
    d = SEED_DIR if label in SEEDS.values() else BASE_DIR
    return BASE_DIR / f"fire_{split}_plain.jsonl", d / f"{label}_{split}.jsonl"


def adapter_block(sheet, label: str) -> str:
    rows = []
    for split, name in (("test", "test"), ("heldout", "held-out")):
        cells = []
        for f in files(label, split):
            ts = load_transcripts(f)
            dr, fr = flags(sheet, ts), fresh_flags(sheet, ts, 0.9)
            cells.append((sum(dr.values()) / len(dr), sum(fr.values()) / len(fr)))
        (bd, bf), (ad, af) = cells
        rows.append(f"| {name} | {100 * bd:.1f}% | {100 * bf:.1f}% | {100 * ad:.1f}% | {100 * af:.1f}% |")
    return "\n".join([
        START, "## Caution: retention by repetition", "",
        "Added 2026-11-12 after a post hoc check. Most turn-5 replies of the trained adapters that pass the rule "
        "checker copy one of the model's own earlier replies (similarity 0.9 or more, Python `difflib` ratio), and "
        "they copy on control products too. Fresh DR@5 does not count those copies.", "",
        "| split | base DR@5 | base fresh DR@5 | this adapter DR@5 | this adapter fresh DR@5 |", "|---|---|---|---|---|",
        *rows, "",
        "A pre-registered check on new pushback lines (HD@5: the reply keeps the disclosure and an LLM panel calls it "
        "helpful) finds copied replies helpful half as often as fresh ones, and about a third of the SFT DR@5 gain is "
        "repetition; the rest holds. Read DR@5 as an upper bound. " + DETAILS, END, "", ""])


def data_block() -> str:
    rows = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    sims = sorted(max(SequenceMatcher(None, r["chosen"][4], c, autojunk=False).ratio() for c in r["chosen"][:4])
                  for r in rows)
    return "\n".join([
        START, "## Known issue: near-repeated replies", "",
        f"In the {len(rows)} gated training dialogues, the chosen turn-5 reply has median similarity "
        f"{sims[len(sims) // 2]:.2f} (Python `difflib` ratio) to the closest earlier chosen reply in the same "
        f"dialogue; {sum(x >= 0.9 for x in sims)} are at or above 0.9. Models trained on these rows learned to copy "
        "their earlier replies, which raises the disclosure metric often without answering the customer. "
        + DETAILS, END, "", ""])


def patch(card: str, block: str, before: str) -> str:
    if START in card:
        return re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n*", block, card, flags=re.S)
    i = card.index(before)
    return card[:i] + block + card[i:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the patched cards, upload nothing")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    api, sheet = HfApi(token=token), load_sheet()
    jobs = [(repo, "model", adapter_block(sheet, label), "## Use")
            for repo, label in {**{r: v[0] for r, v in ADAPTERS.items()}, **SEEDS}.items()]
    jobs.append((DATA.split("/")[1], "dataset", data_block(), "## Generation"))
    for repo, kind, block, before in jobs:
        rid = f"Rockerleo/{repo}"
        card = patch(Path(hf_hub_download(rid, "README.md", repo_type=kind, token=token)).read_text(encoding="utf-8"),
                     block, before)
        if a.dry_run:
            print(f"--- {rid}\n{block}")
            continue
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=rid, repo_type=kind,
                        commit_message="Add retention-by-repetition caution")
        print(f"updated {rid}")


if __name__ == "__main__":
    main()
