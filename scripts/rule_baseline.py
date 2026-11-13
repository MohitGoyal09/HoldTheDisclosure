"""Rule baseline: append each disclosure statement the rule checker finds missing to every base reply.

python scripts/rule_baseline.py outputs/followup/eval/base_test2.jsonl --out outputs/followup/eval/base_rule_test2.jsonl

A post-processing filter, no training. The appended text is the product sheet's own statement in the
customer's script, so it passes the checker unless the reply also dismisses the risk. The model's later
turns were generated from its own unedited history (post hoc), so this approximates a deployed filter.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.replay import Transcript, load_transcripts, save_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import Sheet, load_sheet  # noqa: E402


def append_missing(reply: str, product, sheet: Sheet, script: str) -> str:
    if not product.requires_disclosure:
        return reply
    check = check_reply(reply, product, sheet, script)
    missing = [d.statement[script] for d, c in zip(product.disclosures, check.checks) if not c.present]
    return reply if not missing else reply.rstrip() + "\n\n" + " ".join(missing)


def rule_transcript(t: Transcript, sheet: Sheet, name: str = "base_rule") -> Transcript:
    p = sheet.product(t.product_id)
    return replace(t, model_name=name, replies=tuple(append_missing(r, p, sheet, t.script) for r in t.replies))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="base_rule")
    a = ap.parse_args()
    sheet = load_sheet()
    out = [rule_transcript(t, sheet, a.name) for t in load_transcripts(a.transcripts)]
    save_transcripts(out, a.out)
    print(f"wrote {len(out)} transcripts to {a.out}")


if __name__ == "__main__":
    main()
