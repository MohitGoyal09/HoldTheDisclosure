"""Diagnose disclosure failures in transcripts.

python scripts/diagnose.py outputs/eval/dpo5_test.jsonl --turn 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.replay import load_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--turn", type=int, default=5)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    sheet = load_sheet()
    ts = [t for f in a.files for t in load_transcripts(f)]
    missing = Counter()
    by_script = Counter()
    total = Counter()
    samples = []
    for t in ts:
        if not sheet.product(t.product_id).requires_disclosure or len(t.replies) < a.turn:
            continue
        product = sheet.product(t.product_id)
        check = check_reply(t.replies[a.turn - 1], product, sheet)
        total[t.script] += 1
        if check.all_present:
            continue
        by_script[t.script] += 1
        for c in check.checks:
            if not c.present:
                missing[c.disclosure_id] += 1
        if len(samples) < a.samples:
            samples.append((t, check))
    print(f"turn {a.turn}: failures by script")
    for s in sorted(total):
        print(f"  {s}: {by_script[s]}/{total[s]} missing")
    print("missing disclosure ids:", dict(missing.most_common()))
    for t, check in samples:
        print(f"\n--- {t.dialogue_id} | {t.product_id} | {t.script}")
        print("customer:", t.user_turns[a.turn - 1])
        print("reply:", t.replies[a.turn - 1][:500])
        print("missing ids:", [c.disclosure_id for c in check.checks if not c.present])


if __name__ == "__main__":
    main()
