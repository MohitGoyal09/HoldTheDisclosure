"""Recompute the headline test2 numbers from a snapshot of the HF dataset and compare with the report.

python scripts/verify_headline.py --data-dir <snapshot of HF Rockerleo/hold-the-disclosure>

DR@5 (rule checker) and HD@5 (kept and LLM panel majority helpful) per system, as counts of 120. Exits 1 on any
mismatch. No GPU and no API key: the transcripts and panel labels are read from the snapshot.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_gallery import SYSTEMS, build  # noqa: E402
from htd.helpful import helpful_votes  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from render_robustness import system_votes  # noqa: E402
from rule_baseline import rule_transcript  # noqa: E402

# (DR@5, HD@5) out of 120, as reported in docs/REPORT.md and docs/RESULTS_ROBUSTNESS.md (sections 9 and 11).
EXPECTED = {"base": (20, 5), "base_strict": (48, 12), "sft": (63, 33), "dpo": (62, 20), "qwen_strict": (42, 37),
            "rule": (120, 66)}


def rule_counts(data_dir: Path) -> tuple[int, int]:
    """Rebuild the rule baseline from the base transcripts and score it with its published panel labels."""
    sheet = load_sheet()
    rd = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
    ts = [rule_transcript(t, sheet) for t in load_transcripts(data_dir / "followup/eval/base_test2.jsonl")]
    published = {t.dialogue_id: t.replies for t in load_transcripts(data_dir / "followup/eval/base_rule_test2.jsonl")}
    if any(published[t.dialogue_id] != t.replies for t in ts):
        raise SystemExit("rebuilt rule transcripts differ from the published ones")
    votes = system_votes(ts, helpful_votes([rd(data_dir / f"followup/judges_rule/helpful_{n}.jsonl")
                                            for n in ("llama", "deepseek", "gptoss")]))
    kept = {t.dialogue_id: check_reply(t.replies[4], sheet.product(t.product_id), sheet, t.script).all_present
            for t in ts if sheet.product(t.product_id).requires_disclosure}
    return sum(kept.values()), sum(k and votes.get(i, False) for i, k in kept.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    a = ap.parse_args()
    data = build(a.data_dir)
    got = {}
    for key, _, _ in SYSTEMS:
        runs = [d["runs"][key] for d in data["dialogues"] if key in d["runs"]]
        got[key] = (sum(r["kept"][4] for r in runs), sum(bool(r["kept"][4] and r["helpful5"]) for r in runs))
    got["rule"] = rule_counts(a.data_dir)
    labels = dict((k, lab) for k, lab, _ in SYSTEMS) | {"rule": "Rule baseline (post hoc)"}
    bad = 0
    print(f"{'system':32} {'DR@5':>9} {'HD@5':>9}  check")
    for key, (dr, hd) in got.items():
        ok = EXPECTED[key] == (dr, hd)
        bad += not ok
        print(f"{labels[key]:32} {dr:>4}/120 {hd:>4}/120  {'OK' if ok else f'MISMATCH, report says {EXPECTED[key]}'}")
    if bad:
        raise SystemExit(f"{bad} systems do not match the report")
    print("all headline numbers match")


if __name__ == "__main__":
    main()
