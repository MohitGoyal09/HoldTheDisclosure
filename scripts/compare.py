"""Paired DR@k comparison of two transcript files:
python scripts/compare.py A.jsonl B.jsonl [--turn 5] [--out out.json]"""

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.audit import compare_systems  # noqa: E402
from htd.report import comparison_markdown  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def to_markdown(r: dict, name_a: str, name_b: str) -> str:
    return comparison_markdown(r, name_a, name_b)


def _label(ts) -> str:
    return "/".join(sorted({f"{t.model_name}:{t.variant}" for t in ts}))


def _by_index(ts):
    """Drop the split name from `split-NNN-script` ids, so the same dialogue in two splits pairs up."""
    return [replace(t, dialogue_id=t.dialogue_id.split("-", 1)[1]) for t in ts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--turn", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--pair-by-index", action="store_true",
                    help="pair across splits by dialogue index and script (test2 against test2_shuffled)")
    args = ap.parse_args()
    ta, tb = load_transcripts(args.a), load_transcripts(args.b)
    if args.pair_by_index:
        ta, tb = _by_index(ta), _by_index(tb)
    result = compare_systems(ta, tb, load_sheet(), turn=args.turn)
    result["system_a"], result["system_b"] = _label(ta), _label(tb)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(to_markdown(result, result["system_a"], result["system_b"]))


if __name__ == "__main__":
    main()
