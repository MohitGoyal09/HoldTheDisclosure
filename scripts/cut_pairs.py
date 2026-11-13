"""Cut generated dialogues into DPO and SFT rows.

python scripts/cut_pairs.py data/generated/gen_*.jsonl --out-dir data/generated/pairs/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.pairs import cut_pairs  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generated", nargs="+", help="generated JSONL files (output of generate_data.py)")
    ap.add_argument("--out-dir", default="data/generated/pairs")
    ap.add_argument("--recovery-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--always-recover-turn", type=int, default=None, help="add a recovery pair at this turn for every dialogue")
    a = ap.parse_args()

    records = []
    for path in a.generated:
        records += [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    skipped = sum(r["status"] != "ok" for r in records)
    out = cut_pairs(
        records, load_sheet(), a.recovery_frac, a.val_frac, a.seed, always_recover_turn=a.always_recover_turn
    )

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        (out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"{name}: {len(rows)}")
    rec = sum(r["recovery"] for r in out["dpo_train"] + out["dpo_val"])
    print(f"recovery DPO rows: {rec}; failed dialogues skipped: {skipped}; written to {out_dir}")


if __name__ == "__main__":
    main()
