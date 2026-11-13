"""Build pilot dialogues: python scripts/build_pilot.py --split pilot --n-per-script 20 --out data/generated/pilot_dialogues.jsonl"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.dialogues import SPLIT_PRESETS, build_split, save_dialogues  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-script", type=int, default=20)
    ap.add_argument("--split", default="pilot", choices=list(SPLIT_PRESETS))
    ap.add_argument("--products", default=None, help="comma-separated; filters the preset's products")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="data/generated/pilot_dialogues.jsonl")
    a = ap.parse_args()
    ds = build_split(
        load_sheet(),
        a.split,
        a.n_per_script,
        rng_seed=a.rng_seed,
        product_ids=a.products.split(",") if a.products else None,
    )
    save_dialogues(ds, a.out)
    print(f"wrote {len(ds)} dialogues to {a.out}")


if __name__ == "__main__":
    main()
