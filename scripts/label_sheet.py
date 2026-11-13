"""Make a human label sheet from judged replies, and score agreement afterwards.

python scripts/label_sheet.py make outputs/judge/turn5.jsonl --out outputs/labels/sheet.csv --n 100
python scripts/label_sheet.py score outputs/labels/sheet.csv --out outputs/labels/agreement.json

Fill the `human_label` column with PASS, DROP, DISMISS or OVER-CORRECT, then run score.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.agreement import report  # noqa: E402
from htd.labels import item_id, pair_labels, stratified_sample  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402

# Blind sheet: no system name, no judge label. `item` is a hash of the judged row id.
FIELDS = [
    "item",
    "script",
    "turn",
    "product_id",
    "required_disclosures",
    "customer_message",
    "assistant_reply",
    "human_label",
    "human_notes",
]


def make(a) -> None:
    sheet = load_sheet()
    rows = [json.loads(line) for line in Path(a.judged).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if "row_id" not in r:
            r["row_id"] = r["dialogue_id"]
        r["item"] = item_id(r["row_id"])
        product = sheet.product(r["product_id"])
        r["required_disclosures"] = " | ".join(d.statement["en"] for d in product.disclosures)
        r["customer_message"] = r.get("customer", "")
        r["assistant_reply"] = r.get("reply", "")
    # Strata do not use any judge's label, so the sheet can score several judges fairly.
    sample = stratified_sample(rows, a.n, seed=a.seed, strata=("script", "model"))
    sample = sorted(sample, key=lambda r: r["item"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({**r, "human_label": "", "human_notes": ""})
    print(json.dumps({"sampled": len(sample), "sheet": str(out)}, indent=2))


def score(a) -> None:
    with Path(a.sheet).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    judged = [json.loads(x) for x in Path(a.judged).read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs = pair_labels(rows, judged)
    result = report(pairs)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("judged")
    m.add_argument("--out", default="outputs/labels/sheet.csv")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("sheet")
    s.add_argument("--judged", required=True, help="judge_run.py output; labels are joined by row_id")
    s.add_argument("--out", default="outputs/labels/agreement.json")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
