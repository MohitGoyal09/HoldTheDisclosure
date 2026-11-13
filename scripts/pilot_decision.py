"""Apply the pre-registered pilot go/no-go rule to a summary.json.

python scripts/pilot_decision.py outputs/pilot/summary.json [--out outputs/pilot/decision.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.decide import pilot_go_no_go  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", help="score_run.py output with all three pilot arms")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    decision = pilot_go_no_go(report)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
