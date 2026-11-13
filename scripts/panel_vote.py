"""Majority vote of the LLM reference panel. The result is a reference, not a human label.

python scripts/panel_vote.py outputs/judge_local/panel_llama.jsonl outputs/judge_local/panel_deepseek.jsonl \
  outputs/judge_local/panel_gptoss.jsonl --out outputs/judge_local/panel_reference.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.agreement import fleiss_kappa, majority  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="judge_run.py outputs, one per panel model, rubric panel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    panels = []
    for f in a.files:
        rows = [json.loads(x) for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
        if any(r.get("judge_rubric") != "panel" for r in rows):
            raise SystemExit(f"{f}: every row must use rubric 'panel'")
        panels.append({r["dialogue_id"]: r for r in rows if r["judge_label"] != "ERROR"})
    models = [next(iter(p.values()))["judge_model"] for p in panels if p]
    if len(set(models)) != len(panels):
        raise SystemExit(f"panel files must come from different models, got {models}")
    common = sorted(set.intersection(*(set(p) for p in panels)))
    lost = {m: len(p) - len(common) for m, p in zip(models, panels)}
    if not common:
        raise SystemExit("no row was labelled by every panel model")

    out, ratings, no_majority = [], [], 0
    for rid in common:
        labels = [p[rid]["judge_label"] for p in panels]
        ratings.append(labels)
        ref = majority(labels)
        if ref is None:
            no_majority += 1
            continue
        out.append({**panels[0][rid], "judge_label": ref, "judge_reason": "panel majority",
                    "judge_model": "panel:" + ",".join(p[rid]["judge_model"] for p in panels),
                    "panel_labels": labels})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"rows_all_models": len(common), "rows_lost_per_model": lost, "reference_rows": len(out), "no_majority": no_majority,
                      "fleiss_kappa": fleiss_kappa(ratings)}, indent=2))


if __name__ == "__main__":
    main()
