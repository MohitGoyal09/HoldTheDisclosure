"""Score a trained recipe, append it to experiments/results.jsonl, update state.json.

python scripts/score_experiment.py --candidate dpo --test outputs/eval/summary_test.json \
  --heldout outputs/eval/summary_heldout.json --control outputs/eval/summary_control.json \
  --comparison outputs/eval/cmp_dpo_test.json --judge outputs/judge/judge_test.jsonl \
  --recipe '{"method":"dpo","beta":0.1,"lr":5e-6,"epochs":1,"rank":16,"n_per_script":40,"recovery_frac":0.25}'
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.experiment import score_recipe  # noqa: E402


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="base")
    ap.add_argument("--candidate", default="dpo")
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--test", required=True)
    ap.add_argument("--heldout")
    ap.add_argument("--control")
    ap.add_argument("--comparison", help="compare.py output for candidate vs base on the test set")
    ap.add_argument("--judge", help="judge_run.py output on the test set, turn 5")
    ap.add_argument("--recipe", default="{}", help="JSON object describing the recipe")
    ap.add_argument("--results", default="experiments/results.jsonl")
    ap.add_argument("--state", default="experiments/state.json")
    ap.add_argument("--require-complete", action="store_true", help="refuse to log when any criterion is None")
    a = ap.parse_args()

    cmp = _read(a.comparison)
    mcnemar_p = cmp["significance"]["mcnemar_p"] if cmp else None
    judge_rows = []
    if a.judge and Path(a.judge).exists():
        judge_rows = [json.loads(x) for x in Path(a.judge).read_text(encoding="utf-8").splitlines() if x.strip()]
    elif a.judge:
        print(f"warning: judge file missing: {a.judge}")

    result = score_recipe(
        base=a.base,
        candidate=a.candidate,
        test=_read(a.test),
        heldout=_read(a.heldout),
        control=_read(a.control),
        judge_rows=judge_rows,
        mcnemar_p=mcnemar_p,
        variant=a.variant,
    )
    if a.require_complete and any(v is None for v in result["criteria"].values()):
        print(json.dumps(result, indent=2))
        raise SystemExit("incomplete criteria; not writing a result row")
    recipe = json.loads(a.recipe)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "recipe": recipe, **result}

    state_path = Path(a.state)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"best_validation_score": -1, "run_number": 0}
    state["run_number"] = state.get("run_number", 0) + 1
    keep = result["passed"] > state.get("best_validation_score", -1) and not result["c1_failed"]
    if keep:
        state["best_validation_score"] = result["passed"]
        state["best_recipe"] = recipe
        state["plateau_counter"] = 0
        state["best_run"] = state["run_number"]
    else:
        state["plateau_counter"] = state.get("plateau_counter", 0) + 1
    entry["run"] = state["run_number"]
    entry["decision"] = "keep" if keep else "discard"

    results_path = Path(a.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(json.dumps(entry, indent=2))
    print(f"\nRUN {entry['run']} | {result['passed']}/{result['max_score']} | {entry['decision']}")


if __name__ == "__main__":
    main()
