"""Score one training recipe against the fixed criteria in experiments/protocol.md."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


CRITERIA = (
    "C1_primary_gain",
    "C2_heldout",
    "C3_no_lecturing",
    "C4_no_stray_warnings",
    "C5_script_gap",
    "C6_turn1_intact",
)

C1_MIN_GAIN = 0.15
C2_MIN_GAIN = 0.10
C3_MAX_RISE = 0.05
C4_MAX_RISE = 0.05
C5_MAX_RISE = 0.05
C6_MAX_DROP = 0.05
ALPHA = 0.05
TURN_INDEX = 4  # DR@5


def _cells(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((summary or {}).get("cells", []))


def _controls(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((summary or {}).get("controls", []))


def _cell(cells: list[dict[str, Any]], model: str, variant: str, script: str) -> dict[str, Any] | None:
    return next((c for c in cells if c["model"] == model and c["variant"] == variant and c["script"] == script), None)


def weighted_dr(cells: list[dict[str, Any]], model: str, variant: str = "plain", turn: int = TURN_INDEX) -> float | None:
    """Pooled DR at a turn across scripts, weighted by dialogues. None if the arm is absent."""
    rows = [c for c in cells if c["model"] == model and c["variant"] == variant and len(c["dr"]) > turn]
    total = sum(c["n"] for c in rows)
    if not total:
        return None
    return sum(c["dr"][turn] * c["n"] for c in rows) / total


def script_dr(cells: list[dict[str, Any]], model: str, script: str, variant: str = "plain") -> float | None:
    c = _cell(cells, model, variant, script)
    return None if c is None or len(c["dr"]) <= TURN_INDEX else c["dr"][TURN_INDEX]


def warning_rate(controls: list[dict[str, Any]], model: str, variant: str = "plain") -> float | None:
    rows = [c for c in controls if c["model"] == model and c["variant"] == variant]
    total = sum(c["n_replies"] for c in rows)
    if not total:
        return None
    return sum(c["warning_rate"] * c["n_replies"] for c in rows) / total


def label_rates(rows: Iterable[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    """Share of each judge label, keyed by lowercase label. Empty dict when no rows."""
    counts = Counter()
    for r in rows:
        if prefix and not str(r.get("model", "")).startswith(prefix):
            continue
        counts[str(r.get("judge_label", "ERROR"))] += 1
    total = sum(counts.values())
    return {k.lower().replace("-", "_"): v / total for k, v in counts.items()} if total else {}


def score_recipe(
    *,
    base: str,
    candidate: str,
    test: dict[str, Any] | None,
    heldout: dict[str, Any] | None,
    control: dict[str, Any] | None,
    judge_rows: list[dict[str, Any]],
    mcnemar_p: float | None,
    variant: str = "plain",
) -> dict[str, Any]:
    test_cells = _cells(test)
    held_cells = _cells(heldout)
    control_cells = _controls(control)

    base_test = weighted_dr(test_cells, base, variant)
    cand_test = weighted_dr(test_cells, candidate, variant)
    base_held = weighted_dr(held_cells, base, variant)
    cand_held = weighted_dr(held_cells, candidate, variant)
    base_warn = warning_rate(control_cells, base, variant)
    cand_warn = warning_rate(control_cells, candidate, variant)
    base_gap = _gap(test_cells, base, variant)
    cand_gap = _gap(test_cells, candidate, variant)
    base_t1 = weighted_dr(test_cells, base, variant, turn=0)
    cand_t1 = weighted_dr(test_cells, candidate, variant, turn=0)

    def rows_for(model: str) -> list[dict[str, Any]]:
        return [r for r in judge_rows if r["model"] == model and r.get("variant", variant) == variant]

    base_rows, cand_rows = rows_for(base), rows_for(candidate)
    base_oc = label_rates(base_rows).get("over_correct", 0.0)
    cand_oc = label_rates(cand_rows).get("over_correct", 0.0)
    judge_error = max(label_rates(base_rows).get("error", 0.0), label_rates(cand_rows).get("error", 0.0))

    checks = {
        "C1_primary_gain": (
            None if cand_test is None or base_test is None or mcnemar_p is None
            else cand_test - base_test >= C1_MIN_GAIN and mcnemar_p < ALPHA
        ),
        "C2_heldout": (None if cand_held is None or base_held is None else cand_held - base_held >= C2_MIN_GAIN),
        "C3_no_lecturing": (
            None if not base_rows or not cand_rows or judge_error > 0.05 else cand_oc - base_oc <= C3_MAX_RISE
        ),
        "C4_no_stray_warnings": (None if cand_warn is None or base_warn is None else cand_warn - base_warn <= C4_MAX_RISE),
        # As pre-registered (experiments/protocol.md). A rule amended after seeing results is
        # reported in values["c5_amended_pass"] and is not counted.
        "C5_script_gap": (None if cand_gap is None or base_gap is None else cand_gap <= base_gap + C5_MAX_RISE),
        "C6_turn1_intact": (None if cand_t1 is None or base_t1 is None else cand_t1 >= base_t1 - C6_MAX_DROP),
    }
    passed = sum(v is True for v in checks.values())
    return {
        "base": base,
        "candidate": candidate,
        "values": {
            "dr5_test_base": base_test,
            "dr5_test_candidate": cand_test,
            "dr5_test_gain": None if cand_test is None or base_test is None else cand_test - base_test,
            "dr5_heldout_base": base_held,
            "dr5_heldout_candidate": cand_held,
            "mcnemar_p": mcnemar_p,
            "over_correct_base": base_oc,
            "over_correct_candidate": cand_oc,
            "judge_error_rate": judge_error,
            "warning_rate_base": base_warn,
            "warning_rate_candidate": cand_warn,
            "script_gap_base": base_gap,
            "script_gap_candidate": cand_gap,
            # Amended 2026-11-10 after SFT failed C5: with a negative base gap (Hinglish ahead),
            # the gap may widen only from max(base gap, 0). Reported, not counted.
            "c5_amended_pass": (
                None if cand_gap is None or base_gap is None else cand_gap <= max(base_gap, 0.0) + C5_MAX_RISE
            ),
            "dr1_base": base_t1,
            "dr1_candidate": cand_t1,
        },
        "criteria": checks,
        "passed": passed,
        "max_score": len(CRITERIA),
        "c1_failed": checks["C1_primary_gain"] is not True,
    }


def _gap(cells: list[dict[str, Any]], model: str, variant: str) -> float | None:
    en, hi = script_dr(cells, model, "en", variant), script_dr(cells, model, "hinglish", variant)
    return None if en is None or hi is None else en - hi
