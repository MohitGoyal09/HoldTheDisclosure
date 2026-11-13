"""Deterministic stratified sampling for the human label sheet."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Sequence


def _key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def stratified_sample(
    rows: Sequence[dict[str, Any]],
    n: int,
    seed: int = 0,
    strata: Sequence[str] = ("script", "judge_label"),
    id_field: str = "row_id",
) -> list[dict[str, Any]]:
    """Take up to n rows, round-robin across strata, so every stratum is represented.

    Order inside a stratum is deterministic from `seed` and `id_field`, so the sheet can be
    regenerated exactly. Strata are visited in sorted order.
    """
    if n <= 0 or not rows:
        return []
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in strata)].append(row)
    ordered = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        ordered.append(sorted(groups[key], key=lambda r: _key(seed, str(r.get(id_field, "")))))
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < n and any(depth < len(g) for g in ordered):
        for g in ordered:
            if depth < len(g) and len(out) < n:
                out.append(g[depth])
        depth += 1
    return out


def item_id(row_id: str) -> str:
    """Opaque, stable id for a human label sheet, so the labeller cannot see the system."""
    return hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:10]


def _norm(label: str) -> str:
    return (label or "").strip().upper().replace("_", "-")


def pair_labels(sheet_rows: Sequence[dict[str, Any]], judged_rows: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """(human, judge) label pairs, joined on the opaque item id. Blank human labels are skipped."""
    judged = {
        item_id(r.get("dialogue_id", r.get("row_id", ""))): _norm(r.get("judge_label", ""))
        for r in judged_rows
        if _norm(r.get("judge_label", "")) != "ERROR"
    }
    return [
        (_norm(r.get("human_label", "")), judged[r["item"]])
        for r in sheet_rows
        if _norm(r.get("human_label", "")) and r.get("item") in judged
    ]


_YES, _NO = {"YES", "Y", "1", "TRUE"}, {"NO", "N", "0", "FALSE"}


def yes_no(value: str) -> str:
    """Normalise a yes/no cell to YES, NO or blank."""
    v = (value or "").strip().upper()
    if not v:
        return ""
    if v in _YES:
        return "YES"
    if v in _NO:
        return "NO"
    raise ValueError(f"expected yes or no, got {value!r}")


def human_column(sheet_rows: Sequence[dict[str, Any]], field: str) -> dict[str, str]:
    """{item: normalised value} for one labeller, blank cells skipped."""
    norm = yes_no if field == "states_all" else _norm
    out = {r["item"]: norm(r.get(field, "")) for r in sheet_rows}
    return {k: v for k, v in out.items() if v}


def consensus(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """Items both labellers filled with the same value."""
    return {k: v for k, v in a.items() if b.get(k) == v}
