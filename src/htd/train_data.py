"""Load preference and SFT rows for TRL, keeping only the columns the trainers accept."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def filter_max_turn(rows: list[Row], max_turn: int | None) -> list[Row]:
    """Keep rows up to a turn. Short prompts let DPO fit on a 15 GB GPU."""
    if max_turn is None:
        return rows
    return [r for r in rows if int(r.get("turn", 0)) <= max_turn]


def filter_pair_types(rows: list[Row], pair_types: list[str] | None) -> list[Row]:
    """Keep DPO rows of the given pair types (drop, lecture, warn)."""
    if pair_types is None:
        return rows
    unknown = set(pair_types) - {r.get("pair_type") for r in rows}
    if unknown:
        raise ValueError(f"pair types not in data: {sorted(unknown)}")
    kept = [r for r in rows if r.get("pair_type") in pair_types]
    if not kept:
        raise ValueError(f"no rows of pair types {pair_types}")
    return kept


def load_jsonl(path: Path | str) -> list[Row]:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {p}")
    return rows


def dpo_columns(rows: list[Row]) -> list[Row]:
    """TRL conversational DPO rows: prompt, chosen and rejected message lists only."""
    out = []
    for i, r in enumerate(rows):
        missing = [k for k in ("prompt", "chosen", "rejected") if k not in r]
        if missing:
            raise ValueError(f"row {i} missing {missing}")
        out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return out


def chat_template_pairs(rows: list[Row], tokenizer: Any) -> list[Row]:
    """Render DPO rows as chat-template strings, the way TRL does for conversational rows.

    AlignTune flattens message lists to role-less text before DPO (docs/UPSTREAM.md, item 7);
    plain strings pass through it unchanged, so the trainer sees the same format as evaluation.
    """
    out = []
    for i, r in enumerate(rows):
        prompt = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        new = {**r, "prompt": prompt}
        for k in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(r["prompt"] + r[k], tokenize=False)
            if not full.startswith(prompt):
                raise ValueError(f"row {i}: templated prompt+{k} does not start with the templated prompt")
            new[k] = full[len(prompt):]
        out.append(new)
    return out


def sft_columns(rows: list[Row]) -> list[Row]:
    """TRL SFT rows: one `messages` list only."""
    out = []
    for i, r in enumerate(rows):
        if "messages" not in r:
            raise ValueError(f"row {i} missing ['messages']")
        out.append({"messages": r["messages"]})
    return out


def row_stats(rows: list[Row], column: str = "chosen") -> dict[str, float | int]:
    """Counts and average reply words, for the run log."""
    words = []
    for r in rows:
        value = r.get(column)
        if isinstance(value, list) and value and isinstance(value[-1], dict):
            words.append(len(value[-1].get("content", "").split()))
        elif isinstance(value, str):
            words.append(len(value.split()))
    return {
        "rows": len(rows),
        "replies": len(words),
        "mean_words": round(sum(words) / len(words), 1) if words else 0.0,
        "max_words": max(words) if words else 0,
    }
