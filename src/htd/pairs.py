"""Cut generated dialogues into turn-level DPO and SFT rows (TRL conversational format)."""

from __future__ import annotations

import hashlib
from typing import Any

from htd.prompts import system_prompt
from htd.sheet import SCRIPTS, Sheet

TURNS = 5
MIN_RECOVERY_TURN = 3
Row = dict[str, Any]


def base_index(dialogue_id: str) -> str:
    """Dialogue id without the trailing script suffix (all scripts of one index share it)."""
    head, _, tail = dialogue_id.rpartition("-")
    return head if tail in SCRIPTS and head else dialogue_id


def _digest(seed: int, *parts: str) -> int:
    return int(hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest(), 16)


def _pick_count(n: int, frac: float, at_least_one: bool = False) -> int:
    if n == 0 or frac <= 0:
        return 0
    k = round(frac * n)
    return max(1, k) if at_least_one else k


def _val_bases(bases: list[str], val_frac: float, seed: int) -> set[str]:
    k = _pick_count(len(bases), val_frac, at_least_one=len(bases) >= 2)
    return set(sorted(bases, key=lambda b: _digest(seed, "val", b))[:k])


def _recovery_plan(records: list[dict], frac: float, seed: int) -> dict[str, int]:
    """{dialogue_id: turn k} for the dialogues that get a recovery history."""
    eligible = sorted(r["dialogue_id"] for r in records if r.get("requires_disclosure") and r.get("drop"))
    k = _pick_count(len(eligible), frac)
    ranked = sorted(eligible, key=lambda d: _digest(seed, "rec", d))[:k]
    span = TURNS - MIN_RECOVERY_TURN + 1
    return {d: MIN_RECOVERY_TURN + _digest(seed, "turn", d) % span for d in ranked}


def _prompt(system: str, rec: dict, turn: int, recovery: bool) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    for i in range(turn - 1):
        reply = rec["drop"][i] if recovery and i == turn - 2 else rec["chosen"][i]
        msgs.append({"role": "user", "content": rec["user_turns"][i]})
        msgs.append({"role": "assistant", "content": reply})
    msgs.append({"role": "user", "content": rec["user_turns"][turn - 1]})
    return msgs


def _meta(rec: dict, turn: int, recovery: bool) -> Row:
    return {
        "dialogue_id": rec["dialogue_id"],
        "turn": turn,
        "script": rec["script"],
        "product_id": rec["product_id"],
        "recovery": recovery,
    }


def _rows_for(
    rec: dict, sheet: Sheet, recovery_turn: int | None, extra_turn: int | None = None
) -> tuple[list[Row], list[Row]]:
    system = system_prompt(sheet.product(rec["product_id"]), sheet, "plain", rec["script"])
    rejected_types = ("drop", "lecture") if rec["requires_disclosure"] else ("warn",)
    dpo, sft = [], []
    for turn in range(1, TURNS + 1):
        modes = [recovery_turn == turn]
        if extra_turn == turn and recovery_turn != turn:
            modes.append(True)
        for recovery in modes:
            prompt = _prompt(system, rec, turn, recovery)
            chosen = [{"role": "assistant", "content": rec["chosen"][turn - 1]}]
            sft.append({"messages": prompt + chosen, **_meta(rec, turn, recovery)})
            for pair_type in rejected_types:
                rejected = [{"role": "assistant", "content": rec[pair_type][turn - 1]}]
                dpo.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "pair_type": pair_type,
                        **_meta(rec, turn, recovery),
                    }
                )
    return dpo, sft


def cut_pairs(
    generated: list[dict],
    sheet: Sheet,
    recovery_frac: float = 0.25,
    val_frac: float = 0.10,
    seed: int = 0,
    always_recover_turn: int | None = None,
) -> dict[str, list[Row]]:
    """Turn generated dialogue records into dpo_train, dpo_val, sft_train and sft_val rows."""
    ok = sorted((r for r in generated if r.get("status") == "ok"), key=lambda r: r["dialogue_id"])
    val = _val_bases(sorted({base_index(r["dialogue_id"]) for r in ok}), val_frac, seed)
    recovery = _recovery_plan(ok, recovery_frac, seed)
    out: dict[str, list[Row]] = {"dpo_train": [], "dpo_val": [], "sft_train": [], "sft_val": []}
    for rec in ok:
        extra = always_recover_turn if rec["requires_disclosure"] else None
        dpo, sft = _rows_for(rec, sheet, recovery.get(rec["dialogue_id"]), extra)
        suffix = "val" if base_index(rec["dialogue_id"]) in val else "train"
        out[f"dpo_{suffix}"].extend(dpo)
        out[f"sft_{suffix}"].extend(sft)
    return out


def pair_profile(rows: list[dict], sheet: Sheet) -> dict[str, dict[str, float]]:
    """Per pair type: mean words, mean warning markers and checker passes, chosen against rejected.

    Shows which pairs push the model away from warning wording.
    """
    from htd.rule_checker import check_reply

    def last(msgs: Any) -> str:
        return msgs[-1]["content"] if isinstance(msgs, list) else str(msgs)

    acc: dict[str, dict[str, float]] = {}
    for r in rows:
        product = sheet.product(r["product_id"])
        a = acc.setdefault(r["pair_type"], {"n": 0})
        a["n"] += 1
        for side in ("chosen", "rejected"):
            text = last(r[side])
            check = check_reply(text, product, sheet, r["script"])
            a[f"words_{side}"] = a.get(f"words_{side}", 0) + len(text.split())
            a[f"markers_{side}"] = a.get(f"markers_{side}", 0) + len(check.warning_markers_found)
            a[f"checker_pass_{side}"] = a.get(f"checker_pass_{side}", 0) + (product.requires_disclosure and check.all_present)
    for a in acc.values():
        for k in ("words_chosen", "words_rejected", "markers_chosen", "markers_rejected"):
            a[k] = round(a[k] / a["n"], 2)
    return acc
