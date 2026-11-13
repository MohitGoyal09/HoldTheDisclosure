"""Build 5-turn pushback dialogues, matched across scripts."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from htd.seeds import DEFAULT_SEEDS_PATH, Seed, load_seeds
from htd.sheet import Sheet

DEFAULT_SLOTS = ("t2_brevity", "t3_social_proof", "t4_urgency", "t5_override")

_DISCLOSURE_TRAIN = tuple((slot, "train") for slot in DEFAULT_SLOTS)
_DISCLOSURE_TEST = tuple((slot, "test") for slot in DEFAULT_SLOTS)
_DISCLOSURE_TEST2 = tuple((slot, "test2") for slot in DEFAULT_SLOTS)
_CONTROL_SLOTS = ("t2_brevity", "c3_impatience", "t4_urgency", "c5_just_answer")

# split -> (product split, ((slot, seed split) for turns 2 to 5)). See docs/DESIGN.md D2, D3.
SPLIT_PRESETS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "pilot": ("train", _DISCLOSURE_TRAIN),
    "train": ("train", _DISCLOSURE_TRAIN),
    "test": ("train", _DISCLOSURE_TEST),
    "heldout": (
        "heldout",
        (
            ("heldout_anger", "heldout"),
            ("heldout_authority", "heldout"),
            ("heldout_anger", "heldout"),
            ("heldout_authority", "heldout"),
        ),
    ),
    "control_train": ("control", tuple((slot, "train") for slot in _CONTROL_SLOTS)),
    "control_test": ("control", tuple((slot, "test") for slot in _CONTROL_SLOTS)),
    # Replication on new pushback lines (docs/PREREGISTRATION.md, follow-up F1, F2).
    "test2": ("train", _DISCLOSURE_TEST2),
    "test2_shuffled": ("train", _DISCLOSURE_TEST2),
}
# Splits that use the last product opening, so turn 1 never matches a training dialogue.
_EVAL_SPLITS = {"test", "heldout", "control_test", "test2", "test2_shuffled"}
# Williams square for 4 tactics: each tactic once per position, each ordered pair once as neighbours.
WILLIAMS_4 = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def williams_row(i: int, n_products: int) -> int:
    """Williams row for dialogue i. Products go round-robin, so i % 4 alone would tie each row to one
    product when there are 4 products; shifting by i // n_products spreads every row over every product."""
    return (i + i // n_products) % len(WILLIAMS_4)


@dataclass(frozen=True)
class Dialogue:
    dialogue_id: str
    split: str
    product_id: str
    script: str
    user_turns: tuple[str, ...]
    tactic_ids: tuple[str, ...]


def build_dialogues(
    sheet: Sheet,
    seeds: dict[str, tuple[Seed, ...]],
    product_ids: list[str] | tuple[str, ...],
    scripts: tuple[str, ...] = ("en", "hi", "hinglish"),
    n_per_script: int = 20,
    rng_seed: int = 0,
    slots: tuple[str, ...] = DEFAULT_SLOTS,
    split_name: str = "pilot",
) -> list[Dialogue]:
    if not product_ids:
        raise ValueError("product_ids is empty")
    for slot in slots:
        if not seeds.get(slot):
            raise ValueError(f"no seeds for slot {slot!r}")
    products = [sheet.product(pid) for pid in product_ids]
    rng = random.Random(rng_seed)
    # One plan per index, shared by every script, so scripts stay matched.
    plans = []
    for i in range(n_per_script):
        product = products[i % len(products)]
        # Train uses the first opening, test and held-out use the last, so turn 1 is
        # never the same string on both sides of the split.
        if split_name in ("test", "heldout", "control_test"):
            opening_idx = len(product.openings[scripts[0]]) - 1
        else:
            opening_idx = 0
        chosen = tuple(rng.choice(seeds[slot]) for slot in slots)
        plans.append((product, opening_idx, chosen))
    return _render(plans, scripts, split_name)


def build_split(
    sheet: Sheet,
    split: str,
    n_per_script: int,
    rng_seed: int = 0,
    scripts: tuple[str, ...] = ("en", "hi", "hinglish"),
    seeds_path: Path | str | None = None,
    product_ids: list[str] | tuple[str, ...] | None = None,
) -> list[Dialogue]:
    """Dialogues for one named split, matched across scripts, without leakage across seed splits.

    Products go round-robin. Seed combinations are drawn without repeats until every
    combination has been used, then the cycle restarts.
    """
    if split not in SPLIT_PRESETS:
        raise ValueError(f"unknown split {split!r}, expected one of {tuple(SPLIT_PRESETS)}")
    product_split, slot_plan = SPLIT_PRESETS[split]
    products = sheet.by_split(product_split)
    if product_ids is not None:
        unknown = set(product_ids) - {p.id for p in products}
        if unknown:
            raise ValueError(f"products {sorted(unknown)} are not in the {split!r} preset")
        products = tuple(p for p in products if p.id in set(product_ids))
    if not products:
        raise ValueError("no products for this split")
    path = DEFAULT_SEEDS_PATH if seeds_path is None else seeds_path
    seeds_by_split = {sp: load_seeds(path, split=sp) for sp in {sp for _, sp in slot_plan}}
    pools = []
    for slot, seed_split in slot_plan:
        pool = seeds_by_split[seed_split].get(slot)
        if not pool:
            raise ValueError(f"no {seed_split} seeds for slot {slot!r}")
        pools.append(pool)
    rng = random.Random(rng_seed)
    combos: list[tuple[Seed, ...]] = []
    plans = []
    for i in range(n_per_script):
        product = products[i % len(products)]
        # Train uses the first opening, test and held-out use the last, so turn 1 is
        # never the same string on both sides of the split.
        if split in _EVAL_SPLITS:
            opening_idx = len(product.openings[scripts[0]]) - 1
        else:
            opening_idx = 0
        if not combos:
            combos = list(itertools.product(*pools))
            rng.shuffle(combos)
        chosen = combos.pop()
        if split.endswith("_shuffled"):
            chosen = tuple(chosen[j] for j in WILLIAMS_4[williams_row(i, len(products))])
        plans.append((product, opening_idx, chosen))
    return _render(plans, scripts, split)


def _render(plans, scripts: tuple[str, ...], split_name: str) -> list[Dialogue]:
    out = []
    for script in scripts:
        for i, (product, opening_idx, chosen) in enumerate(plans):
            turns = (product.openings[script][opening_idx], *(s.text[script] for s in chosen))
            out.append(
                Dialogue(
                    dialogue_id=f"{split_name}-{i:03d}-{script}",
                    split=split_name,
                    product_id=product.id,
                    script=script,
                    user_turns=turns,
                    tactic_ids=tuple(s.id for s in chosen),
                )
            )
    return out


def save_dialogues(dialogues: list[Dialogue], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in dialogues), encoding="utf-8")


def load_dialogues(path: Path | str) -> list[Dialogue]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["user_turns"] = tuple(r["user_turns"])
            r["tactic_ids"] = tuple(r["tactic_ids"])
            out.append(Dialogue(**r))
    return out
