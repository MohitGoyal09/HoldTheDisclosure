"""Load pushback seeds (data/seeds/pushback_seeds.jsonl)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from htd.sheet import REPO_ROOT, SCRIPTS

DEFAULT_SEEDS_PATH = REPO_ROOT / "data" / "seeds" / "pushback_seeds.jsonl"
SEED_SPLITS: tuple[str, ...] = ("train", "test", "heldout", "test2")


class SeedError(ValueError):
    """A seed line is malformed."""


@dataclass(frozen=True)
class Seed:
    id: str
    slot: str
    status: str
    text: dict[str, str]
    split: str = "train"


def load_seeds(
    path: Path | str = DEFAULT_SEEDS_PATH, split: str | None = None
) -> dict[str, tuple[Seed, ...]]:
    """Seeds grouped by slot. With `split`, keep only seeds of that split (see docs/DESIGN.md D2)."""
    if split is not None and split not in SEED_SPLITS:
        raise SeedError(f"unknown split {split!r}")
    by_slot: dict[str, list[Seed]] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [s for s in SCRIPTS if not raw.get(s)]
        if missing:
            raise SeedError(f"line {n} ({raw.get('id')!r}): missing script keys {missing}")
        seed_split = raw.get("split", "train")
        if seed_split not in SEED_SPLITS:
            raise SeedError(f"line {n} ({raw.get('id')!r}): unknown split {seed_split!r}")
        if split is not None and seed_split != split:
            continue
        seed = Seed(raw["id"], raw["slot"], raw.get("status", ""), {s: raw[s] for s in SCRIPTS}, seed_split)
        by_slot.setdefault(seed.slot, []).append(seed)
    return {slot: tuple(v) for slot, v in by_slot.items()}
