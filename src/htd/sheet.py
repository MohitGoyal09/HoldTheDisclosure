"""Load and validate the product sheet (data/product_sheet.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS: tuple[str, ...] = ("en", "hi", "hinglish")
SPLITS: tuple[str, ...] = ("train", "heldout", "control")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHEET_PATH = REPO_ROOT / "data" / "product_sheet.json"


class SheetError(ValueError):
    """The product sheet is malformed."""


@dataclass(frozen=True)
class KeyFactGroup:
    """One fact that must appear. Matches if any pattern (any script) appears."""

    patterns: dict[str, tuple[str, ...]]

    def all_patterns(self) -> tuple[str, ...]:
        return tuple(p for script in SCRIPTS for p in self.patterns.get(script, ()))


@dataclass(frozen=True)
class Disclosure:
    id: str
    statement: dict[str, str]
    groups: tuple[KeyFactGroup, ...]


@dataclass(frozen=True)
class Product:
    id: str
    split: str
    requires_disclosure: bool
    name: dict[str, str]
    facts_for_prompt: str
    disclosures: tuple[Disclosure, ...]
    openings: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Sheet:
    bank: str
    fictional_notice: str
    warning_markers: dict[str, tuple[str, ...]]
    products: tuple[Product, ...]
    dismissal_markers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def product(self, product_id: str) -> Product:
        for p in self.products:
            if p.id == product_id:
                return p
        raise KeyError(product_id)

    def by_split(self, split: str) -> tuple[Product, ...]:
        if split not in SPLITS:
            raise SheetError(f"unknown split {split!r}")
        return tuple(p for p in self.products if p.split == split)

    def all_warning_markers(self) -> tuple[str, ...]:
        return tuple(m for s in SCRIPTS for m in self.warning_markers.get(s, ()))

    def all_dismissal_markers(self) -> tuple[str, ...]:
        return tuple(m for s in SCRIPTS for m in self.dismissal_markers.get(s, ()))


def _check_scripts(obj: dict, where: str) -> None:
    keys = set(obj)
    if keys != set(SCRIPTS):
        raise SheetError(f"{where}: script keys must be exactly {SCRIPTS}, got {sorted(keys)}")


def _script_tuple_map(obj: dict, where: str) -> dict[str, tuple[str, ...]]:
    _check_scripts(obj, where)
    return {s: tuple(obj[s]) for s in SCRIPTS}


def _parse_disclosure(raw: dict, where: str) -> Disclosure:
    _check_scripts(raw["statement"], f"{where}.statement")
    groups = tuple(
        KeyFactGroup(_script_tuple_map(g, f"{where}.key_fact_groups[{i}]"))
        for i, g in enumerate(raw["key_fact_groups"])
    )
    if not groups:
        raise SheetError(f"{where}: needs at least one key-fact group")
    return Disclosure(id=raw["id"], statement=dict(raw["statement"]), groups=groups)


def _parse_product(raw: dict) -> Product:
    where = f"product {raw.get('id')!r}"
    if raw["split"] not in SPLITS:
        raise SheetError(f"{where}: unknown split {raw['split']!r}")
    _check_scripts(raw["name"], f"{where}.name")
    disclosures = tuple(
        _parse_disclosure(d, f"{where}.disclosures[{i}]") for i, d in enumerate(raw["disclosures"])
    )
    requires = bool(raw["requires_disclosure"])
    if requires != bool(disclosures):
        raise SheetError(f"{where}: requires_disclosure={requires} but has {len(disclosures)} disclosures")
    if (raw["split"] == "control") == requires:
        raise SheetError(f"{where}: control products must have no disclosures, others must have some")
    return Product(
        id=raw["id"],
        split=raw["split"],
        requires_disclosure=requires,
        name=dict(raw["name"]),
        facts_for_prompt=raw["facts_for_prompt"],
        disclosures=disclosures,
        openings=_script_tuple_map(raw["openings"], f"{where}.openings"),
    )


def load_sheet(path: Path | str = DEFAULT_SHEET_PATH) -> Sheet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    products = tuple(_parse_product(p) for p in raw["products"])
    ids = [p.id for p in products]
    if len(ids) != len(set(ids)):
        raise SheetError(f"duplicate product ids: {ids}")
    return Sheet(
        bank=raw["bank"],
        fictional_notice=raw["fictional_notice"],
        warning_markers=_script_tuple_map(raw["warning_markers"], "warning_markers"),
        products=products,
        dismissal_markers=(
            _script_tuple_map(raw["dismissal_markers"], "dismissal_markers")
            if "dismissal_markers" in raw
            else {}
        ),
    )
