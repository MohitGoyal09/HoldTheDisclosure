"""Agreement between the LLM judge and human labels, for docs/DESIGN.md D4."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from htd.judge import LABELS


def agreement(pairs: Iterable[tuple[str, str]]) -> float:
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(a == b for a, b in pairs) / len(pairs)


def cohen_kappa(pairs: Sequence[tuple[str, str]], labels: Sequence[str] = LABELS) -> float:
    """Cohen's kappa. Returns 1.0 when both raters use one label perfectly, 0.0 when empty."""
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return 0.0
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((a_counts[l] / n) * (b_counts[l] / n) for l in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def confusion(pairs: Sequence[tuple[str, str]], labels: Sequence[str] = LABELS) -> dict[str, dict[str, int]]:
    """{human label: {judge label: count}}."""
    out = {h: {j: 0 for j in labels} for h in labels}
    for human, judge in pairs:
        if human in out and judge in out[human]:
            out[human][judge] += 1
    return out


def report(pairs: Sequence[tuple[str, str]]) -> dict:
    pairs = [(a, b) for a, b in pairs if a and b]
    per_label = {}
    for label in LABELS:
        rows = [(a, b) for a, b in pairs if b == label]
        per_label[label] = {
            "n": len(rows),
            "agreement": agreement(rows) if rows else None,
        }
    return {
        "n": len(pairs),
        "agreement": agreement(pairs),
        "cohen_kappa": cohen_kappa(pairs),
        "per_judge_label": per_label,
        "confusion": confusion(pairs),
    }


def majority(labels: Sequence[str]) -> str | None:
    """Label given by more than half of the raters, else None."""
    label, count = Counter(labels).most_common(1)[0]
    return label if count * 2 > len(labels) else None


def fleiss_kappa(items: Sequence[Sequence[str]], labels: Sequence[str] = LABELS) -> float:
    """Fleiss' kappa for items each rated by the same number of raters."""
    n = len(items[0])
    counts = [[Counter(row)[c] for c in labels] for row in items]
    total = len(items) * n
    p_j = [sum(r[j] for r in counts) / total for j in range(len(labels))]
    p_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in counts]
    p_bar, p_e = sum(p_i) / len(items), sum(p * p for p in p_j)
    return 1.0 if p_e == 1 else round((p_bar - p_e) / (1 - p_e), 6)
