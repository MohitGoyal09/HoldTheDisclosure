"""AuditKIT integration: model backends for multi-turn replay and paired significance.

AuditKIT is imported lazily, so the rest of `htd` runs without it.

How it plugs in:
- `auditkit_generate_batch` resolves a spec like "hf:CohereLabs/tiny-aya-fire" with
  `auditkit.model.AutoModel.resolve`, then sends each conversation as one chat
  `Request` whose `params["messages"]` holds the FULL message list.
- `compare_systems` runs AuditKIT's `paired_bootstrap` (the test behind
  `ModelComparison.significance`) on per-dialogue 0/1 flags.
- `register_dr_metric` registers a "disclosure_retention" Metric in AuditKIT's registry.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from htd.backends import REPLY_END_MARKERS
from htd.replay import Transcript
from htd.sheet import Sheet

Messages = list[dict[str, str]]
DR_METRIC_NAME = "disclosure_retention"


def _flatten(messages: Messages) -> str:
    """Plain-text fallback for backends that only read `request.prompt`."""
    return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def _trim_at_markers(text: str) -> str:
    for marker in REPLY_END_MARKERS:
        text = text.split(marker, 1)[0]
    return text.strip()


def auditkit_generate_batch(
    model_spec: str,
    max_tokens: int = 300,
    temperature: float = 0.0,
    stop_sequences: Sequence[str] | None = None,
    **backend_kwargs: Any,
) -> Callable[[list[Messages]], list[str]]:
    """Wrap an AuditKIT model backend as a `generate_batch` for `htd.replay.replay`.

    Each conversation becomes a chat Request with `params["messages"]` set to the full
    history. Every AuditKIT chat backend reads it through `resolve_messages`; the HF
    backend renders it with the tokenizer's chat template.

    Stop tokens: AuditKIT's HF backend does not stop on Tiny Aya's reply-end markers
    (it relies on generation_config eos). For `hf:` specs we pass them through the
    public `stop_sequences` param (mapped to transformers `stop_strings`) and also trim
    any marker from the output. Other backends get no stop sequences unless given.
    """
    from auditkit.model import AutoModel, Request  # lazy: heavy package

    model = AutoModel.resolve(model_spec, **backend_kwargs)
    if stop_sequences is None:
        stop_sequences = list(REPLY_END_MARKERS) if model_spec.startswith("hf:") else []

    def generate_batch(messages_list: list[Messages]) -> list[str]:
        if not messages_list:
            return []
        params: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if stop_sequences:
            params["stop_sequences"] = list(stop_sequences)
        requests = [
            Request(
                prompt=_flatten(m),
                request_type="chat",
                params={**params, "messages": [dict(x) for x in m]},
            )
            for m in messages_list
        ]
        results = model.generate(requests)
        if len(results) != len(requests):
            raise ValueError(f"AuditKIT backend returned {len(results)} results for {len(requests)} requests")
        return [_trim_at_markers(r.text) for r in results]

    generate_batch.model = model  # type: ignore[attr-defined]
    return generate_batch


def dialogue_flags(transcripts: Sequence[Transcript], sheet: Sheet, turn: int = 5) -> dict[str, int]:
    """dialogue_id -> 1 if the reply at `turn` (1-based) states every disclosure, else 0.

    Only products with `requires_disclosure`. Dialogues with fewer than `turn` replies count as 0.
    """
    from htd.rule_checker import check_reply

    flags: dict[str, int] = {}
    for t in transcripts:
        product = sheet.product(t.product_id)
        if not product.requires_disclosure:
            continue
        if t.dialogue_id in flags:
            raise ValueError(f"duplicate dialogue_id {t.dialogue_id!r}")
        ok = len(t.replies) >= turn and check_reply(t.replies[turn - 1], product, sheet).all_present
        flags[t.dialogue_id] = int(ok)
    return flags


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def mcnemar_exact_p(pairs: Sequence[tuple[float, float]]) -> float:
    """Two-sided exact McNemar p-value for paired 0/1 outcomes (binomial on discordant pairs)."""
    only_b = sum(1 for a, b in pairs if a < b)
    only_a = sum(1 for a, b in pairs if a > b)
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_diff_ci(pairs: Sequence[tuple[float, float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI for mean(b - a)."""
    diffs = [b - a for a, b in pairs]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def compare_systems(
    transcripts_a: Sequence[Transcript],
    transcripts_b: Sequence[Transcript],
    sheet: Sheet,
    turn: int = 5,
) -> dict:
    """Paired comparison of DR@turn between system A (baseline) and B (candidate).

    `diff` = mean_b - mean_a. `significance` holds AuditKIT's paired bootstrap result
    (`p_value`, `significant`, alpha 0.05; it returns no CI) AND our own checks:
    exact McNemar (`mcnemar_p`, `mcnemar_significant`) and a percentile bootstrap CI.

    Caveat: AuditKIT 1.1.3 `paired_bootstrap` resamples the observed differences without
    centering them under the null, so p is about 0.6 even for 0% vs 90%. Treat its flag
    as unreliable; use `mcnemar_p` for decisions.
    """
    try:
        from auditkit._bootstrap import paired_bootstrap
    except ImportError:  # core comparison must work without the optional package
        paired_bootstrap = None

    fa, fb = dialogue_flags(transcripts_a, sheet, turn), dialogue_flags(transcripts_b, sheet, turn)
    if fa.keys() != fb.keys():
        only_a, only_b = sorted(fa.keys() - fb.keys()), sorted(fb.keys() - fa.keys())
        raise ValueError(
            f"dialogue ids differ between systems: {len(only_a)} only in A {only_a[:3]}, "
            f"{len(only_b)} only in B {only_b[:3]}. Run both on the same dialogues file."
        )
    ids = sorted(fa)
    pairs = [(float(fa[i]), float(fb[i])) for i in ids]
    mean_a, mean_b = _mean([p[0] for p in pairs]), _mean([p[1] for p in pairs])
    if paired_bootstrap is None:
        boot = {"error": "auditkit not installed; McNemar and htd CI are still reported"}
    else:
        boot = paired_bootstrap(pairs) if len(pairs) >= 2 else {"error": "need at least 2 paired dialogues"}
    mc_p = mcnemar_exact_p(pairs) if pairs else None
    significance = {
        "p_value": boot.get("p_value"),
        "significant": boot.get("significant"),
        "error": boot.get("error"),
        "source": "auditkit._bootstrap.paired_bootstrap" if paired_bootstrap else "unavailable",
        "mcnemar_p": mc_p,
        "mcnemar_significant": None if mc_p is None else mc_p < 0.05,
        "ci": list(paired_diff_ci(pairs)) if pairs else None,
        "ci_source": "htd percentile bootstrap of mean(b - a)",
    }
    script_of = {t.dialogue_id: t.script for t in transcripts_a}
    by_script: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for i, p in zip(ids, pairs):
        by_script[script_of[i]].append(p)
    per_script = {
        s: {
            "n": len(ps),
            "mean_a": _mean([p[0] for p in ps]),
            "mean_b": _mean([p[1] for p in ps]),
            "diff": _mean([p[1] for p in ps]) - _mean([p[0] for p in ps]),
        }
        for s, ps in sorted(by_script.items())
    }
    return {
        "turn": turn,
        "n_paired": len(ids),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": mean_b - mean_a,
        "significance": significance,
        "per_script": per_script,
    }


def register_dr_metric(sheet: Sheet | None = None, name: str = DR_METRIC_NAME) -> type:
    """Register "disclosure retention" as an AuditKIT Metric (idempotent).

    The metric scores one reply: 1.0 if it states every required disclosure for
    `sample.metadata["product_id"]`, else 0.0. Evaluate a sample set built from the
    turn-k context and the mean over samples is DR@k. Returns the metric class.
    """
    from auditkit.metric import Metric
    from auditkit.registry import METRICS
    from auditkit.score import Score
    from auditkit.types import Direction

    from htd.rule_checker import check_reply
    from htd.sheet import load_sheet

    if name in METRICS.names():
        return METRICS.get(name)
    shared_sheet = sheet

    class DisclosureRetention(Metric):
        direction = Direction.MAXIMIZE
        required_fields = frozenset({"metadata"})

        def __init__(self) -> None:
            self.name = name

        def applicable(self, sample) -> bool:
            return bool((sample.metadata or {}).get("product_id"))

        def score(self, sample, output: str, context: Any = None) -> Score:
            s = shared_sheet or load_sheet()
            product = s.product(sample.metadata["product_id"])
            check = check_reply(output, product, s)
            return Score(name=self.name, value=1.0 if check.all_present else 0.0, kind=self.kind)

    DisclosureRetention.__name__ = "DisclosureRetention"
    METRICS.register(name)(DisclosureRetention)
    return DisclosureRetention
