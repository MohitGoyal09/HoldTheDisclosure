"""Render score summaries and paired comparisons as Markdown, from JSON artifacts."""

from __future__ import annotations

from typing import Any

PCT = lambda x: f"{100 * x:.0f}%"  # noqa: E731


def summary_markdown(summary: dict[str, Any]) -> str:
    cells = summary.get("cells", [])
    controls = summary.get("controls", [])
    lines = []
    if cells:
        lines += [
            "| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    for c in cells:
        kept = "n/a" if c["kept5_given_t1"] is None else PCT(c["kept5_given_t1"])
        lines.append(
            f"| {c['model']} | {c['variant']} | {c['script']} | {c['n']} | "
            + " | ".join(PCT(x) for x in c["dr"])
            + f" | {'n/a' if not c.get('dr_in_script') else PCT(c['dr_in_script'][4])}"
            + f" | {PCT(c['ci_low'])}-{PCT(c['ci_high'])} | {c['mean_tof']:.2f} | {c['n_stated_t1']} | {kept} |"
        )
    if controls:
        lines += ["", "Control products (unneeded-warning rate)", "",
                  "| model | variant | script | replies | warning rate |", "|---|---|---|---|---|"]
        for c in controls:
            lines.append(f"| {c['model']} | {c['variant']} | {c['script']} | {c['n_replies']} | {PCT(c['warning_rate'])} |")
    return "\n".join(lines)


def comparison_markdown(cmp: dict[str, Any], name_a: str, name_b: str) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    sig = cmp["significance"]
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    flag = {True: "yes", False: "no", None: "n/a"}[sig["mcnemar_significant"]]
    ci = "n/a" if not sig["ci"] else f"[{100 * sig['ci'][0]:+.1f}, {100 * sig['ci'][1]:+.1f}] pts"
    lines = [
        f"DR@{cmp['turn']}: A = {name_a}, B = {name_b} (diff = B - A)",
        "",
        "| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |",
        "|---|---|---|---|---|---|---|---|",
        f"| all | {cmp['n_paired']} | {pct(cmp['mean_a'])} | {pct(cmp['mean_b'])} | "
        f"{100 * cmp['diff']:+.1f} pts | {ci} | {fmt(sig['mcnemar_p'])} | {flag} |",
    ]
    for s, v in cmp["per_script"].items():
        lines.append(f"| {s} | {v['n']} | {pct(v['mean_a'])} | {pct(v['mean_b'])} | {100 * v['diff']:+.1f} pts | | | |")
    return "\n".join(lines)


def judge_markdown(rows: list[dict[str, Any]], labels: tuple[str, ...] = ("PASS", "DROP", "DISMISS", "OVER-CORRECT")) -> str:
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        m = r.get("model", "?")
        counts.setdefault(m, {k: 0 for k in labels})
        lab = str(r.get("judge_label", "ERROR"))
        counts[m][lab] = counts[m].get(lab, 0) + 1
    lines = ["| system | judged | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for m in sorted(counts):
        total = sum(counts[m].values())
        row = " | ".join(f"{counts[m].get(k, 0)} ({100 * counts[m].get(k, 0) / total:.0f}%)" for k in labels)
        lines.append(f"| {m} | {total} | {row} |")
    return "\n".join(lines)


def checker_judge_markdown(rows: list[dict[str, Any]]) -> str:
    """Rule checker against the judge, per system. Rows need `checker_pass` (bool).

    A checker pass that the judge calls DROP or DISMISS means the reply matched the
    patterns without really stating the disclosure; a high count for a trained system
    is the sign of checker-matching phrasing.
    """
    stats: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("judge_label") == "ERROR":
            continue
        s = stats.setdefault(r.get("model", "?"), {"n": 0, "chk": 0, "pass": 0, "agree": 0, "false_pass": 0, "miss": 0})
        chk, ok = bool(r["checker_pass"]), r["judge_label"] == "PASS"
        s["n"] += 1
        s["chk"] += chk
        s["pass"] += ok
        s["agree"] += chk == ok
        s["false_pass"] += chk and r["judge_label"] in ("DROP", "DISMISS")
        s["miss"] += ok and not chk
    lines = [
        "| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in sorted(stats):
        s = stats[m]
        f = lambda k: f"{s[k]} ({PCT(s[k] / s['n'])})"  # noqa: E731
        lines.append(f"| {m} | {s['n']} | {f('chk')} | {f('pass')} | {f('agree')} | {s['false_pass']} | {s['miss']} |")
    return "\n".join(lines)


SYSTEM_NAMES = {
    "sft": "SFT",
    "sft_s1": "SFT, seed 1",
    "sft_v5": "SFT, training run v5",
    "at_drop_lr5e5": "AlignTune DPO, drop+warn pairs, lr 5e-5",
    "at_lr5e5": "AlignTune DPO, all pairs, lr 5e-5",
    "llama_sft": "Llama-3.2-3B SFT",
    "at": "AlignTune DPO",
    "at_drop": "AlignTune DPO, drop+warn pairs",
    "at_sft": "AlignTune SFT",
    "dpo5": "DPO (lr 5e-6)",
    "dpo1": "DPO (lr 1e-5)",
    "strict": "Strict system prompt (no training)",
    "third": "Third-person prompt (no training)",
}
SPLIT_NAMES = {"test": "test", "heldout": "held-out"}


def comparison_title(tag: str) -> str:
    """`cmp_<system>_<split>` to a readable heading."""
    system, split = tag.removeprefix("cmp_").rsplit("_", 1)
    return f"{SYSTEM_NAMES.get(system, system)} vs base, {SPLIT_NAMES.get(split, split)}"


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = round(running, 6)
    return adjusted
