"""Figures for the report, drawn only from evaluation artifacts.

python scripts/make_figures.py --eval outputs/kaggle-eval-final/htd/outputs/eval \
  --llama outputs/kaggle-second-model/htd/outputs/eval_llama \
  --extra outputs/kaggle-eval-2x2/htd/outputs/eval --out docs/figures

Every number plotted comes from summary_*.json, cmp_*.json, judge/panel/helpfulness JSONL or the
DPO training rows. Missing inputs skip a figure instead of inventing values. Colours are Okabe-Ito;
markers repeat the colour cue. Each figure is written as PDF and PNG, with a caption and alt text in
docs/figures/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.agreement import report as agreement_report  # noqa: E402
from htd.audit import wilson  # noqa: E402
from htd.helpful import helpful_rates  # noqa: E402
from htd.pairs import pair_profile  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402

OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
         "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
# label, colour, marker, linestyle
SYSTEMS = {
    "base": ("Tiny Aya base", OKABE["black"], "o", "-"),
    "sft": ("SFT run v6", OKABE["blue"], "s", "-"),
    "sft_s1": ("SFT seed 1", OKABE["sky"], "D", "-"),
    "sft_v5": ("SFT run v5", OKABE["green"], "^", "-"),
    "at": ("DPO all pairs, 5e-6", OKABE["vermillion"], "v", "--"),
    "at_drop": ("DPO drop+warn, 5e-6", OKABE["orange"], "<", "--"),
    "at_lr5e5": ("DPO all pairs, 5e-5", OKABE["purple"], "X", ":"),
    "at_drop_lr5e5": ("DPO drop+warn, 5e-5", OKABE["grey"], "P", ":"),
    "llama_base": ("Llama base", OKABE["black"], "o", "-"),
    "llama_sft": ("Llama SFT", OKABE["blue"], "s", "-"),
}
SCRIPTS = ("en", "hi", "hinglish")


def face(m: str) -> str:
    """Llama systems use hollow markers so they never share an encoding with Tiny Aya systems."""
    return "white" if m.startswith("llama") else SYSTEMS[m][1]
MM = 1 / 25.4
CAPTIONS: list[tuple[str, str, str]] = []


def pooled(summary: dict, model: str, turn: int = 5) -> tuple[int, int]:
    """(dialogues stating every disclosure at the turn, dialogues) pooled over scripts."""
    cells = [c for c in summary["cells"] if c["model"] == model and c["variant"] == "plain"]
    k = sum(round(c["dr"][turn - 1] * c["n"]) for c in cells)
    return k, sum(c["n"] for c in cells)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def save(fig, out: Path, name: str, caption: str, alt: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor="white")
    CAPTIONS.append((name, caption, alt))


def fig_retention(plt, test: dict, out: Path) -> None:
    models = [m for m in ("base", "sft", "sft_s1", "sft_v5", "at", "at_drop", "at_drop_lr5e5", "at_lr5e5")
              if any(c["model"] == m for c in test["cells"])]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), sharey=True, layout="constrained")
    for ax, script in zip(axes, SCRIPTS):
        for m in models:
            cell = next(c for c in test["cells"] if c["model"] == m and c["script"] == script and c["variant"] == "plain")
            label, colour, marker, ls = SYSTEMS[m]
            ax.plot(range(1, 6), [100 * x for x in cell["dr"]], color=colour, marker=marker, ls=ls,
                    lw=1.2, ms=4, label=label)
        ax.set(title={"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (romanized)"}[script],
               xlabel="Turn", xticks=range(1, 6), ylim=(-3, 103))
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Dialogues stating every\nrequired disclosure (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=7, frameon=False)
    save(fig, out, "fig1_retention_by_turn",
         "Disclosure retention DR@k by turn on the test split, per script (20 dialogues per point). Turn 1 is a "
         "plain request; turns 2 to 5 add pushback (brevity, social proof, urgency, direct override).",
         "Three line charts, one per script, showing the share of dialogues that state every required disclosure "
         "at turns 1 to 5. Base Tiny Aya starts high at turn 1 in English and Hindi and falls towards zero by turn 5; "
         "the three SFT runs stay well above base at turn 5; the DPO runs at lr 5e-5 end above base, the runs at 5e-6 near or below it.")
    plt.close(fig)


def fig_forest(plt, dirs: list[Path], out: Path) -> None:
    order = ["sft", "sft_s1", "sft_v5", "at", "at_drop", "at_drop_lr5e5", "at_lr5e5", "llama_sft"]
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 75 * MM), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("test", "heldout")):
        rows = []
        for m in order:
            c = next((read_json(d / f"cmp_{m}_{split}.json") for d in dirs if (d / f"cmp_{m}_{split}.json").exists()), None)
            if c and c["significance"].get("ci"):
                rows.append((m, c))
        for i, (m, c) in enumerate(rows):
            label, colour, marker, _ = SYSTEMS[m]
            lo, hi = c["significance"]["ci"]
            y = len(rows) - 1 - i
            ax.errorbar(100 * c["diff"], y, xerr=[[100 * (c["diff"] - lo)], [100 * (hi - c["diff"])]],
                        fmt=marker, color=colour, mfc=face(m), ms=5, capsize=2, lw=1)
            ax.text(101, y, f"{100 * c['mean_a']:.0f}% → {100 * c['mean_b']:.0f}%", va="center", ha="right", fontsize=6.5)
            if m == "llama_sft" and i > 0:
                ax.axhline(y + 0.5, color=OKABE["grey"], lw=0.5, ls=":")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(rows)), [SYSTEMS[m][0] for m, _ in reversed(rows)])
        ax.set(title={"test": "Test (unseen pushback lines)", "heldout": "Held-out (unseen products and tactics)"}[split],
               xlabel="Change in DR@5 vs its base (percentage points)", xlim=(-35, 102))
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    save(fig, out, "fig2_effects_forest",
         "Change in turn-5 disclosure retention against the system's own base model, with 95% percentile "
         "bootstrap intervals over 60 paired dialogues. Labels give base and system DR@5. Tiny Aya systems are compared "
         "with Tiny Aya base, Llama SFT (hollow marker, below the dotted line) with Llama base.",
         "Forest plot. All SFT systems, including Llama SFT, sit 30 to 57 points right of zero on both splits with "
         "intervals clear of zero; DPO at lr 5e-5 sits 18 to 35 points right of zero, DPO at 5e-6 near or left of zero.")
    plt.close(fig)


def fig_dpo_grid(plt, summaries: dict[str, dict], out: Path) -> None:
    cells = {("all", "5e-6"): "at", ("drop+warn", "5e-6"): "at_drop",
             ("all", "5e-5"): "at_lr5e5", ("drop+warn", "5e-5"): "at_drop_lr5e5"}
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 62 * MM), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("test", "heldout")):
        s = summaries.get(split)
        if not s:
            continue
        kb, nb = pooled(s, "base")
        ax.axhline(100 * kb / nb, color="black", ls="--", lw=0.8, label="base")
        for j, lr in enumerate(("5e-6", "5e-5")):
            xs, ys, err = [], [], [[], []]
            for i, mix in enumerate(("all", "drop+warn")):
                m = cells[(mix, lr)]
                if not any(c["model"] == m for c in s["cells"]):
                    continue
                k, n = pooled(s, m)
                lo, hi = wilson(k, n)
                xs.append(i + (j - 0.5) * 0.12)
                ys.append(100 * k / n)
                err[0].append(100 * (k / n - lo))
                err[1].append(100 * (hi - k / n))
            if xs:
                ax.errorbar(xs, ys, yerr=err, fmt=["o", "s"][j], color=[OKABE["vermillion"], OKABE["purple"]][j],
                            ms=5, capsize=2, lw=1, label=f"learning rate {lr}")
        ax.set(title={"test": "Test", "heldout": "Held-out"}[split], xticks=[0, 1],
               xticklabels=["all pairs\n(drop, lecture, warn)", "drop + warn\n(lecture removed)"], xlim=(-0.5, 1.5),
               ylim=(0, 100))
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("DR@5 (%)")
    axes[1].legend(fontsize=7, frameon=False)
    save(fig, out, "fig3_dpo_grid",
         "DPO 2x2: pair mix by learning rate. Pooled DR@5 over 60 dialogues with Wilson 95% intervals; dashed line "
         "is base.",
         "Two panels of points with error bars for four DPO runs against a dashed base line, on test and held-out.")
    plt.close(fig)


def fig_validity(plt, judges: dict[str, list[dict]], ref: list[dict], out: Path) -> None:
    sheet = load_sheet()
    # Judge selection was decided on Tiny Aya replies only; Llama rows are excluded from the kappas.
    refd = {r["dialogue_id"]: r["judge_label"] for r in ref if not r["model"].startswith("llama")}
    by: dict[str, list[tuple[bool, bool]]] = {}
    for r in ref:
        c = check_reply(r["reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present
        by.setdefault(r["model"], []).append((c, r["judge_label"] == "PASS"))
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 65 * MM), layout="constrained",
                             gridspec_kw={"width_ratios": [1.4, 1]})
    models = [m for m in SYSTEMS if m in by]
    agree = [100 * sum(a == b for a, b in by[m]) / len(by[m]) for m in models]
    axes[0].bar(range(len(models)), agree, color=[face(m) for m in models], edgecolor=[SYSTEMS[m][1] for m in models],
                hatch=["//" if m.startswith("llama") else "" for m in models], lw=0.8)
    axes[0].set_xticks(range(len(models)), [SYSTEMS[m][0] for m in models], rotation=40, ha="right", fontsize=6.5)
    axes[0].set(ylim=(0, 110), yticks=range(0, 101, 20), ylabel="Agreement with panel (%)",
                title="Rule checker vs LLM panel (pass / fail)")
    for i, v in enumerate(agree):
        axes[0].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    names, kappas = [], []
    for name, rows in judges.items():
        pairs = [(refd[r["dialogue_id"]], r["judge_label"]) for r in rows
                 if r["dialogue_id"] in refd and r["judge_label"] != "ERROR"]
        names.append(name)
        kappas.append(agreement_report(pairs)["cohen_kappa"])
    hatches = ["", "//", "..", "xx"]
    axes[1].bar(range(len(names)), kappas, color="white", edgecolor="black", hatch=hatches[: len(names)], lw=0.6)
    axes[1].set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=6.5)
    axes[1].set(ylim=(0, 1), ylabel="Cohen's kappa vs panel", title="Judge configurations")
    for i, v in enumerate(kappas):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    save(fig, out, "fig4_measurement_validity",
         "Left: agreement between the rule checker and the LLM reference panel (majority of Llama-3.3-70B, "
         "DeepSeek-V3, gpt-oss-120b) on turn-5 test replies, per system. Right: Cohen's kappa of each judge "
         "configuration against the panel on Tiny Aya replies (the rows the pre-registered selection used). The panel is "
         "an LLM reference, not human labels.",
         "Left bar chart: checker-panel agreement between about 78 and 100 percent for every system. Right bar "
         "chart: the pre-registered judge has the lowest kappa; rubric v2 raises kappa sharply with Gemma and "
         "leaves it flat with Qwen.")
    plt.close(fig)


def fig_tradeoff(plt, summaries: list[dict], helpful: list[list[dict]], out: Path) -> None:
    disc = helpful_rates([[r for r in rows if r["requires_disclosure"]] for rows in helpful])
    ctrl = helpful_rates([[r for r in rows if not r["requires_disclosure"]] for rows in helpful])
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), sharex=True, layout="constrained")
    for ax, rates, title in ((axes[0], disc, "Disclosure products"), (axes[1], ctrl, "Control products")):
        pts = {}
        for s in summaries:
            for m in sorted({c["model"] for c in s["cells"]}):
                if m in rates and m in SYSTEMS:
                    k, n = pooled(s, m)
                    pts[m] = (100 * k / n, 100 * rates[m]["yes"] / rates[m]["n"])
        for base, trained in (("base", ("sft", "sft_s1", "sft_v5")), ("llama_base", ("llama_sft",))):
            for t in trained:
                if base in pts and t in pts:
                    ax.annotate("", pts[t], pts[base], arrowprops={"arrowstyle": "->", "color": OKABE["grey"], "lw": 0.6})
        for m, (x, y) in pts.items():
            label, colour, marker, _ = SYSTEMS[m]
            ax.scatter(x, y, color=face(m), marker=marker, s=34, edgecolor=colour if m.startswith("llama") else "black",
                       lw=0.9 if m.startswith("llama") else 0.4, zorder=3, label=label)
        ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)
        ax.set(title=title, xlabel="DR@5 on test (%)", xlim=(-3, 70), ylim=(0, 100))
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Helpful replies at turn 5 (%, panel majority)")
    save(fig, out, "fig5_helpfulness_tradeoff",
         "Disclosure retention against helpfulness. Helpfulness is the LLM panel majority answer to whether the "
         "turn-5 reply gives the help asked for, regardless of warnings (60 replies per system and product type).",
         "Two scatter plots with arrows from each base model to its SFT systems. SFT moves right (more retention) and "
         "down (less helpful), most clearly for Tiny Aya on control products. Hollow markers are Llama.")
    plt.close(fig)


def fig_pairs(plt, rows: list[dict], out: Path) -> None:
    prof = pair_profile(rows, load_sheet())
    types = [t for t in ("drop", "lecture", "warn") if t in prof]
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM), layout="constrained")
    for ax, key, ylabel in ((axes[0], "words", "Mean words per reply"), (axes[1], "markers", "Mean warning phrases")):
        x = range(len(types))
        ax.bar([i - 0.18 for i in x], [prof[t][f"{key}_chosen"] for t in types], 0.36, color=OKABE["blue"],
               edgecolor="black", lw=0.4, label="chosen")
        ax.bar([i + 0.18 for i in x], [prof[t][f"{key}_rejected"] for t in types], 0.36, color="white",
               edgecolor=OKABE["vermillion"], hatch="///", lw=0.8, label="rejected")
        ax.set_xticks(list(x), [f"{t}\n(n={prof[t]['n']})" for t in types])
        ax.set(ylabel=ylabel)
    axes[1].legend(fontsize=7, frameon=False)
    save(fig, out, "fig6_dpo_pair_mix",
         "DPO training pairs by type. In lecture and warn pairs the rejected reply is longer and carries more "
         "warning phrases than the chosen reply, so these pairs push the policy away from warning wording.",
         "Grouped bar charts of words and warning phrases for chosen and rejected replies in drop, lecture and warn "
         "pairs. Only in drop pairs does the chosen reply carry more warnings than the rejected one.")
    plt.close(fig)


def fig_followup(plt, ev: Path, base_dir: Path, out: Path) -> None:
    from htd.replay import load_transcripts

    sheet = load_sheet()

    def rate(f: Path) -> tuple[int, int]:
        ts = load_transcripts(f)
        return sum(check_reply(t.replies[4], sheet.product(t.product_id), sheet, t.script).all_present for t in ts), len(ts)

    fig, axes = plt.subplots(1, 2, figsize=(160 * MM, 66 * MM), layout="constrained", width_ratios=[1.2, 1])
    ax = axes[0]
    rows = [("base", "Tiny Aya base", "base_test2"), ("base", "base, strict prompt", "base_strict_test2"),
            ("sft", "SFT run v6", "sft_test2"), ("sft_s1", "SFT seed 1", "sft_s1_test2"),
            ("sft_v5", "SFT run v5", "sft_v5_test2"), ("at_drop_lr5e5", "DPO drop+warn, 5e-5", "at_drop_lr5e5_test2"),
            ("at_lr5e5", "DPO all pairs, 5e-5", "at_lr5e5_test2")]
    for y, (m, label, f) in enumerate(rows):
        k, n = rate(ev / f"{f}.jsonl")
        lo, hi = wilson(k, n)
        hollow = label.endswith("strict prompt")
        ax.errorbar(100 * k / n, y, xerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], fmt=SYSTEMS[m][2],
                    color=SYSTEMS[m][1], mfc="white" if hollow else SYSTEMS[m][1], ms=5, capsize=2, lw=1)
    ax.set(yticks=range(len(rows)), yticklabels=[r[1] for r in rows], xlim=(0, 100), xlabel="DR@5 (%)",
           title="New pushback lines (test2, 120 dialogues)")
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax = axes[1]
    for j, split in enumerate(("test", "heldout")):
        kb, nb = rate(base_dir / f"fire_{split}_plain.jsonl")
        ax.hlines(100 * kb / nb, j - 0.4, j + 0.4, color="black", ls="--", lw=0.8, label="base" if j == 0 else None)
        for i, (m, label) in enumerate((("at_lr5e5", "all pairs"), ("at_drop_lr5e5", "drop+warn"))):
            files = [base_dir / f"{m}_{split}.jsonl"] + [ev / f"{m}_s{s}_{split}.jsonl" for s in (1, 2)]
            ys = [100 * k / n for k, n in map(rate, files)]
            x = j + (i - 0.5) * 0.36
            ax.scatter([x] * 3, ys, marker=SYSTEMS[m][2], color=SYSTEMS[m][1], s=18, zorder=3,
                       label=f"DPO {label}, 5e-5" if j == 0 else None)
            ax.hlines(sum(ys) / 3, x - 0.1, x + 0.1, color=SYSTEMS[m][1], lw=1.5)
    ax.set(xticks=[0, 1], xticklabels=["Test", "Held-out"], ylim=(0, 100), ylabel="DR@5 (%)",
           title="DPO, three runs per pair mix")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, out, "fig7_followup",
         "Follow-up study. Left: turn-5 retention on 16 new pushback lines (test2, 120 dialogues) with Wilson 95% "
         "intervals; the hollow marker is base with the strict prompt. Right: DPO at 5e-5, three training runs per "
         "pair mix (seeds 0, 1, 2; points) and their mean (bar), 60 dialogues per split; dashed line is base.",
         "Left: horizontal dot plot with error bars; every trained system is to the right of base, the strict prompt "
         "lies between base and the trained systems. Right: three points per pair mix on test and held-out, all above "
         "the dashed base line; the two mixes overlap.")
    plt.close(fig)


def fig_repetition(plt, ev: Path, qwen_strict: Path, out: Path, judges: Path | None = None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_robustness import max_self_sim, system_votes

    from htd.helpful import helpful_votes

    from htd.replay import load_transcripts

    sheet = load_sheet()
    files = [judges / f"helpful_{n}.jsonl" for n in ("llama", "deepseek", "gptoss")] if judges else []
    votes = helpful_votes([[json.loads(x) for x in f.read_text().splitlines() if x.strip()] for f in files]) \
        if files and all(f.exists() for f in files) else None
    rows = [("base", "Tiny Aya base", ev / "base_test2.jsonl"), ("base", "base, strict prompt", ev / "base_strict_test2.jsonl"),
            ("sft", "SFT run v6", ev / "sft_test2.jsonl"), ("sft_s1", "SFT seed 1", ev / "sft_s1_test2.jsonl"),
            ("sft_v5", "SFT run v5", ev / "sft_v5_test2.jsonl"),
            ("at_drop_lr5e5", "DPO drop+warn, 5e-5", ev / "at_drop_lr5e5_test2.jsonl"),
            ("at_lr5e5", "DPO all pairs, 5e-5", ev / "at_lr5e5_test2.jsonl"),
            ("base", "Qwen2.5-72B, strict prompt", qwen_strict)]
    rows = [r for r in rows if r[2].exists()]
    fig, ax = plt.subplots(figsize=(150 * MM, 62 * MM), layout="constrained")
    for y, (m, _, f) in enumerate(rows):
        ts = load_transcripts(f)
        ok = [check_reply(t.replies[4], sheet.product(t.product_id), sheet, t.script).all_present for t in ts]
        fresh = sum(o and max_self_sim(t) < 0.9 for o, t in zip(ok, ts)) / len(ts)
        copied = sum(ok) / len(ts) - fresh
        c = SYSTEMS[m][1]
        ax.barh(y, 100 * fresh, color=c, edgecolor=c, height=0.6, label="fresh reply" if y == 0 else None)
        ax.barh(y, 100 * copied, left=100 * fresh, color="white", edgecolor=c, hatch="////", height=0.6,
                label="copy of an earlier reply" if y == 0 else None)
        if votes is not None:
            v = system_votes(ts, votes)
            hd = sum(o and v.get(t.dialogue_id, False) for o, t in zip(ok, ts)) / len(ts)
            ax.plot(100 * hd, y, marker="D", ms=5, mfc="white", mec="black", mew=1.2, ls="none",
                    label="keeps it and helps (HD@5)" if y == 0 else None)
    ax.set(yticks=range(len(rows)), yticklabels=[r[1] for r in rows], xlim=(0, 100),
           xlabel="Turn-5 replies that state every disclosure (%)", title="New pushback lines (test2, 120 dialogues)")
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, out, "fig8_repetition",
         "Retention by repetition on test2. The full bar is DR@5. The solid part counts replies that differ from "
         "every earlier reply of the same model (fresh DR@5); the hatched part counts replies with similarity 0.9 or "
         "more (Python difflib ratio) to one of the model's own replies at turns 1 to 4 (post hoc). The diamond is "
         "HD@5: the reply states every disclosure and the LLM helpfulness panel calls it helpful (pre-registered "
         "check R1; model labels).",
         "Horizontal stacked bars. Trained systems have long bars that are mostly hatched; their solid parts are as "
         "short as base. The instructed Qwen2.5-72B bar is almost entirely solid and its solid part is the longest. "
         "Diamonds: base about 4%, SFT runs 22 to 28%, DPO 11 to 17%, Qwen2.5-72B strict about 31%, so the SFT "
         "diamonds sit right of their solid parts, inside the hatched parts.")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--llama", type=Path)
    ap.add_argument("--extra", type=Path, help="eval dir with the 2x2 DPO arms (overrides --eval when present)")
    ap.add_argument("--judges", type=Path, default=Path("outputs/judge_local"))
    ap.add_argument("--pairs", type=Path, default=Path("data/generated/pairs/dpo_train.jsonl"))
    ap.add_argument("--followup", type=Path, default=Path("outputs/followup/eval"))
    ap.add_argument("--base-dir", type=Path, default=Path("outputs/staging/base_eval"))
    ap.add_argument("--roi-strict", type=Path, default=Path("outputs/roi_strict/qwen72b_strict_test2.jsonl"))
    ap.add_argument("--test2-judges", type=Path, default=Path("outputs/judge_test2"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    a.out.mkdir(parents=True, exist_ok=True)
    main_dir = a.extra if a.extra and (a.extra / "summary_test.json").exists() else a.eval
    test, held = read_json(main_dir / "summary_test.json"), read_json(main_dir / "summary_heldout.json")
    dirs = [d for d in (a.extra, a.eval, a.llama) if d and d.exists()]
    fig_retention(plt, test, a.out)
    fig_forest(plt, dirs, a.out)
    fig_dpo_grid(plt, {"test": test, "heldout": held}, a.out)
    J = a.judges
    judges = {"v1: rubric v1 + Gemma\n(pre-registered)": read_jsonl(main_dir.parent / "judge" / "judge_test.jsonl"),
              "v2: rubric v2 + Qwen\n(selected)": read_jsonl(J / "v2_qwen.jsonl"),
              "ablation: v1 + Qwen": read_jsonl(J / "v1_qwen.jsonl"),
              "ablation: v2 + Gemma": read_jsonl(J / "v2_gemma.jsonl")}
    ref = read_jsonl(J / "panel_reference.jsonl")
    if ref:
        fig_validity(plt, {k: v for k, v in judges.items() if v}, ref, a.out)
    helpful = [read_jsonl(J / f"helpful_{n}.jsonl") for n in ("llama", "deepseek", "gptoss")]
    if all(helpful):
        llama_test = read_json(a.llama / "summary_test.json") if a.llama else None
        fig_tradeoff(plt, [s for s in (test, llama_test) if s], helpful, a.out)
    if a.pairs.exists():
        fig_pairs(plt, read_jsonl(a.pairs), a.out)
    if (a.followup / "base_test2.jsonl").exists():
        fig_followup(plt, a.followup, a.base_dir, a.out)
        fig_repetition(plt, a.followup, a.roi_strict, a.out, a.test2_judges)

    lines = ["# Figures", "", "Generated by `scripts/make_figures.py` from the evaluation artifacts. Do not edit by hand.", ""]
    for name, caption, alt in CAPTIONS:
        lines += [f"## {name}", "", f"![{name}]({name}.png)", "", f"**Caption.** {caption}", "", f"**Alt text.** {alt}", ""]
    (a.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(CAPTIONS)} figures to {a.out}")


if __name__ == "__main__":
    main()
