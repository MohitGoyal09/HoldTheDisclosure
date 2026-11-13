"""Render the follow-up study (pre-registration F1 to F4, F6) from the artifacts.

python scripts/render_followup.py --out docs/RESULTS_FOLLOWUP.md

Reads transcripts in outputs/followup/eval/ (test2 kernels and DPO seed kernels), seed-0 DPO and base
transcripts in outputs/staging/base_eval/, panel and helpfulness outputs in outputs/judge_local/, and
the blind label sheets in outputs/labels/. Every comparison is recomputed here with the rule checker.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.audit import compare_systems, wilson  # noqa: E402
from htd.dialogues import build_split  # noqa: E402
from htd.helpful import helpful_votes, helpful_vs_base  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from render_results import human_label_lines, read_csv  # noqa: E402

SCRIPTS = ("en", "hi", "hinglish")
F1 = (("sft", "SFT v6"), ("sft_s1", "SFT seed 1"), ("sft_v5", "SFT v5"),
      ("at_drop_lr5e5", "DPO drop+warn 5e-5"), ("at_lr5e5", "DPO all pairs 5e-5"))
F2 = (("base", "base"), ("sft", "SFT v6"), ("at_drop_lr5e5", "DPO drop+warn 5e-5"))
MIXES = (("at_lr5e5", "all pairs"), ("at_drop_lr5e5", "drop+warn"))


def holm(ps: dict[str, float]) -> dict[str, float]:
    out, run, m = {}, 0.0, len(ps)
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def by_index(ts):
    return [replace(t, dialogue_id=t.dialogue_id.split("-", 1)[1]) for t in ts]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def wil(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}, {100 * hi:.0f}]"


def cmp_row(name: str, c: dict, extra: str = "") -> str:
    s = c["significance"]
    return (f"| {name} | {pct(c['mean_b'])} | {100 * c['diff']:+.1f} [{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}] "
            f"| {s['mcnemar_p']:.2g} |{extra}")


def kept(sheet, t, turn: int) -> bool:
    return check_reply(t.replies[turn - 1], sheet.product(t.product_id), sheet, t.script).all_present


def section_f1(sheet, ev: Path) -> list[str]:
    base = load_transcripts(ev / "base_test2.jsonl")
    res = {k: compare_systems(base, load_transcripts(ev / f"{k}_test2.jsonl"), sheet, turn=5) for k, _ in F1}
    adj = holm({k: c["significance"]["mcnemar_p"] for k, c in res.items()})
    strict = compare_systems(base, load_transcripts(ev / "base_strict_test2.jsonl"), sheet, turn=5)
    n = len(base)
    out = ["## F1. Replication on new pushback lines (`test2`)", "",
           f"{n} dialogues (40 per script), 16 new pushback lines reviewed by a native speaker, fixed order, plain prompt. "
           f"Base DR@5 {pct(res['sft']['mean_a'])}. Exact McNemar against base, paired by dialogue; Holm over the five "
           "trained systems.", "",
           "| system | DR@5 | diff vs base (points) [95% CI] | McNemar p | Holm p |", "|---|---|---|---|---|"]
    out += [cmp_row(name, res[k], f" {adj[k]:.2g} |") for k, name in F1]
    out += ["", "Secondary, descriptive: the strict prompt (every required disclosure listed in the system prompt).", "",
            "| system | DR@5 | diff vs base (points) [95% CI] | McNemar p |", "|---|---|---|---|",
            cmp_row("base, strict prompt", strict)]
    for k, name in (("sft", "SFT v6"), ("at_drop_lr5e5", "DPO drop+warn 5e-5")):
        c = compare_systems(load_transcripts(ev / "base_strict_test2.jsonl"), load_transcripts(ev / f"{k}_test2.jsonl"),
                            sheet, turn=5)
        out.append(f"| {name} against strict prompt | {pct(c['mean_b'])} | {100 * c['diff']:+.1f} | "
                   f"{c['significance']['mcnemar_p']:.2g} |")
    out += ["", "Per script DR@5 with Wilson 95% intervals (40 dialogues each):", "",
            "| system | en | hi | hinglish |", "|---|---|---|---|"]
    for k, name in (("base", "base"), ("base_strict", "base, strict prompt")) + F1:
        ts = load_transcripts(ev / f"{k}_test2.jsonl")
        cells = []
        for sc in SCRIPTS:
            rows = [t for t in ts if t.script == sc]
            cells.append(wil(sum(kept(sheet, t, 5) for t in rows), len(rows)))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out + [""]


def section_f2(sheet, ev: Path) -> list[str]:
    out = ["## F2. Turn order (`test2_shuffled`)", "",
           "The same 120 dialogues with turns 2 to 5 reordered by a 4x4 Williams square (override last in a quarter of "
           "dialogues). Fixed against shuffled, paired by dialogue index and script, exact McNemar.", "",
           "| system | fixed DR@5 | shuffled DR@5 | diff (points) [95% CI] | McNemar p |", "|---|---|---|---|---|"]
    for k, name in F2:
        c = compare_systems(by_index(load_transcripts(ev / f"{k}_test2.jsonl")),
                            by_index(load_transcripts(ev / f"{k}_test2shuf.jsonl")), sheet, turn=5)
        s = c["significance"]
        out.append(f"| {name} | {pct(c['mean_a'])} | {pct(c['mean_b'])} | {100 * c['diff']:+.1f} "
                   f"[{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}] | {s['mcnemar_p']:.2g} |")
    tactics = {d.dialogue_id: d.tactic_ids for d in build_split(sheet, "test2_shuffled", 40)}
    out += ["", "Shuffled order, descriptive: share of dialogues that lose the disclosures right after each tactic, among "
            "dialogues that had them at the turn before (Wilson 95% intervals).", "",
            "| system | brevity | social proof | urgency | override |", "|---|---|---|---|---|"]
    for k, name in F2:
        lost = {tac: [0, 0] for tac in ("brevity", "social_proof", "urgency", "override")}
        for t in load_transcripts(ev / f"{k}_test2shuf.jsonl"):
            flags = [kept(sheet, t, i) for i in range(1, 6)]
            for turn, tid in enumerate(tactics[t.dialogue_id], start=2):
                tac = next(x for x in lost if f"_{x}_" in tid)
                if flags[turn - 2]:
                    lost[tac][1] += 1
                    lost[tac][0] += not flags[turn - 1]
        out.append(f"| {name} | " + " | ".join(wil(*lost[x]) if lost[x][1] else "n/a" for x in lost) + " |")
    return out + [""]


def section_f3(sheet, ev: Path, base_dir: Path, helpful: list[list[dict]]) -> list[str]:
    out = ["## F3. DPO seeds", "",
           "DPO at 5e-5, three runs per pair mix: seed 0 (first study, LoRA start not seeded), seeds 1 and 2 "
           "(`set_seed` before model load). DR@5 against base, exact McNemar, 60 dialogues per split.", "",
           "| pair mix | split | base | seed 0 | seed 1 | seed 2 | mean (range) |", "|---|---|---|---|---|---|---|"]
    held = {}
    for mix, mname in MIXES:
        for split in ("test", "heldout"):
            base = load_transcripts(base_dir / f"fire_{split}_plain.jsonl")
            files = [base_dir / f"{mix}_{split}.jsonl"] + [ev / f"{mix}_s{s}_{split}.jsonl" for s in (1, 2)]
            cs = [compare_systems(base, load_transcripts(f), sheet, turn=5) for f in files]
            vals = [c["mean_b"] for c in cs]
            held[(mix, split)] = vals
            cells = [f"{pct(c['mean_b'])} (p={c['significance']['mcnemar_p']:.2g})" for c in cs]
            out.append(f"| {mname} | {split} | {pct(cs[0]['mean_a'])} | " + " | ".join(cells)
                       + f" | {pct(sum(vals) / 3)} ({pct(min(vals))} to {pct(max(vals))}) |")
    a, d = held[("at_lr5e5", "heldout")], held[("at_drop_lr5e5", "heldout")]
    verdict = ("all pairs above drop+warn" if min(a) > max(d) else "drop+warn above all pairs" if min(d) > max(a)
               else "not distinguishable at three runs")
    out += ["", f"Pre-registered mix rule (all three runs of one mix above all three of the other on held-out DR@5): "
            f"{verdict} (all pairs {pct(min(a))} to {pct(max(a))}, drop+warn {pct(min(d))} to {pct(max(d))}).", ""]
    ctrl = [t for f in sorted(ev.glob("at_*_s?_control.jsonl")) for t in load_transcripts(f)]
    replies = [bool(check_reply(r, sheet.product(t.product_id), sheet, t.script).warning_markers_found)
               for t in ctrl for r in t.replies]
    out += [f"Control products: {sum(replies)} of {len(replies)} replies (all turns, seeds 1 and 2) carry an unneeded "
            "warning.", ""]
    if helpful:
        out += ["Helpfulness at turn 5 (LLM panel majority), against base, exact McNemar:", "",
                "| system | disclosure products | p | control products | p |", "|---|---|---|---|---|"]
        votes = {flag: helpful_votes([[x for x in p if x["requires_disclosure"] == flag] for p in helpful])
                 for flag in (True, False)}
        for mix, mname in MIXES:
            for s, key in (("0", mix), ("1", f"{mix}_s1"), ("2", f"{mix}_s2")):
                cells = []
                for flag in (True, False):
                    v = votes[flag][key]
                    k, n = sum(bool(x) for x in v.values()), len(v)
                    cells += [wil(k, n), f"{helpful_vs_base(votes[flag], key, 'base')[1]:.2g}"]
                out.append(f"| DPO {mname}, seed {s} | " + " | ".join(cells) + " |")
        b = [votes[f]["base"] for f in (True, False)]
        out += [f"| base | {wil(sum(bool(x) for x in b[0].values()), len(b[0]))} | | "
                f"{wil(sum(bool(x) for x in b[1].values()), len(b[1]))} | |", ""]
    return out


def panel_lines(sheet, ref: list[dict]) -> list[str]:
    out = ["LLM panel majority against the rule checker on the seed-1 and seed-2 test replies (turn 5; rows without a "
           "majority are dropped):", "", "| system | n | panel PASS | checker pass | agree | panel PASS, checker fail |",
           "|---|---|---|---|---|---|"]
    for m in sorted({r["model"] for r in ref}):
        rows = [r for r in ref if r["model"] == m]
        pp = [r["judge_label"] == "PASS" for r in rows]
        ck = [check_reply(r["reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present for r in rows]
        out.append(f"| {m} | {len(rows)} | {sum(pp)} | {sum(ck)} | {sum(x == y for x, y in zip(pp, ck))} | "
                   f"{sum(x and not y for x, y in zip(pp, ck))} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=Path("outputs/followup/eval"))
    ap.add_argument("--base-dir", type=Path, default=Path("outputs/staging/base_eval"))
    ap.add_argument("--judges", type=Path, default=Path("outputs/judge_local"))
    ap.add_argument("--labels", nargs="*", default=["outputs/labels/label_sheet_kimi.csv",
                                                    "outputs/labels/label_sheet_glm.csv"])
    ap.add_argument("--followup-judges", type=Path, default=Path("outputs/judge_followup"),
                    help="panel and helpfulness rows for the DPO seed runs")
    ap.add_argument("--v1-gemma", type=Path, default=Path("outputs/staging/hf_data/judges/v1_gemma_rubric_v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_FOLLOWUP.md"))
    a = ap.parse_args()
    sheet = load_sheet()
    rd = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    helpful = [rd(a.judges / f"helpful_{n}.jsonl") + rd(a.followup_judges / f"helpful_{n}.jsonl")
               for n in ("llama", "deepseek", "gptoss")]
    lines = ["# Follow-up study (pre-registration addendum F1 to F6)", "",
             "Generated by `scripts/render_followup.py` from the artifacts. Rule checker at turn 5 unless stated. "
             "F5 (large-model reference and cost) is in `docs/RESULTS_ROI.md`.", ""]
    lines += section_f1(sheet, a.eval) + section_f2(sheet, a.eval) + section_f3(sheet, a.eval, a.base_dir, helpful)
    lines += panel_lines(sheet, rd(a.followup_judges / "panel_reference.jsonl"))
    judge_files = (("judge v1, Gemma-3-27B (pre-registered)", a.v1_gemma), ("judge v1, Qwen3-235B", a.judges / "v1_qwen.jsonl"),
                   ("judge v2, Gemma-3-27B", a.judges / "v2_gemma.jsonl"), ("judge v2, Qwen3-235B (selected)",
                   a.judges / "v2_qwen.jsonl"), ("LLM panel majority", a.judges / "panel_reference.jsonl"))
    configs = [(n, None, rd(f)) for n, f in judge_files if f.exists()]
    lab = human_label_lines([read_csv(p) for p in a.labels], configs)
    lab[0] = "## F4. Blind labels"
    lines += lab
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
