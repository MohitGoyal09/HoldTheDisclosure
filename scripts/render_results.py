"""Render docs/RESULTS.md from the evaluation artifacts.

python scripts/render_results.py --dir outputs/eval --judge outputs/judge/judge_test.jsonl \
  --experiments experiments/results.jsonl --out docs/RESULTS.md
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.experiment import CRITERIA  # noqa: E402
from htd.report import (  # noqa: E402
    checker_judge_markdown,
    comparison_markdown,
    comparison_title,
    holm,
    judge_markdown,
    summary_markdown,
)
from htd.agreement import agreement, cohen_kappa, fleiss_kappa  # noqa: E402
from htd.agreement import report as agreement_report  # noqa: E402
from htd.audit import wilson  # noqa: E402
from htd.helpful import helpful_vs_base, helpful_votes  # noqa: E402
from htd.labels import consensus, human_column, item_id  # noqa: E402
from htd.pairs import pair_profile  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def read_csv(path: str) -> list[dict]:
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _agree_row(name: str, ref: dict[str, str], other: dict[str, str], labels: tuple[str, ...]) -> str:
    pairs_ = [(ref[k], other[k]) for k in ref if k in other]
    if not pairs_:
        return f"| {name} | 0 | n/a | n/a |"
    return f"| {name} | {len(pairs_)} | {fmt(agreement(pairs_))} | {fmt(cohen_kappa(pairs_, labels))} |"


def human_label_lines(sheets: list[list[dict]], configs: list) -> list[str]:
    """Human labels (pre-registration F4): labellers against each other, checker and judges against people."""
    sheet = load_sheet()
    rows = {r["item"]: r for r in sheets[0]}
    checker = {k: "YES" if check_reply(r["assistant_reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present
               else "NO" for k, r in rows.items()}
    states = [human_column(s, "states_all") for s in sheets]
    four = [human_column(s, "human_label") for s in sheets]
    notes = [next((r.get("human_notes", "") for r in s if r.get("human_notes", "").startswith("model label: ")), "") for s in sheets]
    names = [n.removeprefix("model label: ").split("/")[-1] or f"labeller {chr(65 + i)}" for i, n in enumerate(notes)]
    kind = "model" if all(notes) else "human" if not any(notes) else "mixed"
    if len(sheets) == 2:
        states.append(consensus(*states))
        four.append(consensus(*four[:2]))
        names.append("consensus")
    yn, labels4 = ("YES", "NO"), ("PASS", "DROP", "DISMISS", "OVER-CORRECT")
    out = ["## Blind labels (F4)", "",
           f"{len(rows)} blind turn-5 replies; {len(sheets)} {kind} labeller(s), independent"
           + (" (open-weight models through `scripts/llm_label.py`, not people)" if kind == "model" else "") + ". `states_all`: the reply states every "
           "required disclosure, in any wording. Consensus: items where both labellers agree.", ""]
    if len(sheets) == 2:
        out += ["Between labellers:", "", "| label | n | agreement | Cohen's kappa |", "|---|---|---|---|",
                _agree_row("states_all", states[0], states[1], yn),
                _agree_row("four-way", four[0], four[1], labels4), ""]
    out += ["Rule checker against labellers (`states_all`):", "", "| labeller | n | agreement | Cohen's kappa |", "|---|---|---|---|"]
    out += [_agree_row(n, s, checker, yn) for n, s in zip(names, states)] + [""]
    if configs:
        out += ["Judges and the LLM panel against labellers (four-way label):", "",
                "| judge | labeller | n | agreement | Cohen's kappa |", "|---|---|---|---|---|"]
        for cname, _, crows in configs:
            judged = {item_id(r["dialogue_id"]): r["judge_label"] for r in crows if r.get("judge_label") != "ERROR"}
            for n, h in zip(names, four):
                out.append(_agree_row(f"{cname} | {n}", h, judged, labels4))
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/eval")
    ap.add_argument("--judge", default="outputs/judge/judge_test.jsonl")
    ap.add_argument("--experiments", default="experiments/results.jsonl")
    ap.add_argument("--pairs", default="data/generated/pairs/dpo_train.jsonl", help="DPO training rows, for the pair-mix table")
    ap.add_argument("--judge-config", action="append", default=[], metavar="NAME=PATH",
                    help="extra judge outputs to render, e.g. v2_qwen=outputs/judge_local/v2_qwen.jsonl")
    ap.add_argument("--reference", default=None, help="LLM reference panel majority (scripts/panel_vote.py)")
    ap.add_argument("--auditkit-dir", type=Path, default=None,
                    help="compare.py outputs computed with AuditKIT installed, to show its bootstrap next to McNemar")
    ap.add_argument("--panel", nargs="*", default=[], help="the panel models' judge_run outputs, for Fleiss' kappa")
    ap.add_argument("--helpful", nargs="*", default=[], help="helpful_run.py outputs, one per panel model")
    ap.add_argument("--labels", nargs="*", default=[], help="filled human label sheets (CSV), one per labeller")
    ap.add_argument("--out", default="docs/RESULTS.md")
    a = ap.parse_args()
    d = Path(a.dir)

    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        commit = ""

    lines = [
        "# Results",
        "",
        f"Generated from the artifacts in `{d}` on {date.today().isoformat()}. Commit `{commit}`.",
        "All numbers come from `scripts/render_results.py`; do not edit them by hand.",
        "",
    ]

    for name, title in (("summary_test", "Test split"), ("summary_heldout", "Held-out products and tactics"), ("summary_control", "Control products")):
        s = read_json(d / f"{name}.json")
        if s:
            lines += [f"## {title}", "", summary_markdown(s), ""]
            if name == "summary_test" and any(c.get("transition") for c in s.get("cells", [])):
                lines += ["Turn-1 to turn-5 transitions", "",
                          "| model | variant | script | stated both | stated then dropped | missed then stated | missed both |",
                          "|---|---|---|---|---|---|---|"]
                for c in s["cells"]:
                    t = c.get("transition") or {}
                    lines.append(f"| {c['model']} | {c['variant']} | {c['script']} | {t.get('stated_both', 0)} | "
                                 f"{t.get('stated_then_dropped', 0)} | {t.get('missed_then_stated', 0)} | {t.get('missed_both', 0)} |")
                lines.append("")

    cmps = sorted(d.glob("cmp_*.json"), key=lambda p: (not p.stem.endswith("_test"), p.stem))
    # The Holm family is fixed in advance: the three trained recipes. Replicates and any other
    # comparison are listed with raw p-values only.
    primary = ("sft", "at", "at_drop")

    def usable(c):
        sig = (c or {}).get("significance") or {}
        return sig.get("mcnemar_p") is not None and sig.get("ci") is not None

    def row(p, c, adj):
        sig = c["significance"]
        lo, hi = sig["ci"]
        name = comparison_title(p.stem).split(" vs base")[0]
        return (f"| {name} | {c['n_paired']} | {100 * c['mean_a']:.1f}% | {100 * c['mean_b']:.1f}% | "
                f"{100 * c['diff']:+.1f} | {100 * lo:+.1f} to {100 * hi:+.1f} | {sig['mcnemar_p']:.4f} | {adj} |")

    for split, title in (("test", "test"), ("heldout", "held-out")):
        here = [(p, read_json(p)) for p in cmps if p.stem.endswith(f"_{split}")]
        skipped = [p.stem for p, c in here if not usable(c)]
        here = [(p, c) for p, c in here if usable(c)]
        system = lambda p: p.stem.removeprefix("cmp_").removesuffix(f"_{split}")  # noqa: E731
        fam = sorted([x for x in here if system(x[0]) in primary], key=lambda x: primary.index(system(x[0])))
        rest = [x for x in here if system(x[0]) not in primary]
        if not here:
            continue
        adj = holm([c["significance"]["mcnemar_p"] for _, c in fam])
        lines += [f"## DR@5 against base, {title}", "",
                  f"McNemar exact test, paired by dialogue. Holm family: {len(fam)} pre-set recipes ({', '.join(primary)}). "
                  "Other rows (replicates) are not in the family.", "",
                  "| system | n | base | system | diff | 95% CI | McNemar p | Holm p |", "|---|---|---|---|---|---|---|---|"]
        lines += [row(p, c, f"{h:.4f}") for (p, c), h in zip(fam, adj)]
        lines += [row(p, c, "not in family") for p, c in rest]
        if skipped:
            lines += ["", f"Skipped (no paired test): {', '.join(skipped)}"]
        lines.append("")
    for p in cmps:
        c = read_json(p)
        if not usable(c):
            continue
        lines += [f"## {comparison_title(p.stem)}", "", comparison_markdown(c, c.get("system_a", "base"), c.get("system_b", p.stem)), ""]

    jr = read_jsonl(a.judge)
    if jr:
        lines += ["## Judge labels (turn 5, test)", "", judge_markdown(jr), ""]
        sheet = load_sheet()
        checked = [dict(r, checker_pass=check_reply(r["reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present)
                   for r in jr if r.get("judge_label") != "ERROR"]
        lines += ["## Rule checker against the judge (turn 5, test)", "",
                  "A checker pass that the judge labels DROP or DISMISS means the reply matched the patterns "
                  "without stating the disclosure. A trained system with many of these learned checker-matching phrasing.", "",
                  checker_judge_markdown(checked), ""]

    configs = [("v1 (pre-registered)", a.judge)] + [tuple(x.split("=", 1)) for x in a.judge_config]
    configs = [(n, p, read_jsonl(p)) for n, p in configs if read_jsonl(p)]
    if len(configs) > 1:
        sheet = load_sheet()
        lines += ["## Judge configurations (turn 5, test)", ""]
        for name, path, rows in configs:
            checked = [dict(r, checker_pass=check_reply(r["reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present)
                       for r in rows if r.get("judge_label") != "ERROR"]
            model = rows[0].get("judge_model", "?")
            lines += [f"### {name}", "", f"`{path}`, judge model `{model}`, rubric `{rows[0].get('judge_rubric', 'v1')}`.", "",
                      judge_markdown(rows), "", checker_judge_markdown(checked), ""]
    # Agreement uses only replies of the systems in this evaluation (the judge-selection population).
    in_dir = {c["model"] for c in (read_json(d / "summary_test.json") or {}).get("cells", [])}
    ref = {r["dialogue_id"]: r["judge_label"] for r in read_jsonl(a.reference) if r["model"] in in_dir} if a.reference else {}
    if ref and configs:
        panel_model = read_jsonl(a.reference)[0].get("judge_model", "")
        lines += ["## Judge agreement with the LLM reference panel (not human labels)", "",
                  f"Reference = majority of three models ({panel_model.removeprefix('panel:')}), neutral rubric from the "
                  f"pre-registered definitions. {len(ref)} reference rows.", "",
                  "| judge | n | agreement | Cohen's kappa |", "|---|---|---|---|"]
        for name, _, rows in configs:
            pairs_ = [(ref[r["dialogue_id"]], r["judge_label"]) for r in rows if r["dialogue_id"] in ref and r["judge_label"] != "ERROR"]
            rep_ = agreement_report(pairs_)
            lines.append(f"| {name} | {rep_.get('n', 0)} | {fmt(rep_.get('agreement'))} | {fmt(rep_.get('cohen_kappa'))} |")
        lines.append("")
        if a.panel:
            per = [{r["dialogue_id"]: r["judge_label"] for r in read_jsonl(f) if r["model"] in in_dir and r["judge_label"] != "ERROR"}
                   for f in a.panel]
            common = sorted(set.intersection(*(set(p) for p in per)))
            if common:
                lines += [f"Panel agreement on these replies: Fleiss' kappa {fleiss_kappa([[p[k] for p in per] for k in common]):.3f} "
                          f"over {len(common)} replies labelled by all {len(per)} panel models.", ""]
    sheets = [read_csv(p) for p in a.labels if Path(p).exists()]
    if sheets:
        lines += human_label_lines(sheets, configs)

    if a.helpful:
        panels = [read_jsonl(f) for f in a.helpful]
        models = ", ".join(rows[0]["judge_model"].split("/")[-1] for rows in panels if rows)
        lines += ["## Helpfulness at turn 5 (LLM panel majority, not human labels)", "",
                  f"Does the reply give the help the customer asked for, regardless of warnings? Panel: {models}. "
                  "Interval: Wilson 95%. p: exact McNemar against the same base model, paired by dialogue "
                  "(descriptive, not corrected).", "",
                  "| system | disclosure products: n | helpful [95% CI] | p vs base | control products: n | helpful [95% CI] | p vs base |",
                  "|---|---|---|---|---|---|---|"]
        disc = helpful_votes([[r for r in rows if r["requires_disclosure"]] for rows in panels])
        ctrl = helpful_votes([[r for r in rows if not r["requires_disclosure"]] for rows in panels])

        def cell(votes: dict, m: str) -> str:
            if m not in votes:
                return "0 | n/a | n/a"
            n, k = len(votes[m]), sum(votes[m].values())
            lo, hi = wilson(k, n)
            base = "llama_base" if m.startswith("llama") else "base"
            p = f"{helpful_vs_base(votes, m, base)[1]:.2g}" if m != base and base in votes else "n/a"
            return f"{n} | {100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] | {p}"

        for m in sorted(set(disc) | set(ctrl)):
            lines.append(f"| {m} | {cell(disc, m)} | {cell(ctrl, m)} |")
        lines.append("")

    if a.auditkit_dir and a.auditkit_dir.exists():
        rows_ = [(p.stem.removeprefix("cmp_"), read_json(p)) for p in sorted(a.auditkit_dir.glob("cmp_*.json"))]
        rows_ = [(n, c) for n, c in rows_ if c["significance"].get("p_value") is not None]
        if rows_:
            lines += ["## AuditKIT paired bootstrap as shipped, next to McNemar", "",
                      f"Same transcripts, recomputed locally with AuditKIT {rows_[0][1]['significance'].get('source', '')}. "
                      "AuditKIT 1.1.3 does not center the resample under the null (`docs/UPSTREAM.md` issue 1), so its p "
                      "stays near 0.5 whatever the effect. Decisions use McNemar.", "",
                      "| comparison | diff (points) | AuditKIT p | AuditKIT significant | McNemar p |", "|---|---|---|---|---|"]
            for n, c in rows_:
                sig = c["significance"]
                lines.append(f"| {n} | {100 * c['diff']:+.1f} | {sig['p_value']:.3f} | {sig['significant']} | {sig['mcnemar_p']:.2g} |")
            lines.append("")

    pairs = read_jsonl(a.pairs)
    if pairs:
        prof = pair_profile(pairs, load_sheet())
        lines += ["## DPO pair mix (training rows)", "",
                  f"From `{a.pairs}`, {len(pairs)} rows. Markers are warning phrases found by the rule checker.", "",
                  "| pair type | n | words chosen | words rejected | markers chosen | markers rejected | checker pass chosen | checker pass rejected |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in sorted(prof):
            v = prof[t]
            lines.append(f"| {t} | {v['n']} | {v['words_chosen']:.0f} | {v['words_rejected']:.0f} | {v['markers_chosen']:.2f} | "
                         f"{v['markers_rejected']:.2f} | {v['checker_pass_chosen']} | {v['checker_pass_rejected']} |")
        lines.append("")

    exps = read_jsonl(a.experiments)
    if exps:
        lines += ["## Experiment log", "", "| run | system | recipe | passed | C1 | C2 | C3 | C4 | C5 | C6 | C5 amended (not counted) | decision |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in exps:
            recipe = ", ".join(f"{k}={v}" for k, v in e.get("recipe", {}).items())
            crit = e.get("criteria", {})
            amended = {True: "pass", False: "fail", None: "n/a"}[e.get("values", {}).get("c5_amended_pass")]
            marks = " | ".join({True: "pass", False: "fail", None: "n/a"}[crit.get(c)] for c in CRITERIA)
            lines.append(f"| {e.get('run')} | {e.get('candidate', '?')} | {recipe} | {e.get('passed')}/{e.get('max_score')} | {marks} | {amended} | {e.get('decision')} |")
        best = max(exps, key=lambda e: e.get("passed", -1))
        v = best.get("values", {})
        lines += [
            "",
            f"Best run: {best.get('run')} ({best.get('passed')}/{best.get('max_score')}).",
            "",
            "| quantity | base | candidate |",
            "|---|---|---|",
            f"| DR@5(test) | {fmt(v.get('dr5_test_base'))} | {fmt(v.get('dr5_test_candidate'))} |",
            f"| DR@5(heldout) | {fmt(v.get('dr5_heldout_base'))} | {fmt(v.get('dr5_heldout_candidate'))} |",
            f"| OVER-CORRECT rate | {fmt(v.get('over_correct_base'))} | {fmt(v.get('over_correct_candidate'))} |",
            f"| control warning rate | {fmt(v.get('warning_rate_base'))} | {fmt(v.get('warning_rate_candidate'))} |",
            f"| script gap (en - hinglish) | {fmt(v.get('script_gap_base'))} | {fmt(v.get('script_gap_candidate'))} |",
            f"| DR@1 | {fmt(v.get('dr1_base'))} | {fmt(v.get('dr1_candidate'))} |",
            f"| McNemar p (DR@5 test) | | {fmt(v.get('mcnemar_p'))} |",
            "",
        ]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
