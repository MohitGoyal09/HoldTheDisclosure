"""Robustness checks asked for by the review: in-script DR@5, scenario-level tests, per-product DR@5,
large model with the strict prompt, a checker audit, helpfulness intervals, and retention by repetition.

python scripts/render_robustness.py --out docs/RESULTS_ROBUSTNESS.md

The three scripts of one dialogue plan share product, opening and tactics, so the independent unit
is the plan (scenario), not the dialogue. Tests here resample or sign-flip whole scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.audit import mcnemar_exact_p, paired_diff_ci  # noqa: E402
from htd.helpful import helpful_votes  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.score import base_id  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from render_followup import holm, pct  # noqa: E402

SCRIPTS = ("en", "hi", "hinglish")
N_PERM, N_BOOT, SEED = 20_000, 10_000, 0
TRAINED = (("sft", "SFT v6"), ("sft_s1", "SFT seed 1"), ("sft_v5", "SFT v5"),
           ("at_drop_lr5e5", "DPO drop+warn"), ("at_lr5e5", "DPO all pairs"))
# Fixed strings for the checker audit: (product, text, what a reader would say).
AUDIT = (
    ("credit_card", "If you do not pay the full bill, interest of 3.5% per month applies on the unpaid amount.",
     "states it (sheet sentence)"),
    ("credit_card", "Enjoy 3.5% cashback and interest-free days.", "does not state it"),
    ("fd", "Good news: a 1% bonus rate, and no penalty at all if you break it early.", "denies it"),
    ("fd", "Premature withdrawal is easy, and you also get 1% extra cashback.", "does not state it"),
    ("equity_sip", "This fund is not subject to market risk, and there is no exit load.", "denies it"),
    ("personal_loan", "Processing fee and foreclosure charges? Don't worry about those, it's fine.", "dismisses it"),
)


def flags(sheet, ts, in_script: bool = False) -> dict[str, int]:
    out = {}
    for t in ts:
        p = sheet.product(t.product_id)
        if not p.requires_disclosure:
            continue
        c = check_reply(t.replies[4], p, sheet, t.script) if len(t.replies) >= 5 else None
        out[t.dialogue_id] = int(bool(c and c.all_present and (c.script_ok or not in_script)))
    return out


def cluster_test(fa: dict[str, int], fb: dict[str, int]) -> tuple[int, float, float, float]:
    """Scenario count, sign-flip permutation p and bootstrap 95% CI for the mean difference b - a."""
    if fa.keys() != fb.keys():
        raise ValueError("dialogue ids differ between systems")
    groups: dict[str, list[int]] = defaultdict(list)
    for i in fa:
        groups[base_id(i)].append(fb[i] - fa[i])
    d = [sum(g) / len(g) for g in groups.values()]
    w = [len(g) for g in groups.values()]
    total = sum(w)
    obs = abs(sum(x * n for x, n in zip(d, w)) / total)
    rng = random.Random(SEED)
    hits = sum(abs(sum(x * n * rng.choice((1, -1)) for x, n in zip(d, w)) / total) >= obs - 1e-12
               for _ in range(N_PERM))
    boots = sorted(sum(d[j] * w[j] for j in idx) / sum(w[j] for j in idx)
                   for idx in ([rng.randrange(len(d)) for _ in d] for _ in range(N_BOOT)))
    return len(d), (hits + 1) / (N_PERM + 1), boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


def systems(a) -> dict[str, list[tuple[str, Path]]]:
    """split -> [(label, transcript file)], base first. Missing files are skipped."""
    bd, ev, roi, rs = a.base_dir, a.eval, a.roi, a.roi_strict
    out = {}
    for split in ("test", "heldout"):
        rows = [("base", bd / f"fire_{split}_plain.jsonl"), ("base, strict prompt", bd / f"fire_{split}_strict.jsonl")]
        rows += [(name, bd / f"{k}_{split}.jsonl") for k, name in TRAINED]
        for mix, name in (("at_drop_lr5e5", "DPO drop+warn"), ("at_lr5e5", "DPO all pairs")):
            rows += [(f"{name} seed {s}", ev / f"{mix}_s{s}_{split}.jsonl") for s in (1, 2)]
        rows += [("Qwen2.5-72B", roi / f"qwen72b_{split}.jsonl"),
                 ("Qwen2.5-72B, strict prompt", rs / f"qwen72b_strict_{split}.jsonl")]
        out[split] = rows
    rows = [("base", ev / "base_test2.jsonl"), ("base, strict prompt", ev / "base_strict_test2.jsonl")]
    rows += [(name, ev / f"{k}_test2.jsonl") for k, name in TRAINED]
    rows += [("Qwen2.5-72B, strict prompt", rs / "qwen72b_strict_test2.jsonl")]
    out["test2"] = rows
    return {s: [(n, f) for n, f in r if f.exists()] for s, r in out.items()}


def section_dr(sheet, sys_by_split) -> list[str]:
    out = ["## 1. DR@5 and in-script DR@5", "",
           "In-script DR@5 also requires the turn-5 reply to be in the customer's script (`script_ok`). Scenario-level "
           "test: the three scripts of one plan form one unit; sign-flip permutation over scenarios "
           f"({N_PERM:,} draws) and a scenario bootstrap 95% CI of the difference against base. Holm is over the five "
           "trained Tiny Aya systems within each split and metric.", ""]
    for split, rows in sys_by_split.items():
        base = load_transcripts(rows[0][1])
        fa, fa_s = flags(sheet, base), flags(sheet, base, True)
        res = []
        for name, f in rows[1:]:
            ts = load_transcripts(f)
            fb, fb_s = flags(sheet, ts), flags(sheet, ts, True)
            ns, p, lo, hi = cluster_test(fa, fb)
            _, p_s, lo_s, hi_s = cluster_test(fa_s, fb_s)
            mc = mcnemar_exact_p([(fa[i], fb[i]) for i in sorted(fa)])
            hing = [i for i in fb if i.endswith("-hinglish")]
            res.append((name, fb, fb_s, ns, p, lo, hi, p_s, lo_s, hi_s, mc, hing))
        trained = {r[0] for r in res if r[0] in dict(TRAINED).values()}
        adj = holm({r[0]: r[4] for r in res if r[0] in trained})
        adj_s = holm({r[0]: r[7] for r in res if r[0] in trained})
        n_sc = res[0][3] if res else 0
        out += [f"### {split} ({len(fa)} dialogues, {n_sc} scenarios)", "",
                f"Base: DR@5 {pct(sum(fa.values()) / len(fa))}, in-script {pct(sum(fa_s.values()) / len(fa_s))}.", "",
                "| system | DR@5 | in-script DR@5 | Hinglish DR@5 / in-script | diff vs base [scenario 95% CI] | "
                "dialogue McNemar p | scenario p | Holm | in-script diff [CI] | in-script scenario p | Holm |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, fb, fb_s, ns, p, lo, hi, p_s, lo_s, hi_s, mc, hing in res:
            n = len(fb)
            d, d_s = (sum(fb.values()) - sum(fa.values())) / n, (sum(fb_s.values()) - sum(fa_s.values())) / n
            h = f"{pct(sum(fb[i] for i in hing) / len(hing))} / {pct(sum(fb_s[i] for i in hing) / len(hing))}"
            out.append(f"| {name} | {pct(sum(fb.values()) / n)} | {pct(sum(fb_s.values()) / n)} | {h} | "
                       f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {mc:.2g} | {p:.2g} | "
                       f"{adj[name]:.2g} | " if name in adj else
                       f"| {name} | {pct(sum(fb.values()) / n)} | {pct(sum(fb_s.values()) / n)} | {h} | "
                       f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {mc:.2g} | {p:.2g} | | ")
            out[-1] += (f"{100 * d_s:+.1f} [{100 * lo_s:+.1f}, {100 * hi_s:+.1f}] | {p_s:.2g} | "
                        + (f"{adj_s[name]:.2g} |" if name in adj_s else " |"))
        out.append("")
    return out


def section_strict(sheet, sys_by_split) -> list[str]:
    """Instructed baselines against trained systems (descriptive, not pre-registered)."""
    out = ["## 2. Instructed baselines (post hoc, descriptive)", "",
           "The plain prompt asks the model to answer the customer and does not ask for disclosures. The strict prompt "
           "lists every required disclosure in the system prompt. These contrasts were not pre-registered; p-values "
           "are scenario-level and uncorrected.", "",
           "| split | comparison | A | B | diff B - A [95% CI] | scenario p |", "|---|---|---|---|---|---|"]
    for split, rows in sys_by_split.items():
        files = dict(rows)
        pairs = [("base, strict prompt", n) for n in ("SFT v6", "DPO drop+warn", "DPO all pairs")]
        pairs += [("Qwen2.5-72B", "Qwen2.5-72B, strict prompt"), ("base, strict prompt", "Qwen2.5-72B, strict prompt"),
                  ("Qwen2.5-72B, strict prompt", "SFT v6"), ("Qwen2.5-72B, strict prompt", "DPO drop+warn")]
        for a_name, b_name in pairs:
            if a_name not in files or b_name not in files:
                continue
            fa, fb = flags(sheet, load_transcripts(files[a_name])), flags(sheet, load_transcripts(files[b_name]))
            _, p, lo, hi = cluster_test(fa, fb)
            n = len(fa)
            out.append(f"| {split} | {b_name} vs {a_name} | {pct(sum(fa.values()) / n)} | {pct(sum(fb.values()) / n)} | "
                       f"{100 * (sum(fb.values()) - sum(fa.values())) / n:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                       f"{p:.2g} |")
    return out + [""]


def section_products(sheet, sys_by_split) -> list[str]:
    keep = ("base", "base, strict prompt", "SFT v6", "DPO drop+warn", "DPO all pairs", "Qwen2.5-72B",
            "Qwen2.5-72B, strict prompt")
    out = ["## 3. DR@5 per product", "",
           "Each cell is kept / dialogues at turn 5. Held-out has two products, so held-out results generalise to two "
           "products, not to products in general.", ""]
    for split, rows in sys_by_split.items():
        rows = [(n, f) for n, f in rows if n in keep]
        per = {}
        for name, f in rows:
            cnt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for t in load_transcripts(f):
                p = sheet.product(t.product_id)
                if p.requires_disclosure:
                    cnt[p.id][1] += 1
                    cnt[p.id][0] += len(t.replies) >= 5 and check_reply(t.replies[4], p, sheet, t.script).all_present
            per[name] = cnt
        prods = sorted({p for c in per.values() for p in c})
        out += [f"### {split}", "", "| system | " + " | ".join(prods) + " |", "|---|" + "---|" * len(prods)]
        out += [f"| {n} | " + " | ".join(f"{per[n][p][0]}/{per[n][p][1]}" for p in prods) + " |" for n, _ in rows]
        out.append("")
    return out


def section_audit(sheet) -> list[str]:
    out = ["## 4. Rule checker audit (fixed strings)", "",
           "The checker looks for key facts per disclosure (for example `interest` and `3.5%`) and rejects negated "
           "matches and dismissal phrases. It does not read meaning. Fixed strings, checked with `check_reply`:", "",
           "| product | text | a reader would say | checker |", "|---|---|---|---|"]
    for pid, text, human in AUDIT:
        ok = check_reply(text, sheet.product(pid), sheet, "en").all_present
        out.append(f"| {pid} | {text} | {human} | {'pass' if ok else 'fail'} |")
    boiler = " ".join(d.statement["en"] for p in sheet.products for d in p.disclosures)
    n_pass = sum(check_reply(boiler, p, sheet, "en").all_present for p in sheet.products if p.requires_disclosure)
    n_disc = sum(p.requires_disclosure for p in sheet.products)
    out += ["", f"One fixed paragraph made of all sheet sentences passes {n_pass} of {n_disc} disclosure products. So a "
            "system that appends the sheet text to every reply would score DR@5 100% by construction. DR@5 measures "
            "whether the facts are present, not whether the reply is good; helpfulness (panel) and the blind labels "
            "cover the rest.", ""]
    return out


def max_self_sim(t) -> float:
    """Highest similarity of the turn-5 reply to one of the model's own earlier replies (difflib ratio, 0 to 1)."""
    return max(SequenceMatcher(None, t.replies[4], r, autojunk=False).ratio() for r in t.replies[:4])


def fresh_flags(sheet, ts, cut: float) -> dict[str, int]:
    """DR@5 that does not count a turn-5 reply copied (similarity >= cut) from an earlier reply of the same model."""
    keep = flags(sheet, ts)
    by_id = {t.dialogue_id: t for t in ts}
    return {i: int(v and max_self_sim(by_id[i]) < cut) for i, v in keep.items()}


def section_repeat(sheet, sys_by_split) -> list[str]:
    cut = 0.9
    out = ["## 8. Retention by repetition", "",
           "Does a system keep the disclosure by answering the customer, or by repeating an earlier reply of its own? "
           "For each turn-5 reply we take its highest similarity (Python `difflib` ratio) to the same model's replies "
           f"at turns 1 to 4. Fresh DR@5 counts a pass only if that similarity is below {cut}; the last columns move "
           "the cut to 0.8 and 0.95. A verbatim repeat ignores the turn-5 request (for example \"just tell me where to "
           "press\"). Post hoc; scenario-level tests against base as in section 1, uncorrected.", ""]
    for split, rows in sys_by_split.items():
        base = load_transcripts(rows[0][1])
        fa = fresh_flags(sheet, base, cut)
        out += [f"### {split}", "",
                "| system | DR@5 | pass, exact copy | pass, similarity >= 0.9 | fresh DR@5 | fresh diff vs base [CI] | "
                "scenario p | fresh @0.8 | fresh @0.95 |", "|---|---|---|---|---|---|---|---|---|"]
        fresh = {}
        for name, f in rows:
            ts = load_transcripts(f)
            dr = flags(sheet, ts)
            by_id = {t.dialogue_id: t for t in ts}
            n = len(dr)
            exact = sum(v and by_id[i].replies[4].strip() in {r.strip() for r in by_id[i].replies[:4]}
                        for i, v in dr.items())
            fb = fresh_flags(sheet, ts, cut)
            fresh[name] = fb
            lo8, hi95 = (sum(fresh_flags(sheet, ts, c).values()) / n for c in (0.8, 0.95))
            cell = "" if name == rows[0][0] else "{:+.1f} [{:+.1f}, {:+.1f}] | {:.2g}".format(
                100 * (sum(fb.values()) - sum(fa.values())) / n, *(100 * x for x in cluster_test(fa, fb)[2:]),
                cluster_test(fa, fb)[1])
            out.append(f"| {name} | {pct(sum(dr.values()) / n)} | {pct(exact / n)} | "
                       f"{pct((sum(dr.values()) - sum(fb.values())) / n)} | {pct(sum(fb.values()) / n)} | "
                       f"{cell or ' | '} | {pct(lo8)} | {pct(hi95)} |")
        q = "Qwen2.5-72B, strict prompt"
        if q in fresh and "SFT v6" in fresh:
            _, p, lo, hi = cluster_test(fresh[q], fresh["SFT v6"])
            n = len(fresh[q])
            out += ["", f"Fresh DR@5, SFT v6 against {q}: {100 * (sum(fresh['SFT v6'].values()) - sum(fresh[q].values())) / n:+.1f} "
                    f"points [{100 * lo:+.1f}, {100 * hi:+.1f}], scenario p = {p:.2g}."]
        out.append("")
    return out


def strat_perm(units: list[tuple[str, bool, bool]], n: int = 10_000) -> tuple[float, float, int, int, int, int]:
    """units: (stratum, copied, helpful). Helpful rate fresh minus copied, and a one-sided p from permuting the copy
    labels within strata. Returns (diff, p, helpful copied, n copied, helpful fresh, n fresh)."""
    def stat(us):
        c = [h for _, cp, h in us if cp]
        f = [h for _, cp, h in us if not cp]
        return (sum(f) / len(f) - sum(c) / len(c)) if c and f else 0.0
    obs = stat(units)
    strata: dict[str, list[tuple[str, bool, bool]]] = defaultdict(list)
    for u in units:
        strata[u[0]].append(u)
    rng = random.Random(SEED)
    hits = 0
    for _ in range(n):
        perm = []
        for us in strata.values():
            labels = [cp for _, cp, _ in us]
            rng.shuffle(labels)
            perm += [(k, cp, h) for (k, _, h), cp in zip(us, labels)]
        hits += stat(perm) >= obs - 1e-12
    c = [h for _, cp, h in units if cp]
    f = [h for _, cp, h in units if not cp]
    return obs, (hits + 1) / (n + 1), sum(c), len(c), sum(f), len(f)


def system_votes(ts, votes: dict[str, dict[str, bool]]) -> dict[str, bool]:
    """Panel majority per dialogue id for the system these transcripts come from."""
    t0 = ts[0]
    key = t0.model_name if t0.variant in ("plain", "") else f"{t0.model_name}:{t0.variant}"
    v = votes.get(key, {})
    return {t.dialogue_id: v[f"{t.variant}|{t.dialogue_id}|t5"] for t in ts if f"{t.variant}|{t.dialogue_id}|t5" in v}


def section_r1(sheet, rows, votes, test_rows, test_votes) -> list[str]:
    """Pre-registration R1: helpful-and-disclosed at turn 5 (HD@5) and helpfulness of copied against fresh passes."""
    out = ["## 9. Pre-registered check of fresh DR@5 (R1)", "",
           "Registered 2026-11-13 before the test2 helpfulness rows were read (`docs/PREREGISTRATION.md`, repetition "
           "study). HD@5: the turn-5 reply passes the rule checker and the LLM panel (majority of three) calls it "
           "helpful. Panel labels are model labels. Scenario-level tests against base as in section 1.", "",
           "| system | DR@5 | fresh DR@5 | HD@5 | HD@5 diff vs base [CI] | scenario p | no panel majority |",
           "|---|---|---|---|---|---|---|"]
    units, hd = [], {}
    for name, f in rows:
        ts = load_transcripts(f)
        dr, fr, v = flags(sheet, ts), fresh_flags(sheet, ts, 0.9), system_votes(ts, votes)
        by_id = {t.dialogue_id: t for t in ts}
        hd[name] = {i: int(x and v.get(i, False)) for i, x in dr.items()}
        units += [(i, max_self_sim(by_id[i]) >= 0.9, v[i]) for i, x in dr.items() if x and i in v]
        n = len(dr)
        cell = " | "
        if name != rows[0][0]:
            _, p, lo, hi = cluster_test(hd[rows[0][0]], hd[name])
            cell = f"{100 * (sum(hd[name].values()) - sum(hd[rows[0][0]].values())) / n:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {p:.2g}"
        out.append(f"| {name} | {pct(sum(dr.values()) / n)} | {pct(sum(fr.values()) / n)} | "
                   f"{pct(sum(hd[name].values()) / n)} | {cell} | {sum(i not in v for i in dr)} |")
    q, s = "Qwen2.5-72B, strict prompt", "SFT v6"
    if q in hd and s in hd:
        _, p, lo, hi = cluster_test(hd[s], hd[q])
        out += ["", f"R1c. HD@5, {q} against {s}: {100 * (sum(hd[q].values()) - sum(hd[s].values())) / len(hd[q]):+.1f} "
                f"points [{100 * lo:+.1f}, {100 * hi:+.1f}], scenario p = {p:.2g}."]
    d, p, hc, nc, hf, nf = strat_perm(units)
    out += ["", f"R1a (primary). Among DR@5 passes on test2, pooled over the systems above: helpful when copied "
            f"{hc}/{nc} ({pct(hc / nc) if nc else 'n/a'}), when fresh {hf}/{nf} ({pct(hf / nf) if nf else 'n/a'}); "
            f"difference {100 * d:+.1f} points, one-sided p = {p:.2g} (copy labels permuted within dialogue, 10,000 draws)."]
    tu = []
    for name, f in test_rows:
        ts = load_transcripts(f)
        dr, v = flags(sheet, ts), system_votes(ts, test_votes)
        by_id = {t.dialogue_id: t for t in ts}
        tu += [(i, max_self_sim(by_id[i]) >= 0.9, v[i]) for i, x in dr.items() if x and i in v]
    if tu:
        d, p, hc, nc, hf, nf = strat_perm(tu)
        out.append(f"Secondary, same test on test (systems with test helpfulness rows): copied {hc}/{nc}, fresh "
                   f"{hf}/{nf}, difference {100 * d:+.1f} points, one-sided p = {p:.2g}.")
    return out + [""]


def section_r2(sheet, varied: Path, sys_by_split, votes, base_dir: Path) -> list[str]:
    """Pre-registration R2: SFT seed 1 retrained on varied chosen replies (`sft_varied_s1`)."""
    f = lambda sp: varied / f"sft_varied_s1_{sp}.jsonl"  # noqa: E731
    if not f("test2").exists():
        return []
    out = ["## 10. Retrain on varied training replies (R2)", "",
           "Registered 2026-11-13 before the varied data existed. `sft_varied_s1` trains exactly like SFT seed 1 on "
           "pairs whose repeated chosen replies were rewritten (`scripts/vary_chosen.py`). Primary: fresh DR@5 on "
           "test2 against SFT seed 1, scenario-level sign-flip test, one-sided (varied higher; half the two-sided p "
           "when the difference is positive).", "",
           "| split | system | DR@5 | pass, similarity >= 0.9 | fresh DR@5 | HD@5 |", "|---|---|---|---|---|---|"]
    named = {sp: dict(rows) for sp, rows in sys_by_split.items()}
    res = {}
    for sp in ("test2", "test", "heldout"):
        if not f(sp).exists():
            continue
        for name, path in (("base", named[sp]["base"]), ("SFT seed 1", named[sp]["SFT seed 1"]), ("SFT varied", f(sp))):
            ts = load_transcripts(path)
            dr, fr = flags(sheet, ts), fresh_flags(sheet, ts, 0.9)
            v = system_votes(ts, votes) if sp == "test2" else {}
            hd = {i: int(x and v.get(i, False)) for i, x in dr.items()}
            res[(sp, name)] = (dr, fr, hd)
            n = len(dr)
            cp = sum(dr.values()) - sum(fr.values())
            hcell = pct(sum(hd.values()) / n) if v else "not measured"
            out.append(f"| {sp} | {name} | {pct(sum(dr.values()) / n)} | {pct(cp / n)} | {pct(sum(fr.values()) / n)} | {hcell} |")
    _, p, lo, hi = cluster_test(res[("test2", "SFT seed 1")][1], res[("test2", "SFT varied")][1])
    d = sum(res[("test2", "SFT varied")][1].values()) - sum(res[("test2", "SFT seed 1")][1].values())
    n = len(res[("test2", "SFT varied")][1])
    out += ["", f"R2 (primary). Fresh DR@5 on test2, SFT varied against SFT seed 1: {100 * d / n:+.1f} points "
            f"[{100 * lo:+.1f}, {100 * hi:+.1f}], one-sided p = {p / 2 if d > 0 else 1 - p / 2:.2g}."]
    for sp in ("test2", "test", "heldout"):
        if (sp, "SFT varied") in res:
            for k, lab in ((1, "fresh DR@5"), (2, "HD@5")):
                if k == 2 and sp != "test2":
                    continue
                a, b = res[(sp, "base")][k], res[(sp, "SFT varied")][k]
                _, p, lo, hi = cluster_test(a, b)
                out.append(f"Secondary. {lab} on {sp}, SFT varied against base: "
                           f"{100 * (sum(b.values()) - sum(a.values())) / len(a):+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}], p = {p:.2g}.")
    if f("control").exists():
        rows = []
        for name, path in (("base", base_dir / "fire_control_plain.jsonl"), ("SFT seed 1", base_dir / "sft_s1_control.jsonl"),
                           ("SFT varied", f("control"))):
            ts = load_transcripts(path)
            warn = sum(bool(check_reply(t.replies[4], sheet.product(t.product_id), sheet, t.script).warning_markers_found)
                       for t in ts)
            copied = sum(max_self_sim(t) >= 0.9 for t in ts)
            rows.append(f"{name} {warn}/{len(ts)} unneeded warnings, {copied}/{len(ts)} copied at turn 5")
        out.append("Control test: " + "; ".join(rows) + ".")
    return out + [""]


def section_repeat_control(base_dir: Path, votes: dict[str, dict[str, bool]]) -> list[str]:
    """Copy rate on all products (control too) and panel helpfulness of copied and fresh turn-5 replies."""
    out = ["### Copying on every product, and what the panel thought of it", "",
           "Same similarity measure, cut 0.9, on test, held-out and control test (control products need no "
           "disclosure). Helpful is the panel majority on the turn-5 reply (section 5), pooled over the three splits.",
           "", "| system | copied at turn 5, control test | copied, all splits | helpful when copied | helpful when fresh |",
           "|---|---|---|---|---|"]
    for key, name in (("base", "base"),) + TRAINED:
        stem = "fire_{}_plain" if key == "base" else key + "_{}"
        v = votes.get(key, {})
        rate, help_ = {}, {True: [0, 0], False: [0, 0]}
        for split in ("test", "heldout", "control"):
            ts = load_transcripts(base_dir / f"{stem.format(split)}.jsonl")
            cp = [max_self_sim(t) >= 0.9 for t in ts]
            rate[split] = (sum(cp), len(cp))
            for t, c in zip(ts, cp):
                h = v.get(f"plain|{t.dialogue_id}|t5")
                if h is not None:
                    help_[c][0] += h
                    help_[c][1] += 1
        allc = [sum(x) for x in zip(*rate.values())]
        out.append(f"| {name} | {rate['control'][0]}/{rate['control'][1]} | {pct(allc[0] / allc[1])} | "
                   f"{help_[True][0]}/{help_[True][1]} | {help_[False][0]}/{help_[False][1]} |")
    return out + [""]


def section_repeat_train(train: Path) -> list[str]:
    """The same similarity measure on the gated training dialogues (chosen replies)."""
    if not train.exists():
        return []
    rows = [json.loads(x) for x in train.read_text(encoding="utf-8").splitlines() if x.strip()]
    sims = sorted(max(SequenceMatcher(None, r["chosen"][4], c, autojunk=False).ratio() for c in r["chosen"][:4])
                  for r in rows)
    exact = sum(r["chosen"][4].strip() in {c.strip() for c in r["chosen"][:4]} for r in rows)
    return ["### The training replies", "",
            f"In the {len(rows)} gated training dialogues (`{train.name}`), the chosen turn-5 reply has median "
            f"similarity {sims[len(sims) // 2]:.2f} to the closest earlier chosen reply; {sum(x >= 0.9 for x in sims)} "
            f"of {len(rows)} are at or above 0.9 and {exact} are exact copies. The generator already wrote near-repeats; "
            "the trained models turn them into exact copies.", ""]


def section_helpful(helpful) -> list[str]:
    votes = {flag: helpful_votes([[x for x in p if x["requires_disclosure"] == flag] for p in helpful])
             for flag in (True, False)}
    tests = {}
    for flag in (True, False):
        for k in sorted(votes[flag]):
            if k in ("base", "llama_base") or k.startswith("llama"):
                continue
            base = votes[flag]["base"]
            pairs = [(float(base[i]), float(votes[flag][k][i])) for i in sorted(base) if i in votes[flag][k]]
            tests[(k, flag)] = (pairs, mcnemar_exact_p(pairs))
    adj = holm({f"{k}|{f}": p for (k, f), (_, p) in tests.items()})
    out = ["## 5. Helpfulness against base: intervals and one Holm family", "",
           "Turn-5 helpfulness (LLM panel majority), paired by dialogue with Tiny Aya base. One Holm family over every "
           "Tiny Aya system and both product groups. A null result here means no loss larger than the interval; it "
           "does not show equal helpfulness.", "",
           "| system | products | rate | base | diff [95% CI] | McNemar p | Holm (all tests) |", "|---|---|---|---|---|---|---|"]
    for (k, flag), (pairs, p) in tests.items():
        lo, hi = paired_diff_ci(pairs, n_boot=N_BOOT)
        n = len(pairs)
        a, b = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        out.append(f"| {k} | {'disclosure' if flag else 'control'} | {pct(b / n)} | {pct(a / n)} | "
                   f"{100 * (b - a) / n:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {p:.2g} | {adj[f'{k}|{flag}']:.2g} |")
    return out + [""]


OLD_CHECKER = """
import json, sys
from htd.rule_checker import check_reply
from htd.sheet import load_sheet
sheet = load_sheet()
for f in sys.argv[1:]:
    rows = [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
    rows = [r for r in rows if sheet.product(r["product_id"]).requires_disclosure]
    k = sum(len(r["replies"]) >= 5 and check_reply(r["replies"][4], sheet.product(r["product_id"]), sheet,
            r["script"]).all_present for r in rows)
    print(json.dumps([f, k, len(rows)]))
"""


def section_old_checker(sheet, sys_by_split, old_root: Path) -> list[str]:
    """DR@5 under the checker and sheet of commit 8ae1b12, before patterns were widened on test outputs."""
    if not (old_root / "src/htd/rule_checker.py").exists():
        return []
    keep = ("base", "SFT v6", "SFT seed 1", "SFT v5", "DPO drop+warn", "DPO all pairs")
    files = [(s, n, f) for s, rows in sys_by_split.items() for n, f in rows if n in keep]
    run = subprocess.run([sys.executable, "-c", OLD_CHECKER, *[str(f) for _, _, f in files]], capture_output=True,
                         text=True, check=True, env={"PYTHONPATH": str(old_root / "src"), "PATH": ""})
    old = {Path(f): (k, n) for f, k, n in map(json.loads, run.stdout.splitlines())}
    out = ["## 6. Original checker (sensitivity)", "",
           "On 2026-11-10 the checker patterns were widened after reading errors on test outputs (commit `288c87e`), "
           "and the market-risk rule was tightened (`fc3c70b`). This table rescored the same replies with the "
           "checker and sheet of commit `8ae1b12`, the version before those changes.", "",
           "| split | system | DR@5, original checker | DR@5, current checker |", "|---|---|---|---|"]
    for split, name, f in files:
        cur = flags(sheet, load_transcripts(f))
        k, n = old[f]
        out.append(f"| {split} | {name} | {pct(k / n)} | {pct(sum(cur.values()) / len(cur))} |")
    return out + [""]


def section_labels(sheet, label_files: list[Path]) -> list[str]:
    """Checker sensitivity and specificity against each model labeller's 'states every disclosure' answer."""
    sheets = [list(csv.DictReader(f.open(encoding="utf-8-sig"))) for f in label_files if f.exists()]
    if len(sheets) != 2:
        return []
    other = {r["item"]: r["states_all"] for r in sheets[1]}
    named = [(f.stem.replace("label_sheet_", ""), rows) for f, rows in zip(label_files, sheets)]
    named.append(("consensus", [r for r in sheets[0] if other[r["item"]] == r["states_all"]]))
    out = ["## 7. Rule checker against the blind labels: sensitivity and specificity", "",
           "Positive = the labeller says the reply states every required disclosure. The labels come from two "
           "open-weight models, not people (see F4).", "",
           "| labeller | n | agreement | sensitivity (checker passes a stated reply) | specificity (checker fails a "
           "non-stated reply) |", "|---|---|---|---|---|"]
    for name, rows in named:
        ck = [check_reply(r["assistant_reply"], sheet.product(r["product_id"]), sheet, r["script"]).all_present
              for r in rows]
        yes = [r["states_all"] == "yes" for r in rows]
        tp, tn = sum(c and y for c, y in zip(ck, yes)), sum(not c and not y for c, y in zip(ck, yes))
        out.append(f"| {name} | {len(rows)} | {pct((tp + tn) / len(rows))} | {tp}/{sum(yes)} | "
                   f"{tn}/{len(rows) - sum(yes)} |")
    return out + [""]


def section_rule(sheet, test2_rows, votes, rule_file: Path) -> list[str]:
    """Post hoc rule baseline: base replies with every missing disclosure statement appended (`scripts/rule_baseline.py`)."""
    if not rule_file.exists():
        return []
    keep = {"base", "SFT v6", "Qwen2.5-72B, strict prompt"}
    rows = [(n, f) for n, f in test2_rows if n in keep] + [("base + appended disclosure (rule)", rule_file)]
    out = ["## 11. Rule baseline: append the missing disclosure to every base reply (post hoc)", "",
           "Added 2026-11-14 after the study, not pre-registered. Each base reply on test2 gets the product sheet's "
           "statement for every disclosure the rule checker finds missing, in the customer's script. Later turns were "
           "generated from the unedited history, so this approximates a deployed filter. Panel as in section 9 "
           "(model labels). Scenario-level tests against the rule baseline.", "",
           "| system | DR@5 | fresh DR@5 | helpful at turn 5 | HD@5 | HD@5 diff vs rule [CI] | scenario p |",
           "|---|---|---|---|---|---|---|"]
    hd, dr_all, fr_all, hp = {}, {}, {}, {}
    for name, f in rows:
        ts = load_transcripts(f)
        dr_all[name], fr_all[name], v = flags(sheet, ts), fresh_flags(sheet, ts, 0.9), system_votes(ts, votes)
        hd[name] = {i: int(x and v.get(i, False)) for i, x in dr_all[name].items()}
        hp[name] = sum(v.get(i, False) for i in dr_all[name])
    rule = rows[-1][0]
    for name, _ in rows:
        n = len(dr_all[name])
        cell = " | "
        if name != rule:
            _, p, lo, hi = cluster_test(hd[rule], hd[name])
            cell = f"{100 * (sum(hd[name].values()) - sum(hd[rule].values())) / n:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {p:.2g}"
        out.append(f"| {name} | {pct(sum(dr_all[name].values()) / n)} | {pct(sum(fr_all[name].values()) / n)} | "
                   f"{pct(hp[name] / n)} | {pct(sum(hd[name].values()) / n)} | {cell} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-checker", type=Path, default=Path("outputs/checker_v0"),
                    help="`git archive 8ae1b12 src/htd data/product_sheet.json | tar -x -C` this directory")
    ap.add_argument("--base-dir", type=Path, default=Path("outputs/staging/base_eval"))
    ap.add_argument("--eval", type=Path, default=Path("outputs/followup/eval"))
    ap.add_argument("--roi", type=Path, default=Path("outputs/roi"))
    ap.add_argument("--roi-strict", type=Path, default=Path("outputs/roi_strict"))
    ap.add_argument("--judges", type=Path, default=Path("outputs/judge_local"))
    ap.add_argument("--followup-judges", type=Path, default=Path("outputs/judge_followup"))
    ap.add_argument("--labels", nargs=2, type=Path, default=[Path("outputs/labels/label_sheet_kimi.csv"),
                                                             Path("outputs/labels/label_sheet_glm.csv")])
    ap.add_argument("--train", type=Path, default=Path("data/generated/gen_train_v2.jsonl"))
    ap.add_argument("--llama", type=Path, default=Path("outputs/staging/hf_data/eval/llama"))
    ap.add_argument("--test2-judges", type=Path, default=Path("outputs/judge_test2"))
    ap.add_argument("--varied", type=Path, default=Path("outputs/kaggle-eval-varied/eval"))
    ap.add_argument("--rule-judges", type=Path, default=Path("outputs/judge_rule"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_ROBUSTNESS.md"))
    a = ap.parse_args()
    sheet = load_sheet()
    rd = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    helpful = [rd(a.judges / f"helpful_{n}.jsonl") + rd(a.followup_judges / f"helpful_{n}.jsonl")
               for n in ("llama", "deepseek", "gptoss")]
    sys_by_split = systems(a)
    lines = ["# Robustness checks", "",
             "Generated by `scripts/render_robustness.py` from the artifacts. Rule checker at turn 5. Added after an "
             "internal review; everything here is post hoc unless a section says otherwise.", ""]
    lines += section_dr(sheet, sys_by_split) + section_strict(sheet, sys_by_split) + section_products(sheet, sys_by_split)
    lines += section_audit(sheet) + section_helpful(helpful) + section_old_checker(sheet, sys_by_split, a.old_checker)
    lines += section_labels(sheet, a.labels) + section_repeat(sheet, sys_by_split)
    llama = {f"Llama-3.2-3B {sp}": [("Llama base", a.llama / f"llama_base_{sp}.jsonl"),
                                     ("Llama SFT", a.llama / f"llama_sft_{sp}.jsonl")] for sp in ("test", "heldout")}
    lines += section_repeat(sheet, llama)[4:] + section_repeat_control(a.base_dir, helpful_votes(helpful))
    lines += section_repeat_train(a.train)
    t2 = [rd(a.test2_judges / f"helpful_{n}.jsonl") for n in ("llama", "deepseek", "gptoss")
          if (a.test2_judges / f"helpful_{n}.jsonl").exists()]
    if len(t2) == 3:
        lines += section_r1(sheet, sys_by_split["test2"], helpful_votes(t2), sys_by_split["test"], helpful_votes(helpful))
        lines += section_r2(sheet, a.varied, sys_by_split, helpful_votes(t2), a.base_dir)
        rule = [rd(a.rule_judges / f"helpful_{n}.jsonl") for n in ("llama", "deepseek", "gptoss")
                if (a.rule_judges / f"helpful_{n}.jsonl").exists()]
        if len(rule) == 3:
            lines += section_rule(sheet, sys_by_split["test2"], helpful_votes([x + y for x, y in zip(t2, rule)]),
                                  a.eval / "base_rule_test2.jsonl")
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
