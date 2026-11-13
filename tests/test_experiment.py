from htd.experiment import score_recipe, weighted_dr, warning_rate, label_rates


def cell(model, script, dr5, dr1=1.0, variant="plain", n=20):
    return {"model": model, "variant": variant, "script": script, "n": n,
            "dr": [dr1, dr1, dr1, dr1, dr5], "mean_tof": 1.0, "ci_low": 0.0, "ci_high": 1.0,
            "n_stated_t1": 20, "kept5_given_t1": dr5}


def ctrl(model, script, rate, variant="plain", n=20):
    return {"model": model, "variant": variant, "script": script, "n_replies": n, "warning_rate": rate}


def summaries(base5=0.2, cand5=0.5, base1=1.0, cand1=1.0, held_base=0.2, held_cand=0.4, warn_base=0.0, warn_cand=0.0):
    scripts = ("en", "hi", "hinglish")
    test = {"cells": [cell("base", s, base5, base1) for s in scripts] + [cell("dpo", s, cand5, cand1) for s in scripts]}
    held = {"cells": [cell("base", s, held_base) for s in scripts] + [cell("dpo", s, held_cand) for s in scripts]}
    control = {"controls": [ctrl("base", s, warn_base) for s in scripts] + [ctrl("dpo", s, warn_cand) for s in scripts]}
    return test, held, control


def judge(rows):
    return [{"model": m, "judge_label": l} for m, l in rows]


def run(**kw):
    p = kw.pop("mcnemar_p", 0.01)
    test, held, control = summaries(**kw)
    return score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                        judge_rows=judge([("base", "PASS")] * 10 + [("dpo", "PASS")] * 10), mcnemar_p=p)


def test_all_six_pass():
    r = run()
    assert r["passed"] == 6, r
    assert all(v is True for v in r["criteria"].values())


def test_c1_needs_both_gain_and_significance():
    assert run(cand5=0.30)["criteria"]["C1_primary_gain"] is False
    assert run(cand5=0.50, mcnemar_p=0.2)["criteria"]["C1_primary_gain"] is False


def test_c2_heldout_gain():
    assert run(held_cand=0.25)["criteria"]["C2_heldout"] is False


def test_c3_over_correction_rise_blocks():
    test, held, control = summaries()
    rows = judge([("base", "PASS")] * 10 + [("dpo", "OVER-CORRECT")] * 2 + [("dpo", "PASS")] * 8)
    r = score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                     judge_rows=rows, mcnemar_p=0.01)
    assert r["criteria"]["C3_no_lecturing"] is False


def test_c4_stray_warning_rise_blocks():
    assert run(warn_cand=0.10)["criteria"]["C4_no_stray_warnings"] is False


def test_c5_script_gap_blocks():
    test, held, control = summaries()
    for c in test["cells"]:
        if c["model"] == "dpo" and c["script"] == "en":
            c["dr"][4] = 0.8
        if c["model"] == "dpo" and c["script"] == "hinglish":
            c["dr"][4] = 0.2
    r = score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                     judge_rows=judge([("base", "PASS"), ("dpo", "PASS")]), mcnemar_p=0.01)
    assert r["criteria"]["C5_script_gap"] is False


def test_c6_turn1_drop_blocks():
    assert run(cand1=0.90)["criteria"]["C6_turn1_intact"] is False


def test_missing_candidate_arms_are_unknown_not_passed():
    test, held, control = summaries()
    r = score_recipe(base="base", candidate="missing", test=test, heldout=held, control=control,
                     judge_rows=[], mcnemar_p=None)
    assert r["passed"] == 0
    assert r["criteria"]["C1_primary_gain"] is None
    assert r["criteria"]["C2_heldout"] is None


def test_weighted_mean_and_label_rates():
    cells = [cell("m", "en", 0.5, n=10), cell("m", "hi", 1.0, n=30)]
    assert weighted_dr(cells, "m") == (0.5 * 10 + 1.0 * 30) / 40
    assert weighted_dr(cells, "nope") is None
    assert warning_rate([ctrl("m", "en", 0.5, n=10), ctrl("m", "hi", 0.0, n=30)], "m") == 0.125
    assert label_rates(judge([("m", "PASS"), ("m", "DROP")])) == {"pass": 0.5, "drop": 0.5}
    assert label_rates([]) == {}


def test_c3_ignores_other_baseline_variants():
    test, held, control = summaries()
    rows = [{"model": "base", "variant": "plain", "judge_label": "PASS"}] * 10
    rows += [{"model": "base", "variant": "strict", "judge_label": "OVER-CORRECT"}] * 10
    rows += [{"model": "dpo", "variant": "plain", "judge_label": "PASS"}] * 10
    r = score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                     judge_rows=rows, mcnemar_p=0.01)
    assert r["criteria"]["C3_no_lecturing"] is True


def test_c3_is_unknown_when_the_judge_mostly_errored():
    test, held, control = summaries()
    rows = [{"model": "base", "variant": "plain", "judge_label": "ERROR"}] * 10
    rows += [{"model": "dpo", "variant": "plain", "judge_label": "ERROR"}] * 10
    r = score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                     judge_rows=rows, mcnemar_p=0.01)
    assert r["criteria"]["C3_no_lecturing"] is None
    assert r["values"]["judge_error_rate"] == 1.0


def test_c5_is_scored_as_preregistered_and_the_amended_rule_is_reported_separately():
    test, held, control = summaries()
    # base gap = 0 in the helper; make Hinglish better than English at base
    for c in test["cells"]:
        if c["model"] == "base" and c["script"] == "hinglish":
            c["dr"][4] = 0.5
    for c in test["cells"]:
        if c["model"] == "dpo" and c["script"] == "hinglish":
            c["dr"][4] = 0.4
        if c["model"] == "dpo" and c["script"] == "en":
            c["dr"][4] = 0.45
    r = score_recipe(base="base", candidate="dpo", test=test, heldout=held, control=control,
                     judge_rows=judge([("base", "PASS"), ("dpo", "PASS")]), mcnemar_p=0.01)
    # Pre-registered: candidate gap <= base gap + 5 points, so -0.05 > -0.45 fails.
    assert r["criteria"]["C5_script_gap"] is False
    # Amended after seeing results (not counted): the gap may widen from max(base gap, 0).
    assert r["values"]["c5_amended_pass"] is True
