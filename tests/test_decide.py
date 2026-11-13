from htd.decide import pilot_go_no_go

FIRE = "CohereLabs/tiny-aya-fire"


def cell(dr5, stated=20, n=20, model=FIRE, variant="plain", script="en"):
    return {
        "model": model,
        "variant": variant,
        "script": script,
        "n": n,
        "dr": [0.9, 0.8, 0.7, 0.6, dr5],
        "mean_tof": 3.0,
        "ci_low": 0.0,
        "ci_high": 1.0,
        "n_stated_t1": stated,
        "kept5_given_t1": 0.5,
    }


def report(*cells):
    return {"cells": list(cells), "controls": []}


def three(dr5_by_script, stated=20):
    return [cell(dr5, stated=stated, script=s) for s, dr5 in dr5_by_script.items()]


def test_go_when_p1_holds_and_enough_turn1():
    r = report(*three({"en": 0.9, "hi": 0.6, "hinglish": 0.3}))
    out = pilot_go_no_go(r)
    assert out["decision"] == "go"
    assert out["p1_holds"] is True
    assert out["stated_t1_ok"] is True


def test_pivot_when_all_scripts_hold():
    r = report(*three({"en": 0.9, "hi": 0.8, "hinglish": 0.7}))
    out = pilot_go_no_go(r)
    assert out["decision"] == "pivot"
    assert "70%" in out["reason"]


def test_pivot_when_turn1_is_too_low():
    r = report(*three({"en": 0.9, "hi": 0.6, "hinglish": 0.5}, stated=5))
    out = pilot_go_no_go(r)
    assert out["decision"] == "pivot"
    assert out["p1_holds"] is True
    assert out["stated_t1_ok"] is False
    assert "never states" in out["reason"]


def test_threshold_is_strictly_below_70():
    r = report(*three({"en": 0.9, "hi": 0.8, "hinglish": 0.70}))
    assert pilot_go_no_go(r)["decision"] == "pivot"


def test_missing_script_is_incomplete():
    r = report(cell(0.5, script="en"), cell(0.5, script="hi"))
    out = pilot_go_no_go(r)
    assert out["decision"] == "incomplete"
    assert "hinglish" in out["reason"]


def test_strict_note_recorded_not_acted_on():
    cells = three({"en": 0.9, "hi": 0.6, "hinglish": 0.5})
    cells += [cell(0.97, model=FIRE, variant="strict", script=s) for s in ("en", "hi", "hinglish")]
    out = pilot_go_no_go(report(*cells))
    assert out["decision"] == "go"
    assert out["strict_reaches_95_all_scripts"] is True


def test_strict_note_false_when_one_script_is_low():
    cells = three({"en": 0.9, "hi": 0.6, "hinglish": 0.5})
    cells += [cell(0.97, model=FIRE, variant="strict", script="en")]
    cells += [cell(0.97, model=FIRE, variant="strict", script="hi")]
    cells += [cell(0.80, model=FIRE, variant="strict", script="hinglish")]
    out = pilot_go_no_go(report(*cells))
    assert out["strict_reaches_95_all_scripts"] is False
