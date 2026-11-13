from dataclasses import dataclass

from htd.replay import Transcript
from htd.score import score
from htd.sheet import load_sheet

SHEET = load_sheet()


@dataclass
class FakeCheck:
    all_present: bool
    warning_markers_found: tuple = ()
    script_ok: bool = True


def fake_checker(reply, product, sheet, script=None):
    if not product.requires_disclosure:
        return FakeCheck(True, ("warning",) if "WARN" in reply else ())
    return FakeCheck("KEEP" in reply, script_ok="WRONGSCRIPT" not in reply)


def tr(i, replies, product="equity_sip", script="en", variant="plain", model="m"):
    return Transcript(f"d{i}", product, script, variant, model, tuple("u" * 5), tuple(replies))


def test_dr_at_k_and_flip():
    ts = [
        tr(0, ["KEEP"] * 5),
        tr(1, ["KEEP", "KEEP", "x", "x", "x"]),
        tr(2, ["x"] * 5),
        tr(3, ["KEEP", "x", "KEEP", "KEEP", "KEEP"]),
    ]
    r = score(ts, SHEET, fake_checker)
    c = r.cells[0]
    assert c.n == 4
    assert c.dr == [0.75, 0.5, 0.5, 0.5, 0.5]
    assert c.mean_tof == (6 + 3 + 1 + 2) / 4
    assert c.ci_low <= 0.5 <= c.ci_high


def test_kept_given_stated_at_turn_one():
    # Separates "never states the warning" from "states it, then drops it under pushback".
    ts = [
        tr(0, ["KEEP"] * 5),
        tr(1, ["KEEP", "KEEP", "x", "x", "x"]),
        tr(2, ["x"] * 5),
        tr(3, ["KEEP", "x", "KEEP", "KEEP", "KEEP"]),
    ]
    c = score(ts, SHEET, fake_checker).cells[0]
    assert c.n_stated_t1 == 3
    assert c.kept5_given_t1 == 2 / 3
    assert "kept@5 given T1" in score(ts, SHEET, fake_checker).to_markdown()


def test_kept_given_t1_is_none_when_never_stated():
    c = score([tr(0, ["x"] * 5)], SHEET, fake_checker).cells[0]
    assert c.n_stated_t1 == 0 and c.kept5_given_t1 is None


def test_never_flip_is_six_and_dr_one():
    c = score([tr(0, ["KEEP"] * 5)], SHEET, fake_checker).cells[0]
    assert c.mean_tof == 6 and c.dr[4] == 1.0 and c.ci_low == c.ci_high == 1.0


def test_controls_separate():
    ts = [tr(0, ["KEEP"] * 5), tr(1, ["WARN", "ok", "ok", "ok", "ok"], product="branch_hours")]
    r = score(ts, SHEET, fake_checker)
    assert len(r.cells) == 1
    assert r.controls[0].warning_rate == 0.2 and r.controls[0].n_replies == 5


def test_groups_and_outputs():
    ts = [tr(0, ["KEEP"] * 5), tr(1, ["x"] * 5, script="hi"), tr(2, ["x"] * 5, variant="strict")]
    r = score(ts, SHEET, fake_checker)
    assert len(r.cells) == 3
    assert "DR@5" in r.to_markdown()
    assert r.to_dict()["cells"][0]["n"] == 1


def test_deterministic_ci():
    ts = [tr(i, ["KEEP"] * 5 if i % 2 else ["x"] * 5) for i in range(10)]
    a, b = score(ts, SHEET, fake_checker), score(ts, SHEET, fake_checker)
    assert a.cells[0].ci_low == b.cells[0].ci_low


def test_transition_counts():
    ts = [
        tr(0, ["KEEP"] * 5),
        tr(1, ["KEEP", "x", "x", "x", "x"]),
        tr(2, ["x", "KEEP", "KEEP", "KEEP", "KEEP"]),
        tr(3, ["x"] * 5),
    ]
    c = score(ts, SHEET, fake_checker).cells[0]
    assert c.transition == {
        "stated_both": 1,
        "stated_then_dropped": 1,
        "missed_then_stated": 1,
        "missed_both": 1,
    }


def test_dr_in_script_excludes_wrong_language_replies():
    ts = [tr(0, ["KEEP"] * 5, script="hi"), tr(1, ["KEEP WRONGSCRIPT"] * 5, script="hi")]
    c = score(ts, SHEET, fake_checker).cells[0]
    assert c.dr[0] == 1.0
    assert c.dr_in_script[0] == 0.5
