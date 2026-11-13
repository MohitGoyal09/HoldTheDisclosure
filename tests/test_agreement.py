import pytest

from htd.agreement import agreement, cohen_kappa, confusion, report


def test_perfect_agreement():
    pairs = [("PASS", "PASS"), ("DROP", "DROP"), ("PASS", "PASS")]
    assert agreement(pairs) == 1.0
    assert cohen_kappa(pairs) == 1.0


def test_no_agreement_and_kappa_below_zero_is_possible():
    pairs = [("PASS", "DROP"), ("DROP", "PASS")]
    assert agreement(pairs) == 0.0
    assert cohen_kappa(pairs) == -1.0


def test_known_kappa_value():
    # 50 items, 20 both PASS, 10 both DROP, 10 human PASS/judge DROP, 10 human DROP/judge PASS
    pairs = [("PASS", "PASS")] * 20 + [("DROP", "DROP")] * 10 + [("PASS", "DROP")] * 10 + [("DROP", "PASS")] * 10
    assert agreement(pairs) == pytest.approx(0.6)
    # po=0.6; human PASS 30/50, judge PASS 30/50 -> pe = .36+.16 = .52; kappa = .08/.48
    assert cohen_kappa(pairs) == pytest.approx(0.08 / 0.48)


def test_empty_inputs():
    assert agreement([]) == 0.0
    assert cohen_kappa([]) == 0.0


def test_confusion_and_report_shape():
    pairs = [("PASS", "PASS"), ("PASS", "DROP"), ("DROP", "DROP")]
    c = confusion(pairs)
    assert c["PASS"]["PASS"] == 1 and c["PASS"]["DROP"] == 1 and c["DROP"]["DROP"] == 1
    r = report(pairs)
    assert r["n"] == 3 and r["per_judge_label"]["PASS"]["n"] == 1
    assert r["per_judge_label"]["DISMISS"]["agreement"] is None


def test_report_drops_rows_with_a_blank_label():
    r = report([("PASS", "PASS"), ("", "DROP"), ("DROP", "")])
    assert r["n"] == 1


def test_majority_needs_two_of_three():
    from htd.agreement import majority

    assert majority(["PASS", "DROP", "PASS"]) == "PASS"
    assert majority(["PASS", "DROP", "DISMISS"]) is None


def test_fleiss_kappa_known_values():
    from htd.agreement import fleiss_kappa

    assert fleiss_kappa([["PASS"] * 3, ["DROP"] * 3]) == 1.0
    k = fleiss_kappa([["PASS", "PASS", "PASS"], ["PASS", "PASS", "DROP"], ["DROP", "DROP", "DROP"]])
    assert abs(k - 0.55) < 1e-3
