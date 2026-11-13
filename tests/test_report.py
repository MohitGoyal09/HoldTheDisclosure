from htd.report import checker_judge_markdown, comparison_title


def jrow(model, label, checker_pass):
    return {"model": model, "judge_label": label, "checker_pass": checker_pass}


def test_checker_judge_counts_false_passes_and_misses():
    rows = [
        jrow("sft", "PASS", True),
        jrow("sft", "DROP", True),
        jrow("sft", "DISMISS", True),
        jrow("sft", "PASS", False),
        jrow("sft", "OVER-CORRECT", True),
        jrow("sft", "ERROR", True),
    ]
    line = [x for x in checker_judge_markdown(rows).splitlines() if x.startswith("| sft")][0]
    cells = [c.strip() for c in line.strip("|").split("|")]
    # system, judged, checker pass, judge PASS, agree, checker pass but judge DROP/DISMISS, judge PASS but checker fail
    assert cells == ["sft", "5", "4 (80%)", "2 (40%)", "1 (20%)", "2", "1"]


def test_comparison_title_names_system_and_split():
    assert comparison_title("cmp_at_drop_heldout") == "AlignTune DPO, drop+warn pairs vs base, held-out"
    assert comparison_title("cmp_sft_test") == "SFT vs base, test"
    assert comparison_title("cmp_new_test") == "new vs base, test"


def test_holm_adjusts_and_keeps_order():
    from htd.report import holm

    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []
