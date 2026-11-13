from htd.labels import stratified_sample


def rows():
    out = []
    for script in ("en", "hi", "hinglish"):
        for label in ("PASS", "DROP"):
            for i in range(3):
                out.append({"row_id": f"{script}-{label}-{i}", "script": script, "judge_label": label})
    return out


def test_covers_every_stratum_before_repeating():
    sample = stratified_sample(rows(), 6)
    keys = {(r["script"], r["judge_label"]) for r in sample}
    assert len(keys) == 6
    assert len(sample) == 6


def test_round_robin_balances_strata():
    sample = stratified_sample(rows(), 12)
    counts = {}
    for r in sample:
        counts[(r["script"], r["judge_label"])] = counts.get((r["script"], r["judge_label"]), 0) + 1
    assert set(counts.values()) == {2}


def test_is_deterministic_and_seed_changes_order():
    a = stratified_sample(rows(), 9, seed=0)
    b = stratified_sample(rows(), 9, seed=0)
    c = stratified_sample(rows(), 9, seed=1)
    assert [r["row_id"] for r in a] == [r["row_id"] for r in b]
    assert [r["row_id"] for r in a] != [r["row_id"] for r in c]


def test_handles_n_larger_than_population_and_zero():
    assert len(stratified_sample(rows(), 1000)) == len(rows())
    assert stratified_sample(rows(), 0) == []
    assert stratified_sample([], 10) == []


def test_item_id_is_stable_and_hides_the_system():
    from htd.labels import item_id

    a = item_id("sft|plain|test-000-en|t5")
    assert a == item_id("sft|plain|test-000-en|t5")
    assert a != item_id("base|plain|test-000-en|t5")
    assert "sft" not in a and len(a) == 10


def test_pair_labels_joins_by_item_and_skips_blank():
    from htd.labels import item_id, pair_labels

    judged = [{"dialogue_id": "sft|plain|d1|t5", "judge_label": "pass"}, {"dialogue_id": "base|plain|d1|t5", "judge_label": "DROP"}]
    sheet = [
        {"item": item_id("sft|plain|d1|t5"), "human_label": "over_correct"},
        {"item": item_id("base|plain|d1|t5"), "human_label": " "},
        {"item": "unknown", "human_label": "PASS"},
    ]
    assert pair_labels(sheet, judged) == [("OVER-CORRECT", "PASS")]


def test_pair_labels_drops_judge_errors():
    from htd.labels import item_id, pair_labels

    judged = [{"dialogue_id": "a", "judge_label": "ERROR"}]
    assert pair_labels([{"item": item_id("a"), "human_label": "PASS"}], judged) == []


def test_yes_no_normalises_and_rejects_other_values():
    import pytest

    from htd.labels import yes_no

    assert [yes_no(v) for v in ("Yes ", "y", "1", "NO", "n", "0", "")] == ["YES", "YES", "YES", "NO", "NO", "NO", ""]
    with pytest.raises(ValueError):
        yes_no("maybe")


def test_human_column_and_consensus():
    from htd.labels import consensus, human_column

    a = [{"item": "i1", "human_label": "pass", "states_all": "yes"},
         {"item": "i2", "human_label": "DROP", "states_all": "no"},
         {"item": "i3", "human_label": "", "states_all": ""}]
    b = [{"item": "i1", "human_label": "PASS", "states_all": "no"},
         {"item": "i2", "human_label": "drop", "states_all": "n"},
         {"item": "i3", "human_label": "DISMISS", "states_all": "yes"}]
    assert human_column(a, "states_all") == {"i1": "YES", "i2": "NO"}
    assert human_column(b, "human_label") == {"i1": "PASS", "i2": "DROP", "i3": "DISMISS"}
    assert consensus(human_column(a, "states_all"), human_column(b, "states_all")) == {"i2": "NO"}
    assert consensus(human_column(a, "human_label"), human_column(b, "human_label")) == {"i1": "PASS", "i2": "DROP"}
