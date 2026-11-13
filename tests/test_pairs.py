from collections import Counter

from htd.pairs import base_index, cut_pairs
from htd.prompts import system_prompt
from htd.rule_checker import check_reply
from htd.sheet import load_sheet

SHEET = load_sheet()
PRODUCT = SHEET.product("equity_sip")
CONTROL = SHEET.product("branch_hours")


def _record(dialogue_id, product_id="equity_sip", script="en", status="ok"):
    product = SHEET.product(product_id)
    disclosed = product.requires_disclosure
    statements = " ".join(d.statement[script] for d in product.disclosures)
    return {
        "dialogue_id": dialogue_id,
        "split": "train",
        "product_id": product_id,
        "script": script,
        "requires_disclosure": disclosed,
        "user_turns": [f"user {i}" for i in range(1, 6)],
        "chosen": [f"chosen {i} {statements}" for i in range(1, 6)],
        "drop": [f"drop {i}" for i in range(1, 6)] if disclosed else None,
        "lecture": [f"lecture {i} long long long" for i in range(1, 6)] if disclosed else None,
        "warn": None if disclosed else [f"warn {i} risk" for i in range(1, 6)],
        "gate_log": [],
        "generator_model": "fake",
        "status": status,
    }


def _many(n=20, product_id="equity_sip"):
    out = []
    for i in range(n):
        for script in ("en", "hi", "hinglish"):
            out.append(_record(f"train-{i:03d}-{script}", product_id, script))
    return out


def test_base_index_strips_script_suffix():
    assert base_index("train-003-hinglish") == "train-003"
    assert base_index("train-003-en") == "train-003"
    assert base_index("odd-id") == "odd-id"


def test_disclosure_product_gives_two_pairs_per_turn():
    out = cut_pairs([_record("train-000-en")], SHEET, val_frac=0.0, recovery_frac=0.0)
    assert len(out["dpo_train"]) == 10
    assert out["dpo_val"] == []
    types = Counter((r["turn"], r["pair_type"]) for r in out["dpo_train"])
    assert all(types[(t, p)] == 1 for t in range(1, 6) for p in ("drop", "lecture"))
    assert len(out["sft_train"]) == 5


def test_control_product_gives_one_pair_per_turn():
    out = cut_pairs([_record("train-000-en", "branch_hours")], SHEET, val_frac=0.0)
    assert len(out["dpo_train"]) == 5
    assert {r["pair_type"] for r in out["dpo_train"]} == {"warn"}
    assert all(r["recovery"] is False for r in out["dpo_train"])


def test_trl_conversational_format():
    out = cut_pairs([_record("train-000-en")], SHEET, val_frac=0.0, recovery_frac=0.0)
    row = next(r for r in out["dpo_train"] if r["turn"] == 3 and r["pair_type"] == "drop")
    prompt = row["prompt"]
    assert prompt[0] == {"role": "system", "content": system_prompt(PRODUCT, SHEET, "plain")}
    assert [m["role"] for m in prompt[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert prompt[-1]["content"] == "user 3"
    assert prompt[2]["content"].startswith("chosen 1")
    assert row["chosen"][0]["role"] == "assistant"
    assert row["chosen"][0]["content"].startswith("chosen 3")
    assert row["rejected"] == [{"role": "assistant", "content": "drop 3"}]
    sft = next(r for r in out["sft_train"] if r["turn"] == 3)
    assert sft["messages"] == prompt + row["chosen"]
    assert row["dialogue_id"] == "train-000-en" and row["script"] == "en"
    assert row["product_id"] == "equity_sip"


def test_recovery_rows_have_drop_in_history_and_disclosure_in_chosen():
    out = cut_pairs(_many(), SHEET, recovery_frac=0.5, val_frac=0.0)
    rec_rows = [r for r in out["dpo_train"] if r["recovery"]]
    assert rec_rows
    for r in rec_rows:
        assert r["turn"] >= 3
        assistant_msgs = [m["content"] for m in r["prompt"] if m["role"] == "assistant"]
        assert assistant_msgs[-1] == f"drop {r['turn'] - 1}"
        assert check_reply(r["chosen"][0]["content"], PRODUCT, SHEET).all_present
    ids = {r["dialogue_id"] for r in rec_rows}
    assert len(ids) == round(0.5 * 60)
    # one recovery turn per dialogue, and counts do not change
    for d in ids:
        assert len({r["turn"] for r in rec_rows if r["dialogue_id"] == d}) == 1
    assert len(out["dpo_train"]) == 60 * 10
    sft_rec = [r for r in out["sft_train"] if r["recovery"]]
    assert len(sft_rec) == len(ids)


def test_recovery_never_applies_to_control():
    out = cut_pairs(_many(10, "branch_hours"), SHEET, recovery_frac=1.0)
    assert not any(r["recovery"] for r in out["dpo_train"])


def test_no_base_index_in_both_train_and_val():
    out = cut_pairs(_many(), SHEET, val_frac=0.2)
    train = {base_index(r["dialogue_id"]) for r in out["dpo_train"]}
    val = {base_index(r["dialogue_id"]) for r in out["dpo_val"]}
    assert val and train and not (train & val)
    assert len(val) == 4  # 20 base indices * 0.2
    # all scripts of a base index stay together
    for b in val:
        assert {r["script"] for r in out["dpo_val"] if base_index(r["dialogue_id"]) == b} == {"en", "hi", "hinglish"}
    sft_train = {base_index(r["dialogue_id"]) for r in out["sft_train"]}
    sft_val = {base_index(r["dialogue_id"]) for r in out["sft_val"]}
    assert sft_train == train and sft_val == val


def test_deterministic_and_seed_sensitive():
    a = cut_pairs(_many(), SHEET, seed=3)
    b = cut_pairs(_many(), SHEET, seed=3)
    assert a == b
    c = cut_pairs(_many(), SHEET, seed=4)
    assert a != c


def test_input_order_does_not_change_the_split():
    recs = _many()
    a = cut_pairs(recs, SHEET)
    b = cut_pairs(list(reversed(recs)), SHEET)
    key = lambda rows: sorted((r["dialogue_id"], r["turn"], r["pair_type"], r["recovery"]) for r in rows)
    assert key(a["dpo_train"]) == key(b["dpo_train"])
    assert key(a["dpo_val"]) == key(b["dpo_val"])


def test_failed_records_are_skipped():
    recs = [_record("train-000-en"), _record("train-001-en", status="failed")]
    out = cut_pairs(recs, SHEET, val_frac=0.0)
    assert {r["dialogue_id"] for r in out["dpo_train"]} == {"train-000-en"}
    assert {r["dialogue_id"] for r in out["sft_train"]} == {"train-000-en"}


def test_empty_input():
    out = cut_pairs([], SHEET)
    assert out == {"dpo_train": [], "dpo_val": [], "sft_train": [], "sft_val": []}


def test_always_recover_turn_adds_one_recovery_pair_per_dialogue():
    out = cut_pairs(_many(10), SHEET, recovery_frac=0.0, val_frac=0.0, always_recover_turn=3)
    rec = [r for r in out["dpo_train"] if r["recovery"]]
    assert {r["dialogue_id"] for r in rec} == {f"train-{i:03d}-{s}" for i in range(10) for s in ("en", "hi", "hinglish")}
    assert {r["turn"] for r in rec} == {3}
    assert len(out["dpo_train"]) == 30 * 12
    control = cut_pairs(_many(2, "branch_hours"), SHEET, always_recover_turn=3)
    assert not any(r["recovery"] for r in control["dpo_train"])


def _pair(pair_type, chosen, rejected, product_id="equity_sip"):
    return {
        "pair_type": pair_type, "product_id": product_id, "script": "en",
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def test_pair_profile_counts_words_and_warning_markers():
    from htd.pairs import pair_profile

    sheet = load_sheet()
    good = "Returns depend on the market, are not guaranteed and can fall; 1% exit load applies within 1 year."
    rows = [_pair("drop", good, "Sure, tap Invest and confirm."), _pair("drop", good, "Done.")]
    prof = pair_profile(rows, sheet)
    assert prof["drop"]["n"] == 2
    assert prof["drop"]["checker_pass_chosen"] == 2 and prof["drop"]["checker_pass_rejected"] == 0
    assert prof["drop"]["words_rejected"] == 3.0
    assert prof["drop"]["markers_chosen"] > prof["drop"]["markers_rejected"] == 0
