import pytest

from htd.helpful import build_prompt, parse_answer
from htd.sheet import load_sheet

SHEET = load_sheet()
ELSS = SHEET.product("elss")


def test_prompt_has_facts_customer_reply_and_ignores_warnings():
    p = build_prompt(ELSS, "Kaise invest karun?", "Open the app, Invest > Tax Saver.", (("hi", "hello"),))
    assert ELSS.facts_for_prompt in p
    assert "Kaise invest karun?" in p and "Open the app, Invest > Tax Saver." in p
    assert "Customer: hi" in p
    assert "regardless of any warnings" in p


@pytest.mark.parametrize("text,expected", [('{"helpful": "yes"}', "YES"), ('ok {"helpful": "No", "reason": "x"}', "NO")])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_parse_answer_rejects_other_values():
    with pytest.raises(ValueError):
        parse_answer('{"helpful": "maybe"}')


def test_helpful_rates_use_majority_and_skip_errors_and_ties():
    from htd.helpful import helpful_rates

    a = [{"dialogue_id": "x", "model": "sft", "helpful": "YES"}, {"dialogue_id": "y", "model": "sft", "helpful": "NO"}]
    b = [{"dialogue_id": "x", "model": "sft", "helpful": "YES"}, {"dialogue_id": "y", "model": "sft", "helpful": "ERROR"}]
    c = [{"dialogue_id": "x", "model": "sft", "helpful": "NO"}, {"dialogue_id": "y", "model": "sft", "helpful": "YES"}]
    assert helpful_rates([a, b, c]) == {"sft": {"n": 1, "yes": 1}}


def _row(system, did, ans):
    return {"dialogue_id": f"{system}|plain|{did}|t5", "model": system, "helpful": ans}


def test_helpful_votes_key_by_dialogue_without_system():
    from htd.helpful import helpful_votes

    panels = [[_row("base", "d1", "YES"), _row("base", "d2", "NO")]] * 3
    assert helpful_votes(panels) == {"base": {"plain|d1|t5": True, "plain|d2|t5": False}}


def test_helpful_vs_base_pairs_by_dialogue():
    from htd.helpful import helpful_vs_base

    base = {f"d{i}": True for i in range(8)}
    sft = {f"d{i}": False for i in range(8)} | {"extra": True}
    n, p = helpful_vs_base({"base": base, "sft": sft}, "sft", "base")
    assert n == 8 and abs(p - 2 / 2**8) < 1e-12
    assert helpful_vs_base({"base": base, "sft": base}, "sft", "base") == (8, 1.0)


def test_helpful_votes_keep_prompt_variants_apart():
    from htd.helpful import helpful_votes

    rows = [{"dialogue_id": "base|plain|d1|t5", "model": "base", "helpful": "YES"},
            {"dialogue_id": "base|strict|d1|t5", "model": "base", "helpful": "NO"}]
    assert helpful_votes([rows] * 3) == {"base": {"plain|d1|t5": True}, "base:strict": {"strict|d1|t5": False}}
