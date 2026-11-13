import json

import pytest
from curatorkit.llm.base import BaseLLM, LLMResponse

from htd.judge import JudgeError, build_prompt, label_reply, parse_label
from htd.sheet import load_sheet

SHEET = load_sheet()
ELSS = SHEET.product("elss")


class ScriptedLLM(BaseLLM):
    def __init__(self, texts):
        super().__init__(model="scripted", max_retries=0)
        self.texts = list(texts)
        self.calls = []

    def _call(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMResponse(text=self.texts.pop(0) if len(self.texts) > 1 else self.texts[0], model=self.model)


def test_prompt_has_facts_disclosures_customer_and_reply():
    p = build_prompt(ELSS, SHEET, "Short mein batao", "Sure, start in the app.")
    assert ELSS.facts_for_prompt in p
    for d in ELSS.disclosures:
        assert d.statement["en"] in p
    assert "Short mein batao" in p and "Sure, start in the app." in p


@pytest.mark.parametrize("label", ["PASS", "DROP", "DISMISS", "OVER-CORRECT"])
def test_parse_all_labels(label):
    assert parse_label(json.dumps({"label": label, "reason": "because"})).label == label


def test_parse_is_tolerant_of_case_underscores_and_wrapping_text():
    assert parse_label('Here: {"label": "over_correct", "reason": "x"} done').label == "OVER-CORRECT"
    assert parse_label('{"label": " drop "}').label == "DROP"


@pytest.mark.parametrize("raw", ["no json", "{bad", '["PASS"]', '{"label": "MAYBE"}', '{"reason": "x"}'])
def test_parse_rejects_unusable_labels(raw):
    with pytest.raises(JudgeError):
        parse_label(raw)


def test_label_reply_retries_then_succeeds():
    llm = ScriptedLLM(["junk", '{"label": "PASS", "reason": "fine"}'])
    assert label_reply(llm, ELSS, SHEET, "q", "a").label == "PASS"
    assert len(llm.calls) == 2


def test_label_reply_gives_up_after_two_retries():
    llm = ScriptedLLM(["junk"])
    with pytest.raises(JudgeError, match="after 2 retries"):
        label_reply(llm, ELSS, SHEET, "q", "a")
    assert len(llm.calls) == 3


def test_prompt_uses_the_reply_script():
    p = build_prompt(ELSS, SHEET, "q", "a", "hi")
    for d in ELSS.disclosures:
        assert d.statement["hi"] in p
        assert d.statement["en"] not in p


def test_prompt_includes_earlier_turns():
    p = build_prompt(ELSS, SHEET, "q", "a", "en", [("earlier question", "earlier answer")])
    assert "Customer: earlier question" in p and "Assistant: earlier answer" in p


def test_label_reply_forwards_history():
    llm = ScriptedLLM(['{"label": "OVER-CORRECT", "reason": "repeats"}'])
    label_reply(llm, ELSS, SHEET, "q", "a", "hi", [("p", "r")])
    content = llm.calls[0][-1]["content"]
    assert "Customer: p" in content and "Assistant: r" in content


def test_v1_rubric_is_unchanged_by_default():
    p = build_prompt(ELSS, SHEET, "c", "r")
    assert "counts as lecturing" in p and p == build_prompt(ELSS, SHEET, "c", "r", rubric="v1")


def test_v2_rubric_requires_disclosure_every_turn_and_lists_facts():
    p = build_prompt(ELSS, SHEET, "Bas haan bolo", "Sure.", rubric="v2")
    assert ELSS.facts_for_prompt in p and "Bas haan bolo" in p
    for d in ELSS.disclosures:
        assert d.statement["en"] in p
    assert "counts as lecturing" not in p
    assert "every reply" in p and "not lecturing" in p
    assert "DISMISS" in p and "OVER-CORRECT" in p


def test_unknown_rubric_raises():
    with pytest.raises(ValueError, match="rubric"):
        build_prompt(ELSS, SHEET, "c", "r", rubric="v9")


def test_panel_rubric_uses_preregistered_definitions_only():
    p = build_prompt(ELSS, SHEET, "c", "r", rubric="panel")
    assert "refuses, stalls or lectures instead of answering" in p
    assert "counts as lecturing" not in p and "not lecturing" not in p
