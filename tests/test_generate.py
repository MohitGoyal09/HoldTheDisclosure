import json

import pytest
from curatorkit.llm.base import BaseLLM, LLMResponse

from htd.dialogues import Dialogue
from htd.generate import (
    CandidateError,
    FakeGeneratorLLM,
    append_record,
    build_prompt,
    completed_ids,
    dedupe_records,
    gate_report,
    generate_dialogue,
    parse_candidates,
    request_candidates,
    reward_judge,
    run_gates,
)
from htd.sheet import SCRIPTS, load_sheet

SHEET = load_sheet()
EQUITY = SHEET.product("equity_sip")
HOURS = SHEET.product("branch_hours")
KINDS = ("chosen", "drop", "lecture")


class ScriptedLLM(BaseLLM):
    """Returns queued texts in order; repeats the last one when the queue runs out."""

    def __init__(self, texts):
        super().__init__(model="scripted", max_retries=0)
        self.texts = list(texts)
        self.calls = []

    def _call(self, messages, **kwargs):
        self.calls.append(messages)
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return LLMResponse(text=text, model=self.model)


def _dialogue(product_id="equity_sip", script="en", dialogue_id="t-000-en"):
    return Dialogue(
        dialogue_id=dialogue_id,
        split="train",
        product_id=product_id,
        script=script,
        user_turns=tuple(f"customer message {i}" for i in range(1, 6)),
        tactic_ids=("a", "b", "c", "d"),
    )


def _good(script="en", product=EQUITY):
    """A candidate set that passes every gate, taken from the fake generator."""
    marks = f"PRODUCT_ID: {product.id}\nSCRIPT: {script}\nTURN: 1\n"
    out = FakeGeneratorLLM(SHEET).generate([{"role": "user", "content": marks}]).text
    return json.loads(out)


# ------------------------------------------------------------ parsing

def test_parse_plain_fenced_and_wrapped_json():
    raw = '{"chosen": "a", "warn": "b"}'
    expected = {"chosen": "a", "warn": "b"}
    assert parse_candidates(raw, ("chosen", "warn")) == expected
    assert parse_candidates(f"```json\n{raw}\n```", ("chosen", "warn")) == expected
    assert parse_candidates(f"Here you go: {raw} Hope it helps", ("chosen", "warn")) == expected


@pytest.mark.parametrize("raw", ["no json", "{bad", '["x"]', '{"chosen": "a"}', '{"chosen": " ", "warn": "b"}'])
def test_parse_rejects_unusable_replies(raw):
    with pytest.raises(CandidateError):
        parse_candidates(raw, ("chosen", "warn"))


def test_request_retries_twice_on_bad_json_then_succeeds():
    ok = json.dumps({"chosen": "a", "warn": "b"})
    llm = ScriptedLLM(["junk", "{oops", ok])
    assert request_candidates(llm, "p", ("chosen", "warn"))["chosen"] == "a"
    assert len(llm.calls) == 3


def test_request_gives_up_after_two_retries():
    llm = ScriptedLLM(["junk"])
    with pytest.raises(CandidateError):
        request_candidates(llm, "p", ("chosen", "warn"))
    assert len(llm.calls) == 3


# ------------------------------------------------------------ prompt

def test_prompt_has_facts_statements_history_and_language_rule():
    d = _dialogue(script="hinglish")
    prompt = build_prompt(EQUITY, SHEET, "hinglish", d.user_turns, ["reply one", "reply two"])
    assert EQUITY.facts_for_prompt in prompt
    for disclosure in EQUITY.disclosures:
        assert disclosure.statement["hinglish"] in prompt
    assert "Customer: customer message 2\nAssistant: reply two" in prompt
    assert "customer message 3" in prompt and "customer message 4" not in prompt
    assert "code-mixed Hinglish" in prompt
    assert "TURN: 3" in prompt and "PRODUCT_ID: equity_sip" in prompt


def test_control_prompt_has_no_disclosure_block():
    prompt = build_prompt(HOURS, SHEET, "en", _dialogue("branch_hours").user_turns, [])
    assert '"warn"' in prompt and '"lecture"' not in prompt
    assert "(none" in prompt


# ------------------------------------------------------------ gates

def test_good_candidates_pass_all_gates():
    assert run_gates(_good(), EQUITY, SHEET, "msg") == {}


def test_chosen_without_disclosure_fails_g1():
    c = {**_good(), "chosen": "Sure, open the app and start."}
    assert "g1_rule" in run_gates(c, EQUITY, SHEET, "m")["chosen"]


def test_chosen_with_dismissal_fails_g1():
    c = _good()
    c["chosen"] += " Honestly there is no risk, don't worry about it."
    assert "g1_rule" in run_gates(c, EQUITY, SHEET, "m")["chosen"]


def test_overlong_chosen_fails_g2():
    c = _good()
    c["chosen"] += " filler" * 80
    c["lecture"] += " filler" * 200
    f = run_gates(c, EQUITY, SHEET, "m")
    assert "g2_length" in f["chosen"] and "lecture" not in f


def test_drop_that_leaks_a_key_fact_fails_g1():
    c = {**_good(), "drop": "Sure, start now. The exit load is small."}
    assert "g1_rule" in run_gates(c, EQUITY, SHEET, "m")["drop"]


def test_drop_with_dismissal_fails_g1():
    c = {**_good(), "drop": "Sure, start now. It is completely safe."}
    assert "g1_rule" in run_gates(c, EQUITY, SHEET, "m")["drop"]


def test_short_lecture_fails_g2():
    c = {**_good(), "lecture": "I cannot help."}
    assert "g2_length" in run_gates(c, EQUITY, SHEET, "m")["lecture"]


def test_control_gates():
    good = _good(product=HOURS)
    assert run_gates(good, HOURS, SHEET, "m") == {}
    bad_chosen = {**good, "chosen": "We open at nine. There is a risk of queues."}
    assert "g1_rule" in run_gates(bad_chosen, HOURS, SHEET, "m")["chosen"]
    bad_warn = {**good, "warn": "We open at nine."}
    assert "g1_rule" in run_gates(bad_warn, HOURS, SHEET, "m")["warn"]


def test_judge_hook_is_off_by_default_and_runs_when_given():
    c = _good()
    calls = []

    def judge(kind, reply, msg):
        calls.append(kind)
        return (kind != "chosen", "too bland")

    assert run_gates(c, EQUITY, SHEET, "m") == {}
    f = run_gates(c, EQUITY, SHEET, "m", judge)
    assert f["chosen"]["g3_judge"] == ["too bland"] and set(calls) == set(KINDS)


def test_reward_judge_uses_curatorkit_reward_gate():
    good = reward_judge(ScriptedLLM(['{"overall_score": 0.95}']))
    bad = reward_judge(ScriptedLLM(['{"overall_score": 0.1}']))
    assert good("chosen", "reply", "msg") == (True, "")
    ok, reason = bad("chosen", "reply", "msg")
    assert not ok and "below_reward_threshold" in reason
    assert bad("drop", "reply", "msg") == (True, "")


# ------------------------------------------------------------ generate_dialogue

@pytest.mark.parametrize("product", SHEET.products, ids=lambda p: p.id)
@pytest.mark.parametrize("script", SCRIPTS)
def test_fake_generator_passes_every_gate_for_every_product_and_script(product, script):
    rec = generate_dialogue(_dialogue(product.id, script), SHEET, FakeGeneratorLLM(SHEET))
    assert rec["status"] == "ok", rec.get("failure_reason")
    assert len(rec["chosen"]) == 5
    assert (rec["drop"] is None) == (not product.requires_disclosure)
    assert (rec["warn"] is None) == product.requires_disclosure


def test_record_fields():
    rec = generate_dialogue(_dialogue(), SHEET, FakeGeneratorLLM(SHEET))
    assert set(rec) == {
        "dialogue_id", "split", "product_id", "script", "requires_disclosure", "user_turns",
        "chosen", "drop", "lecture", "warn", "gate_log", "generator_model", "status",
    }
    assert rec["warn"] is None and len(rec["drop"]) == len(rec["lecture"]) == 5
    assert rec["generator_model"] == "fake" and rec["user_turns"][0] == "customer message 1"
    assert len(rec["gate_log"]) == 5 and all(e["passed"] for e in rec["gate_log"])


def test_bad_turn_is_regenerated_then_accepted():
    good = json.dumps(_good())
    bad = json.dumps({**_good(), "drop": "The exit load is 1%."})
    llm = ScriptedLLM([bad, good])
    rec = generate_dialogue(_dialogue(), SHEET, llm)
    assert rec["status"] == "ok"
    first = rec["gate_log"][0]
    assert first["turn"] == 1 and not first["passed"] and "g1_rule" in first["failures"]["drop"]
    assert rec["gate_log"][1]["attempt"] == 2 and rec["gate_log"][1]["passed"]


def test_persistent_failure_marks_dialogue_failed_with_reason():
    bad = json.dumps({**_good(), "drop": "The exit load is 1%."})
    llm = ScriptedLLM([bad])
    rec = generate_dialogue(_dialogue(), SHEET, llm)
    assert rec["status"] == "failed"
    assert len(llm.calls) == 3  # first try plus 2 regenerations
    assert "turn 1" in rec["failure_reason"] and "drop" in rec["failure_reason"]
    assert rec["chosen"] == []


def test_unparseable_output_fails_after_retries():
    llm = ScriptedLLM(["junk"])
    rec = generate_dialogue(_dialogue(), SHEET, llm)
    assert rec["status"] == "failed" and "JSON" in rec["failure_reason"]
    assert len(llm.calls) == 9  # 3 attempts x (1 call + 2 JSON retries)


def test_history_uses_chosen_replies_only():
    llm = ScriptedLLM([json.dumps(_good())])
    generate_dialogue(_dialogue(), SHEET, llm)
    second_prompt = llm.calls[1][-1]["content"]
    good = _good()
    assert f"Assistant: {good['chosen']}" in second_prompt
    assert good["drop"] not in second_prompt.replace(good["chosen"], "")


def test_judge_failure_blocks_dialogue():
    rec = generate_dialogue(_dialogue(), SHEET, FakeGeneratorLLM(SHEET), judge=lambda k, r, m: (False, "no"))
    assert rec["status"] == "failed" and "g3_judge" in rec["failure_reason"]


def test_gate_report_counts():
    bad = json.dumps({**_good(), "drop": "The exit load is 1%."})
    failed = generate_dialogue(_dialogue(dialogue_id="a-000-en"), SHEET, ScriptedLLM([bad]))
    ok = generate_dialogue(_dialogue(dialogue_id="a-001-en"), SHEET, FakeGeneratorLLM(SHEET))
    rep = gate_report([failed, ok])
    assert rep["ok"] == 1 and rep["failed"] == 1
    assert rep["gate_failures"] == {"drop:g1_rule": 3}
    assert "a-000-en" in rep["failed_reasons"]


# ------------------------------------------------------------ resume and crash safety

class BoomLLM(BaseLLM):
    """Every call fails like a provider error."""

    def __init__(self):
        super().__init__(model="boom", max_retries=0)
        self.calls = 0

    def _call(self, messages, **kwargs):
        self.calls += 1
        raise RuntimeError("429 rate limited")


def test_api_error_is_retried_then_recorded_as_failed():
    llm = BoomLLM()
    rec = generate_dialogue(_dialogue(), SHEET, llm)
    assert rec["status"] == "failed"
    assert "RuntimeError" in rec["failure_reason"] and "429" in rec["failure_reason"]
    assert llm.calls == 3


def test_completed_ids_ignores_missing_bad_and_blank_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        '{"dialogue_id": "a", "status": "ok"}\n\nnot json\n{"no_id": 1}\n'
        '{"dialogue_id": "b", "status": "ok"}\n{"dialogue_id": "c", "status": "failed"}\n',
        encoding="utf-8",
    )
    assert completed_ids(p) == {"a", "b"}
    assert completed_ids(tmp_path / "missing.jsonl") == set()


def test_append_record_keeps_earlier_records(tmp_path):
    p = tmp_path / "nested" / "out.jsonl"
    append_record(p, {"dialogue_id": "a"})
    append_record(p, {"dialogue_id": "b"})
    assert [json.loads(x)["dialogue_id"] for x in p.read_text(encoding="utf-8").splitlines()] == ["a", "b"]



def test_gate_rejects_a_reply_in_the_wrong_script():
    c = _good(script="hi")
    c = {k: " ".join(_good(script="en").values()) for k in c}  # English text for a Hindi turn
    c = {"chosen": _good(script="en")["chosen"], "drop": "Sure, open the app.", "lecture": "I cannot help. " * 20}
    f = run_gates(c, EQUITY, SHEET, "m", script="hi")
    assert "g1_rule" in f["chosen"] and any("script" in p for p in f["chosen"]["g1_rule"])


def test_dedupe_keeps_one_record_per_dialogue_last_wins():
    rows = [{"dialogue_id": "b", "v": 1}, {"dialogue_id": "a", "v": 1}, {"dialogue_id": "b", "v": 2}]
    assert dedupe_records(rows) == [{"dialogue_id": "a", "v": 1}, {"dialogue_id": "b", "v": 2}]


def test_completed_ids_accepts_a_custom_done_test(tmp_path):
    p = tmp_path / "judged.jsonl"
    p.write_text(
        '{"dialogue_id": "a", "judge_label": "PASS"}\n{"dialogue_id": "b", "judge_label": "ERROR"}\n',
        encoding="utf-8",
    )
    assert completed_ids(p) == set()
    assert completed_ids(p, done=lambda r: r.get("judge_label") != "ERROR") == {"a"}
