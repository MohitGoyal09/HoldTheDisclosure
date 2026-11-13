import pytest

pytest.importorskip("auditkit")

from auditkit.model import Generated, Model, Result_  # noqa: E402

import htd.audit as audit  # noqa: E402
from htd.audit import auditkit_generate_batch, compare_systems, dialogue_flags  # noqa: E402
from htd.backends import FakeBackend  # noqa: E402
from htd.dialogues import build_dialogues  # noqa: E402
from htd.replay import Transcript, replay  # noqa: E402
from htd.seeds import load_seeds  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402

SHEET = load_sheet()
DS = build_dialogues(SHEET, load_seeds(), ["equity_sip", "branch_hours"], n_per_script=2)
STMT = " ".join(d.statement["en"] for d in SHEET.product("equity_sip").disclosures)


class RecordingModel(Model):
    name = "rec"

    def __init__(self):
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [Result_(completions=[Generated(text=f"reply {len(r.params['messages'])}<|END_RESPONSE|>junk")]) for r in requests]


def _patched(monkeypatch, model):
    from auditkit.model import AutoModel

    monkeypatch.setattr(AutoModel, "resolve", classmethod(lambda cls, spec, **kw: model))


def test_full_message_list_reaches_backend_via_replay(monkeypatch):
    model = RecordingModel()
    _patched(monkeypatch, model)
    gen = auditkit_generate_batch("fake:x")
    ts = replay(DS, SHEET, "plain", gen, "m")
    last = [r for r in model.requests if len(r.params["messages"]) == 10]
    assert len(last) == len(DS)
    roles = [m["role"] for m in last[0].params["messages"]]
    assert roles == ["system"] + ["user", "assistant"] * 4 + ["user"]
    assert last[0].params["messages"][2]["content"] == "reply 2"  # model's own reply kept
    assert last[0].request_type == "chat"
    assert ts[0].replies[0] == "reply 2"  # marker and junk trimmed


def test_hf_spec_passes_stop_sequences_and_others_do_not(monkeypatch):
    model = RecordingModel()
    _patched(monkeypatch, model)
    auditkit_generate_batch("hf:some/model")([[{"role": "user", "content": "hi"}]])
    assert model.requests[0].params["stop_sequences"] == ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]
    model.requests.clear()
    auditkit_generate_batch("openai:gpt")([[{"role": "user", "content": "hi"}]])
    assert "stop_sequences" not in model.requests[0].params


def _t(did, product, script, replies):
    return Transcript(did, product, script, "v", "m", ("q",) * 5, tuple(replies))


def _replies(k_ok):
    """5 replies; the one at turn 5 states the disclosure iff k_ok."""
    return ["x"] * 4 + [STMT if k_ok else "x"]


def test_dialogue_flags_skip_controls_and_score_turn_five():
    ts = [
        _t("a", "equity_sip", "en", _replies(True)),
        _t("b", "equity_sip", "en", _replies(False)),
        _t("c", "branch_hours", "en", _replies(False)),
        _t("d", "equity_sip", "en", [STMT] + ["x"] * 4),
    ]
    assert dialogue_flags(ts, SHEET) == {"a": 1, "b": 0, "d": 0}
    assert dialogue_flags(ts, SHEET, turn=1)["d"] == 1


def test_compare_systems_matches_hand_computed_diff():
    sc = lambda i: "en" if i < 10 else "hi"  # noqa: E731
    a = [_t(f"d{i}", "equity_sip", sc(i), _replies(i < 4)) for i in range(20)]  # 4/20
    b = [_t(f"d{i}", "equity_sip", sc(i), _replies(i < 16)) for i in reversed(range(20))]  # 16/20
    r = compare_systems(a, b, SHEET)
    assert r["n_paired"] == 20
    assert r["mean_a"] == pytest.approx(0.2) and r["mean_b"] == pytest.approx(0.8)
    assert r["diff"] == pytest.approx(0.6)
    sig = r["significance"]
    assert 0 <= sig["p_value"] <= 1  # AuditKIT's own value is reported as is
    assert sig["mcnemar_p"] == pytest.approx(2 / 2**12)  # 12 discordant pairs, all one way
    assert sig["mcnemar_significant"] is True
    assert sig["ci"][0] <= 0.6 <= sig["ci"][1]


def test_mcnemar_exact_p_hand_values():
    from htd.audit import mcnemar_exact_p

    assert mcnemar_exact_p([(0, 0), (1, 1)]) == 1.0
    assert mcnemar_exact_p([(0, 1)] * 3 + [(1, 0)] * 3) == 1.0
    assert mcnemar_exact_p([(0, 1)] * 4) == pytest.approx(0.125)


def test_per_script_values():
    a = [_t(f"d{i}", "equity_sip", "en" if i < 4 else "hi", _replies(i < 2)) for i in range(8)]
    b = [_t(f"d{i}", "equity_sip", "en" if i < 4 else "hi", _replies(i < 6)) for i in range(8)]
    r = compare_systems(a, b, SHEET)
    assert r["per_script"]["en"] == {"n": 4, "mean_a": 0.5, "mean_b": 1.0, "diff": 0.5}
    assert r["per_script"]["hi"] == {"n": 4, "mean_a": 0.0, "mean_b": 0.5, "diff": 0.5}


def test_mismatched_ids_raise():
    a = [_t("d0", "equity_sip", "en", _replies(True)), _t("d1", "equity_sip", "en", _replies(True))]
    b = [_t("d0", "equity_sip", "en", _replies(True)), _t("zzz", "equity_sip", "en", _replies(True))]
    with pytest.raises(ValueError, match="dialogue ids differ"):
        compare_systems(a, b, SHEET)


def test_compare_with_fake_backend_runs():
    ts = replay(DS, SHEET, "plain", FakeBackend().generate_batch, "fake")
    r = compare_systems(ts, ts, SHEET)
    assert r["diff"] == 0 and r["significance"]["significant"] is False


def test_dr_metric_registers_once_and_scores():
    from auditkit.sample import Sample

    cls = audit.register_dr_metric(SHEET)
    assert audit.register_dr_metric(SHEET) is cls
    m = cls()
    s = Sample(input="q", metadata={"product_id": "equity_sip"})
    assert m.applicable(s)
    assert m.score(s, STMT).value == 1.0
    assert m.score(s, "nothing").value == 0.0
