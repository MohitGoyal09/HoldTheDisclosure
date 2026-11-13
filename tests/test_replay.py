from htd.backends import FakeBackend
from htd.dialogues import build_dialogues
from htd.replay import load_transcripts, replay, save_transcripts
from htd.seeds import load_seeds
from htd.sheet import load_sheet

SHEET = load_sheet()
DS = build_dialogues(SHEET, load_seeds(), ["equity_sip", "branch_hours"], n_per_script=2)


def test_one_batch_call_per_turn_and_history_grows():
    calls = []

    def gen(batch):
        calls.append(batch)
        return [f"r{len(m)}" for m in batch]

    ts = replay(DS, SHEET, "plain", gen, "m")
    assert len(calls) == 5
    assert all(len(b) == len(DS) for b in calls)
    # turn t has system + (2t-1) messages
    assert [len(c[0]) for c in calls] == [2, 4, 6, 8, 10]
    t = ts[0]
    assert len(t.replies) == 5 and t.model_name == "m" and t.variant == "plain"
    roles = [m["role"] for m in calls[4][0]]
    assert roles == ["system"] + ["user", "assistant"] * 4 + ["user"]
    assert calls[4][0][2]["content"] == "r2"  # own reply kept in history
    assert calls[4][0][-1]["content"] == DS[0].user_turns[4]


def test_system_prompt_is_per_product():
    seen = []
    replay(DS, SHEET, "plain", lambda b: (seen.append(b[0][0]["content"]), seen.append(b[1][0]["content"]), [""] * len(b))[2], "m", max_turns=1)
    assert seen[0] != seen[1]


def test_fake_backend_and_roundtrip(tmp_path):
    ts = replay(DS, SHEET, "plain", FakeBackend().generate_batch, "fake")
    stmt = SHEET.product("equity_sip").disclosures[0].statement
    t = next(t for t in ts if t.product_id == "equity_sip")
    assert any(stmt[t.script] in r or stmt["en"] in r for r in t.replies[:2])
    assert all(stmt["en"] not in r and stmt[t.script] not in r for r in t.replies[2:])
    p = tmp_path / "t.jsonl"
    save_transcripts(ts, p)
    assert load_transcripts(p) == ts


def test_strict_prompt_uses_the_dialogue_script():
    ds = [d for d in DS if d.script == "hi" and d.product_id == "equity_sip"]
    assert ds
    seen = []

    def gen(batch):
        seen.append(batch[0][0]["content"])
        return [""] * len(batch)

    replay(ds, SHEET, "strict", gen, "m", max_turns=1)
    statement = SHEET.product("equity_sip").disclosures[0].statement
    assert statement["hi"] in seen[0]
    assert statement["en"] not in seen[0]


def test_edit_is_applied_before_the_reply_enters_the_history():
    calls = []

    def gen(batch):
        calls.append(batch)
        return ["r"] * len(batch)

    ts = replay(DS, SHEET, "plain", gen, "m", edit=lambda d, r: f"{r}+{d.product_id}")
    assert ts[0].replies[0] == f"r+{DS[0].product_id}"
    assert calls[1][0][2]["content"] == f"r+{DS[0].product_id}"
