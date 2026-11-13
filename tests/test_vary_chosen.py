import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from htd.sheet import load_sheet  # noqa: E402
from vary_chosen import vary_record  # noqa: E402


class Fixed:
    """Returns one fixed chosen reply for every call."""

    def __init__(self, reply):
        self.reply = reply

    def generate(self, messages):
        return type("R", (), {"text": json.dumps({"chosen": self.reply})})()


def _record(sheet):
    product = next(p for p in sheet.products if not p.requires_disclosure)
    a = "You can check it in the Sahyog app on the home screen."
    return {"status": "ok", "product_id": product.id, "script": "en", "user_turns": ["q1", "q2", "q3", "q4", "q5"],
            "chosen": [a, a, "Something quite different about branch timings and documents.", a, a]}


def test_vary_record_rewrites_only_repeated_turns():
    sheet = load_sheet()
    rec = _record(sheet)
    fresh = "Sure, send BAL by SMS from your registered number and the balance arrives in a minute."
    out = vary_record(rec, sheet, Fixed(fresh), cut=0.7, keep=0.8)
    assert out["chosen"][0] == rec["chosen"][0]
    assert out["chosen"][2] == rec["chosen"][2]
    assert out["chosen"][1] == fresh
    assert out["chosen_original"] == rec["chosen"]
    assert [x["turn"] for x in out["vary_log"]] == [2, 4, 5]


def test_vary_record_keeps_original_when_no_rewrite_is_different_enough():
    sheet = load_sheet()
    rec = _record(sheet)
    out = vary_record(rec, sheet, Fixed(rec["chosen"][0]), cut=0.7, keep=0.8)
    assert out["chosen"] == rec["chosen"]
    assert all(x["after"] is None for x in out["vary_log"])
