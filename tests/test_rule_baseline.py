import importlib.util
from pathlib import Path

from htd.replay import Transcript
from htd.rule_checker import check_reply
from htd.sheet import load_sheet

_p = Path(__file__).resolve().parents[1] / "scripts" / "rule_baseline.py"
_spec = importlib.util.spec_from_file_location("rule_baseline", _p)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def test_every_statement_passes_its_own_checker():
    sheet = load_sheet()
    for p in sheet.products:
        if not p.requires_disclosure:
            continue
        for script in ("en", "hi", "hinglish"):
            text = rb.append_missing("Sure.", p, sheet, script)
            assert check_reply(text, p, sheet, script).all_present, (p.id, script)


def test_reply_that_already_discloses_is_unchanged_and_control_is_untouched():
    sheet = load_sheet()
    p = next(x for x in sheet.products if x.requires_disclosure)
    full = rb.append_missing("Sure.", p, sheet, "en")
    assert rb.append_missing(full, p, sheet, "en") == full
    c = next(x for x in sheet.products if not x.requires_disclosure)
    t = Transcript("d", c.id, "en", "plain", "base", ("q",), ("a",))
    assert rb.rule_transcript(t, sheet).replies == ("a",)
