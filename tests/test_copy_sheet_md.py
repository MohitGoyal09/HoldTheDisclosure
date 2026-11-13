import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "copy_sheet.py"
_spec = importlib.util.spec_from_file_location("copy_sheet", _p)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

ITEMS = [{"item": "aa11", "script": "en", "product": "FD", "history": [["q1", "r1\nQ1: yes"]] * 4,
          "customer": "c5", "reply": "Q2: no\nline two"},
         {"item": "bb22", "script": "hi", "product": "ELSS", "history": [["q", "r"]] * 4, "customer": "c", "reply": "r"}]


def test_md_roundtrip_reads_only_the_answer_lines():
    md = cs.to_md(ITEMS)
    assert "System names are hidden" in md and "## 1 / 2 · item `aa11`" in md
    labels, problems = cs.from_md(md)
    assert labels == {} and len(problems) == 4  # reply text quoted with "> " is never read as an answer
    filled = md.replace("Q1: \n", "Q1: Yes\n", 1).replace("Q2: \n", "Q2: n\n", 1).replace("Note: \n", "Note: unsure\n", 1)
    labels, problems = cs.from_md(filled)
    assert labels == {"aa11": {"answers": "yes", "accept": "no", "note": "unsure"}}
    assert problems == ["bb22 Q1: missing", "bb22 Q2: missing"]
