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


def test_rule_sheet_reads_three_answers_and_fisher_is_symmetric():
    p = Path(__file__).resolve().parents[1] / "scripts" / "rule_sheet.py"
    spec = importlib.util.spec_from_file_location("rule_sheet", p)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    items = [dict(ITEMS[0], required=["Returns are not guaranteed."]), dict(ITEMS[1], required=["Lock-in of 3 years."])]
    md = rs.to_md(items)
    assert "Required disclosures for this product" in md and rs.from_md(md)[0] == {}
    filled = md.replace("Q1: \n", "Q1: yes\n", 1).replace("Q2: \n", "Q2: no\n", 1).replace("Q3: \n", "Q3: Y\n", 1)
    labels, problems = rs.from_md(filled)
    assert labels == {"aa11": {"discloses": "yes", "answers": "no", "accept": "yes"}} and len(problems) == 3
    assert rs.fisher_two_sided(10, 5, 4, 11) == rs.fisher_two_sided(4, 11, 10, 5)
    assert rs.fisher_two_sided(5, 5, 5, 5) == 1.0
