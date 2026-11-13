"""compare_systems must work when the optional auditkit package is absent."""

from htd.audit import compare_systems
from htd.replay import Transcript
from htd.sheet import load_sheet

SHEET = load_sheet()
PRODUCT = SHEET.product("equity_sip")
STATEMENT = " ".join(d.statement["en"] for d in PRODUCT.disclosures)


def tr(i, replies):
    return Transcript(f"d{i}", "equity_sip", "en", "plain", "m", tuple("u" * 5), tuple(replies))


def test_compare_systems_without_auditkit_still_reports_mcnemar():
    a = [tr(0, [STATEMENT] * 5), tr(1, [STATEMENT] * 5)]
    b = [tr(0, [STATEMENT] * 5), tr(1, [STATEMENT] + ["no disclosure"] * 4)]
    r = compare_systems(a, b, SHEET)
    assert r["n_paired"] == 2
    assert r["significance"]["mcnemar_p"] is not None
    assert r["significance"]["ci"] is not None
    assert "diff" in r
