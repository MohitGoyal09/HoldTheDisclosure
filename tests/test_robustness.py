import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_robustness import cluster_test  # noqa: E402


def test_cluster_test_counts_scenarios_not_dialogues():
    a = {f"t-{i:03d}-{s}": 0 for i in range(10) for s in ("en", "hi", "hinglish")}
    n, p, lo, hi = cluster_test(a, dict(a))
    assert n == 10 and p == 1.0 and lo == hi == 0.0
    b = {k: 1 for k in a}
    n, p, lo, hi = cluster_test(a, b)
    # 10 scenarios all improve: the exact sign-flip p is 2 / 2**10, well below 0.01
    assert n == 10 and p < 0.01 and lo == hi == 1.0


def test_fresh_flags_drops_a_pass_copied_from_an_earlier_turn():
    from types import SimpleNamespace

    from htd.sheet import load_sheet
    from render_robustness import fresh_flags

    sheet = load_sheet()
    ok = "If you do not pay the full bill, interest of 3.5% per month applies on the unpaid amount."
    mk = lambda i, replies: SimpleNamespace(dialogue_id=f"t-{i:03d}-en", product_id="credit_card", script="en",  # noqa: E731
                                            replies=replies)
    copied = mk(0, [ok, "a", "b", "c", ok])
    fresh = mk(1, ["Sure, here is how to apply.", "a", "b", "c", ok])
    assert fresh_flags(sheet, [copied, fresh], 0.9) == {"t-000-en": 0, "t-001-en": 1}


def test_strat_perm_detects_copied_passes_that_are_less_helpful():
    from render_robustness import strat_perm

    units = [(f"d{i}", cp, not cp) for i in range(12) for cp in (True, False)]
    d, p, hc, nc, hf, nf = strat_perm(units, n=2000)
    assert (d, hc, nc, hf, nf) == (1.0, 0, 12, 12, 12)
    assert p < 0.01
    flat = [(f"d{i}", cp, True) for i in range(12) for cp in (True, False)]
    assert strat_perm(flat, n=200)[:2] == (0.0, 1.0)
