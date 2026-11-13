from htd.audit import wilson


def test_wilson_interval_known_values():
    lo, hi = wilson(8, 60)
    assert abs(lo - 0.0691) < 1e-3 and abs(hi - 0.2417) < 1e-3
    assert wilson(0, 20)[0] == 0.0 and abs(wilson(0, 20)[1] - 0.1611) < 1e-3
