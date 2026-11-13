import importlib.util
from pathlib import Path


def _mod():
    spec = importlib.util.spec_from_file_location("make_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_wilson_interval_known_values():
    lo, hi = _mod().wilson(8, 60)
    assert abs(lo - 0.0691) < 1e-3 and abs(hi - 0.2417) < 1e-3
    assert _mod().wilson(0, 20)[0] == 0.0


def test_pooled_counts_dialogues_over_scripts():
    s = {"cells": [{"model": "m", "variant": "plain", "n": 20, "dr": [1, 1, 1, 1, 0.25]},
                   {"model": "m", "variant": "plain", "n": 20, "dr": [1, 1, 1, 1, 0.5]},
                   {"model": "m", "variant": "strict", "n": 20, "dr": [1, 1, 1, 1, 1.0]}]}
    assert _mod().pooled(s, "m") == (15, 40)
