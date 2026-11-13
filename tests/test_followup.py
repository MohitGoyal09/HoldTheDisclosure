import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_followup import holm  # noqa: E402


def test_holm_step_down_is_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == 0.04          # 4 * 0.01
    assert adj["c"] == 0.09          # 3 * 0.03
    assert adj["b"] == 0.09          # 2 * 0.04 = 0.08, raised to keep order
    assert adj["d"] == 0.5           # 1 * 0.5
    assert holm({"x": 0.9, "y": 0.8}) == {"y": 1.0, "x": 1.0}
