import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path, split, replies):
    row = {"dialogue_id": f"{split}-000-en", "product_id": "fd", "script": "en", "variant": "plain",
           "model_name": "base", "user_turns": ["a", "b", "c", "d", "e"], "replies": replies}
    path.write_text(json.dumps(row) + "\n")


def test_pair_by_index_matches_fixed_and_shuffled_ids(tmp_path):
    _write(tmp_path / "a.jsonl", "test2", ["x"] * 5)
    _write(tmp_path / "b.jsonl", "test2_shuffled", ["x"] * 5)
    cmd = [sys.executable, str(ROOT / "scripts/compare.py"), str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    r = subprocess.run(cmd + ["--pair-by-index", "--out", str(tmp_path / "c.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "c.json").read_text())["per_script"]["en"]["n"] == 1
