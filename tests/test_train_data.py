import importlib.util
import json
from pathlib import Path

import pytest

from htd.train_data import (
    dpo_columns,
    filter_max_turn,
    filter_pair_types,
    load_jsonl,
    row_stats,
    sft_columns,
)


def row(i=1):
    return {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "chosen": [{"role": "assistant", "content": "good reply here"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "good reply here"}],
        "dialogue_id": f"d-{i}",
        "turn": i,
    }


def test_dpo_columns_keeps_only_trainer_fields():
    out = dpo_columns([row()])
    assert set(out[0]) == {"prompt", "chosen", "rejected"}
    assert out[0]["chosen"][0]["content"] == "good reply here"


def test_sft_columns_keeps_only_messages():
    out = sft_columns([row()])
    assert set(out[0]) == {"messages"}


def test_missing_fields_are_named():
    with pytest.raises(ValueError, match="rejected"):
        dpo_columns([{"prompt": [], "chosen": []}])
    with pytest.raises(ValueError, match="messages"):
        sft_columns([{"prompt": []}])


def test_load_jsonl_skips_blank_lines_and_rejects_empty(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(row(1)) + "\n\n" + json.dumps(row(2)) + "\n", encoding="utf-8")
    assert len(load_jsonl(p)) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_jsonl(empty)


def test_row_stats_counts_words_in_last_message():
    stats = row_stats([row(), row()], "chosen")
    assert stats["rows"] == 2 and stats["replies"] == 2
    assert stats["mean_words"] == 3.0 and stats["max_words"] == 3


def test_filter_max_turn_keeps_short_rows():
    rows = [{"turn": 1}, {"turn": 2}, {"turn": 3}]
    assert filter_max_turn(rows, 2) == [{"turn": 1}, {"turn": 2}]
    assert filter_max_turn(rows, None) == rows


def test_filter_pair_types_keeps_listed_types():
    rows = [{"pair_type": "drop"}, {"pair_type": "lecture"}, {"pair_type": "warn"}]
    assert filter_pair_types(rows, ["drop", "warn"]) == [{"pair_type": "drop"}, {"pair_type": "warn"}]
    assert filter_pair_types(rows, None) == rows


def test_filter_pair_types_rejects_empty_result():
    with pytest.raises(ValueError, match="no rows"):
        filter_pair_types([{"pair_type": "drop"}], [])


def test_filter_pair_types_rejects_unknown_type():
    with pytest.raises(ValueError, match="warnn"):
        filter_pair_types([{"pair_type": "drop"}, {"pair_type": "warn"}], ["drop", "warnn"])


def _train_aligntune():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_aligntune.py"
    spec = importlib.util.spec_from_file_location("train_aligntune", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_aligntune_writes_filtered_pairs(tmp_path, capsys):
    data = tmp_path / "dpo_train.jsonl"
    rows = [dict(row(i), pair_type=t) for i, t in enumerate(["drop", "lecture", "warn", "lecture"], 1)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "aligntune_dpo_drop"
    _train_aligntune().main(["--data", str(data), "--output", str(out), "--pair-types", "drop,warn", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    kept = load_jsonl(plan["data"])
    assert [r["pair_type"] for r in kept] == ["drop", "warn"]
    assert plan["pair_types"] == ["drop", "warn"] and plan["rows"] == 2
    assert kept[0] == rows[0] and len(load_jsonl(data)) == 4


def test_train_aligntune_strips_spaces_in_pair_types(tmp_path, capsys):
    data = tmp_path / "dpo_train.jsonl"
    rows = [dict(row(i), pair_type=t) for i, t in enumerate(["drop", "warn"], 1)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _train_aligntune().main(["--data", str(data), "--output", str(tmp_path / "o"), "--pair-types", "drop, warn", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["rows"] == 2


def test_train_aligntune_without_pair_types_uses_source_file(tmp_path, capsys):
    data = tmp_path / "dpo_train.jsonl"
    data.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    _train_aligntune().main(["--data", str(data), "--output", str(tmp_path / "o"), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["data"] == str(data) and "pair_types" not in plan
