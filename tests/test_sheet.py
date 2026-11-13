"""Tests for sheet loading and validation."""

import copy
import json

import pytest

from htd.sheet import SheetError, load_sheet


def test_load_default_sheet_counts():
    sheet = load_sheet()
    assert len(sheet.products) == 10
    assert len(sheet.by_split("train")) == 4
    assert len(sheet.by_split("heldout")) == 2
    assert len(sheet.by_split("control")) == 4


def _base_raw():
    from htd.sheet import DEFAULT_SHEET_PATH

    return json.loads(DEFAULT_SHEET_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, raw):
    path = tmp_path / "sheet.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def test_bad_script_key_raises(tmp_path):
    raw = copy.deepcopy(_base_raw())
    raw["products"][0]["name"]["fr"] = raw["products"][0]["name"].pop("hinglish")
    with pytest.raises(SheetError):
        load_sheet(_write(tmp_path, raw))


def test_bad_warning_marker_script_key_raises(tmp_path):
    raw = copy.deepcopy(_base_raw())
    raw["warning_markers"]["xx"] = raw["warning_markers"].pop("hinglish")
    with pytest.raises(SheetError):
        load_sheet(_write(tmp_path, raw))


def test_duplicate_product_id_raises(tmp_path):
    raw = copy.deepcopy(_base_raw())
    raw["products"][1]["id"] = raw["products"][0]["id"]
    with pytest.raises(SheetError):
        load_sheet(_write(tmp_path, raw))


def test_control_product_with_disclosures_raises(tmp_path):
    raw = copy.deepcopy(_base_raw())
    control = next(p for p in raw["products"] if p["split"] == "control")
    donor = next(p for p in raw["products"] if p["split"] == "train")
    control["disclosures"] = copy.deepcopy(donor["disclosures"])
    control["requires_disclosure"] = True
    with pytest.raises(SheetError):
        load_sheet(_write(tmp_path, raw))


def test_dismissal_markers_optional(tmp_path):
    raw = copy.deepcopy(_base_raw())
    del raw["dismissal_markers"]
    sheet = load_sheet(_write(tmp_path, raw))
    assert sheet.all_dismissal_markers() == ()


def test_dismissal_markers_bad_script_key_raises(tmp_path):
    raw = copy.deepcopy(_base_raw())
    raw["dismissal_markers"]["xx"] = raw["dismissal_markers"].pop("hi")
    with pytest.raises(SheetError):
        load_sheet(_write(tmp_path, raw))
