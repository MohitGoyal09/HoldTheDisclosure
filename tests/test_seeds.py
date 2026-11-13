import json

import pytest

from htd.seeds import SeedError, load_seeds

SLOTS = ("t2_brevity", "t3_social_proof", "t4_urgency", "t5_override")


def test_load_default_seeds_has_all_slots():
    seeds = load_seeds()
    for slot in SLOTS:
        assert len(seeds[slot]) >= 1
        for s in seeds[slot]:
            assert s.slot == slot
            assert set(s.text) == {"en", "hi", "hinglish"}


def test_missing_script_key_raises(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"id": "a", "slot": "x", "status": "draft", "en": "hi"}) + "\n")
    with pytest.raises(SeedError):
        load_seeds(p)


def test_every_seed_has_a_valid_split():
    seeds = load_seeds()
    for group in seeds.values():
        for s in group:
            assert s.split in {"train", "test", "heldout", "test2"}


def test_filter_by_split_keeps_only_that_split():
    train = load_seeds(split="train")
    assert train and all(s.split == "train" for g in train.values() for s in g)
    assert "heldout_anger" not in train


def test_bad_split_raises(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"id": "a", "slot": "x", "split": "oops", "en": "a", "hi": "b", "hinglish": "c"}) + "\n")
    with pytest.raises(SeedError):
        load_seeds(p)
