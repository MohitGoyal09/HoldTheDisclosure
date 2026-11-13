from htd.dialogues import build_dialogues, load_dialogues, save_dialogues
from htd.seeds import load_seeds
from htd.sheet import load_sheet

PRODUCTS = ["equity_sip", "elss", "fd", "credit_card"]


def _build(**kw):
    return build_dialogues(load_sheet(), load_seeds(), PRODUCTS, **kw)


def test_counts_and_shape():
    ds = _build(n_per_script=5)
    assert len(ds) == 15
    for d in ds:
        assert len(d.user_turns) == 5
        assert len(d.tactic_ids) == 4


def test_ids_and_round_robin_products():
    ds = [d for d in _build(n_per_script=6) if d.script == "en"]
    assert ds[0].dialogue_id == "pilot-000-en"
    assert [d.product_id for d in ds] == [PRODUCTS[i % 4] for i in range(6)]


def test_matched_across_scripts():
    ds = _build(n_per_script=8)
    by = {}
    for d in ds:
        by.setdefault(d.dialogue_id.rsplit("-", 1)[0], {})[d.script] = d
    for group in by.values():
        assert len({(d.product_id, d.tactic_ids) for d in group.values()}) == 1
        assert len(group) == 3


def test_turn_one_is_opening_and_turns_use_seeds():
    sheet, seeds = load_sheet(), load_seeds()
    d = _build(n_per_script=1, scripts=("hi",))[0]
    assert d.user_turns[0] in sheet.product(d.product_id).openings["hi"]
    texts = {s.id: s.text["hi"] for ss in seeds.values() for s in ss}
    assert list(d.user_turns[1:]) == [texts[t] for t in d.tactic_ids]


def test_deterministic_and_seed_sensitive():
    assert _build(n_per_script=10) == _build(n_per_script=10)
    assert _build(n_per_script=10) != _build(n_per_script=10, rng_seed=1)


def test_roundtrip(tmp_path):
    ds = _build(n_per_script=2)
    p = tmp_path / "d.jsonl"
    save_dialogues(ds, p)
    assert load_dialogues(p) == ds


# ---------- build_split (DESIGN D2, D3) ----------

import pytest  # noqa: E402

from collections import Counter  # noqa: E402

from htd.dialogues import SPLIT_PRESETS, WILLIAMS_4, build_split, williams_row  # noqa: E402

SHEET = load_sheet()
SEEDS = load_seeds()


def _seed_ids(split):
    return {s.id for ss in load_seeds(split=split).values() for s in ss}


def _split_seed_ids(ds):
    return {t for d in ds for t in d.tactic_ids}


@pytest.mark.parametrize("split", list(SPLIT_PRESETS))
def test_split_ids_shape_and_matching(split):
    ds = build_split(SHEET, split, 6)
    assert len(ds) == 18
    assert ds[0].dialogue_id == f"{split}-000-en"
    assert all(d.split == split and len(d.user_turns) == 5 and len(d.tactic_ids) == 4 for d in ds)
    by = {}
    for d in ds:
        by.setdefault(d.dialogue_id.rsplit("-", 1)[0], set()).add((d.product_id, d.tactic_ids))
    assert all(len(v) == 1 for v in by.values())


@pytest.mark.parametrize(
    "split,product_split",
    [("pilot", "train"), ("train", "train"), ("test", "train"), ("heldout", "heldout"),
     ("control_train", "control"), ("control_test", "control"), ("test2", "train"), ("test2_shuffled", "train")],
)
def test_split_products(split, product_split):
    allowed = {p.id for p in SHEET.by_split(product_split)}
    ds = build_split(SHEET, split, 8)
    assert {d.product_id for d in ds} == allowed
    en = [d for d in ds if d.script == "en"]
    ids = sorted(allowed, key=[p.id for p in SHEET.products].index)
    assert [d.product_id for d in en] == [ids[i % len(ids)] for i in range(8)]


def test_pilot_equals_train_except_ids():
    a = [(d.product_id, d.user_turns) for d in build_split(SHEET, "pilot", 5)]
    b = [(d.product_id, d.user_turns) for d in build_split(SHEET, "train", 5)]
    assert a == b


def test_test_split_has_no_seed_in_common_with_train():
    train_ids = _seed_ids("train")
    test_ds = build_split(SHEET, "test", 12)
    assert _split_seed_ids(test_ds) <= _seed_ids("test")
    assert not (_split_seed_ids(test_ds) & train_ids)
    assert not (_split_seed_ids(build_split(SHEET, "train", 40)) & _seed_ids("test"))


def test_heldout_slots_and_seed_splits():
    ds = build_split(SHEET, "heldout", 10)
    for d in ds:
        slots = [t.rsplit("_", 1)[0] for t in d.tactic_ids]
        assert slots == ["heldout_anger", "heldout_authority", "heldout_anger", "heldout_authority"]
    assert _split_seed_ids(ds) <= _seed_ids("heldout")


def test_test_openings_are_held_out_from_train():
    train = build_split(SHEET, "train", 20)
    test = build_split(SHEET, "test", 20)
    train_open = {(d.product_id, d.script, d.user_turns[0]) for d in train}
    test_open = {(d.product_id, d.script, d.user_turns[0]) for d in test}
    assert not (train_open & test_open)


def test_control_slots():
    ds = build_split(SHEET, "control_train", 4)
    assert [t.rsplit("_", 1)[0] for t in ds[0].tactic_ids] == [
        "t2_brevity", "c3_impatience", "t4_urgency", "c5_just_answer"]
    assert _split_seed_ids(build_split(SHEET, "control_test", 6)) <= _seed_ids("test")


def test_distinct_combinations_first():
    ds = [d for d in build_split(SHEET, "train", 6) if d.script == "en"]
    assert len({d.tactic_ids for d in ds}) == 6
    ds = [d for d in build_split(SHEET, "test", 2) if d.script == "en"]
    assert len({d.tactic_ids for d in ds}) == 2


def test_split_deterministic_and_seed_sensitive():
    assert build_split(SHEET, "test", 8) == build_split(SHEET, "test", 8)
    assert build_split(SHEET, "train", 8) != build_split(SHEET, "train", 8, rng_seed=1)


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        build_split(SHEET, "nope", 2)


def test_build_split_with_custom_seeds_path_and_scripts():
    from htd.seeds import DEFAULT_SEEDS_PATH
    ds = build_split(SHEET, "test", 2, scripts=("hi",), seeds_path=DEFAULT_SEEDS_PATH)
    assert {d.script for d in ds} == {"hi"} and len(ds) == 2


def test_product_ids_filter_and_unknown():
    ds = build_split(SHEET, "train", 4, product_ids=["fd"])
    assert {d.product_id for d in ds} == {"fd"}
    with pytest.raises(ValueError):
        build_split(SHEET, "train", 4, product_ids=["ulip"])


SLOTS = ["t2_brevity", "t3_social_proof", "t4_urgency", "t5_override"]


def _slots(d):
    return [t.rsplit("_", 1)[0] for t in d.tactic_ids]


def test_test2_uses_only_new_seeds_in_fixed_order():
    ds = build_split(SHEET, "test2", 40)
    assert all(_slots(d) == SLOTS for d in ds)
    assert _split_seed_ids(ds) <= _seed_ids("test2")
    assert not (_seed_ids("test2") & (_seed_ids("train") | _seed_ids("test") | _seed_ids("heldout")))
    assert len({d.tactic_ids for d in ds if d.script == "en"}) == 40


def test_test2_shuffled_has_same_content_in_williams_order():
    fixed = build_split(SHEET, "test2", 40)
    shuf = build_split(SHEET, "test2_shuffled", 40)
    for i, (f, s) in enumerate(zip(fixed, shuf)):
        assert (f.product_id, f.script, f.user_turns[0]) == (s.product_id, s.script, s.user_turns[0])
        order = WILLIAMS_4[williams_row(i % 40, 4)]
        assert s.tactic_ids == tuple(f.tactic_ids[j] for j in order)
        assert s.user_turns[1:] == tuple(f.user_turns[1 + j] for j in order)


def test_test2_shuffled_balances_position_and_carryover():
    en = [d for d in build_split(SHEET, "test2_shuffled", 40) if d.script == "en"]
    for pos in range(4):
        assert {s: sum(_slots(d)[pos] == s for d in en) for s in SLOTS} == {s: 10 for s in SLOTS}
    pairs = Counter((a, b) for d in en for a, b in zip(_slots(d), _slots(d)[1:]))
    assert len(pairs) == 12 and set(pairs.values()) == {10}


def test_test2_shuffled_order_is_not_tied_to_product():
    en = [d for d in build_split(SHEET, "test2_shuffled", 40) if d.script == "en"]
    for product in {d.product_id for d in en}:
        last = {_slots(d)[3] for d in en if d.product_id == product}
        assert last == set(SLOTS), (product, last)
