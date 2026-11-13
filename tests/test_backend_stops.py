from htd.backends import stop_token_ids


class FakeTok:
    unk_token_id = 0
    eos_token_id = 6
    _vocab = {"<|END_RESPONSE|>": 261001, "<|END_OF_TURN_TOKEN|>": 6}

    def convert_tokens_to_ids(self, t):
        return self._vocab.get(t, self.unk_token_id)


def test_stops_on_end_response_end_of_turn_and_generation_eos():
    # Tiny Aya: generation_config eos is 3, but replies end with END_RESPONSE then END_OF_TURN.
    assert stop_token_ids(FakeTok(), 3) == [3, 6, 261001]


def test_accepts_list_eos_and_dedupes():
    assert stop_token_ids(FakeTok(), [6, 3]) == [3, 6, 261001]


def test_skips_unknown_marker():
    class NoMarker(FakeTok):
        _vocab = {}

    assert stop_token_ids(NoMarker(), None) == [6]
