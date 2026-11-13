from types import SimpleNamespace

import pytest

from htd.backends import APIBackend


class StubLLM:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = 0, fail_first

    def generate(self, messages):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("rate limited")
        return SimpleNamespace(text=f"  reply to {messages[-1]['content']}  ")


def test_api_backend_keeps_order_and_strips():
    msgs = [[{"role": "user", "content": str(i)}] for i in range(6)]
    assert APIBackend(StubLLM(), workers=3).generate_batch(msgs) == [f"reply to {i}" for i in range(6)]


def test_api_backend_retries_then_raises():
    one = [[{"role": "user", "content": "x"}]]
    assert APIBackend(StubLLM(fail_first=2), retries=2, wait=0).generate_batch(one) == ["reply to x"]
    with pytest.raises(RuntimeError):
        APIBackend(StubLLM(fail_first=5), retries=2, wait=0).generate_batch(one)


def test_api_backend_retries_empty_reply():
    class Empty(StubLLM):
        def generate(self, messages):
            self.calls += 1
            return SimpleNamespace(text="" if self.calls == 1 else "ok")

    assert APIBackend(Empty(), retries=1, wait=0).generate_batch([[{"role": "user", "content": "x"}]]) == ["ok"]
