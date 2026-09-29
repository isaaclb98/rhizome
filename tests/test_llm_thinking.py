"""Tests for reasoning-model handling: think blocks and truncation recovery."""

import pytest

from rhizome.llm.base import LLMError, TruncatedResponseError
from rhizome.llm.gateway import GatewayLLM, strip_thinking


class RecordingPost:
    """A requests.post double that yields queued (content, finish_reason) pairs."""

    def __init__(self, responses: list[tuple[str, str | None]]):
        self.responses = responses
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append(kwargs["json"])
        content, finish_reason = self.responses[
            min(len(self.calls) - 1, len(self.responses) - 1)
        ]
        return self._Response(content, finish_reason)

    class _Response:
        status_code = 200

        def __init__(self, content, finish_reason):
            self._content = content
            self._finish_reason = finish_reason

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": self._content},
                        "finish_reason": self._finish_reason,
                    }
                ]
            }


def install(monkeypatch, responses):
    """Monkeypatch requests.post in the gateway module and return the recorder."""
    recorder = RecordingPost(responses)
    monkeypatch.setattr("rhizome.llm.gateway.requests.post", recorder)
    return recorder


class TestStripThinking:
    def test_removes_closed_block(self):
        text = "<think>reasoning here</think>{\"a\": 1}"
        assert strip_thinking(text) == '{"a": 1}'

    def test_removes_unterminated_block(self):
        text = "<think>the model was cut off mid-thought"
        assert strip_thinking(text).strip() == ""

    def test_keeps_text_without_block(self):
        assert strip_thinking('{"a": 1}') == '{"a": 1}'

    def test_removes_multiple_blocks(self):
        text = "<think>x</think>middle<think>y</think>end"
        assert strip_thinking(text) == "middleend"

    def test_multiline_block(self):
        text = "<think>\nline one\nline two\n</think>answer"
        assert strip_thinking(text) == "answer"

    def test_empty_string(self):
        assert strip_thinking("") == ""

    def test_none_is_safe(self):
        assert strip_thinking(None) == ""


class TestTruncationDetection:
    def test_length_finish_with_content_raises(self, monkeypatch):
        fake = install(monkeypatch, [('{"a": 1}', "length")])
        with pytest.raises(TruncatedResponseError, match="cut off"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}], max_tokens=100
            )

    def test_length_finish_with_only_thinking_raises_budget_error(self, monkeypatch):
        fake = install(monkeypatch, [("<think>endless reasoning", "length")])
        with pytest.raises(TruncatedResponseError, match="budget on reasoning"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}], max_tokens=100
            )

    def test_stop_finish_returns_content(self, monkeypatch):
        fake = install(monkeypatch, [('{"a": 1}', "stop")])
        result = GatewayLLM(base_url="http://gw.test").complete(
            [{"role": "user", "content": "hi"}]
        )
        assert result == '{"a": 1}'

    def test_thinking_then_json_parses(self, monkeypatch):
        fake = install(monkeypatch, [('<think>pondering\nlots of thoughts</think>{"a": 1}', "stop")])
        result = GatewayLLM(base_url="http://gw.test").complete_json(
            [{"role": "user", "content": "hi"}]
        )
        assert result == {"a": 1}

    def test_missing_finish_reason_is_tolerated(self, monkeypatch):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": '{"a": 1}'}}]}

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        assert GatewayLLM(base_url="http://gw.test").complete_json(
            [{"role": "user", "content": "hi"}]
        ) == {"a": 1}


class TestBudgetRetry:
    def test_truncation_doubles_budget_and_retries(self, monkeypatch):
        fake = install(monkeypatch, [
            ("only reasoning", "length"),
            ('{"a": 1}', "stop"),
        ])
        result = GatewayLLM(base_url="http://gw.test").complete_json(
            [{"role": "user", "content": "hi"}], max_tokens=4096, retries=1
        )
        assert result == {"a": 1}
        assert len(fake.calls) == 2
        assert fake.calls[0]["max_tokens"] == 4096
        assert fake.calls[1]["max_tokens"] == 8192

    def test_retry_restarts_from_original_messages(self, monkeypatch):
        fake = install(monkeypatch, [
            ("only reasoning", "length"),
            ('{"a": 1}', "stop"),
        ])
        GatewayLLM(base_url="http://gw.test").complete_json(
            [{"role": "user", "content": "hi"}], max_tokens=100, retries=1
        )
        assert len(fake.calls[1]["messages"]) == 1
        assert fake.calls[1]["messages"][0]["content"] == "hi"

    def test_budget_capped_at_32768(self, monkeypatch):
        fake = install(monkeypatch, [("reasoning", "length")] * 6)
        with pytest.raises(LLMError, match="never returned valid JSON"):
            GatewayLLM(base_url="http://gw.test").complete_json(
                [{"role": "user", "content": "hi"}], max_tokens=30000, retries=5
            )
        budgets = [c["max_tokens"] for c in fake.calls]
        assert budgets == [30000, 32768, 32768, 32768, 32768, 32768]

    def test_repeated_truncation_gives_up(self, monkeypatch):
        fake = install(monkeypatch, [("reasoning", "length")])
        with pytest.raises(LLMError, match="never returned valid JSON"):
            GatewayLLM(base_url="http://gw.test").complete_json(
                [{"role": "user", "content": "hi"}], max_tokens=100, retries=2
            )
        assert len(fake.calls) == 3

    def test_truncation_error_is_an_llm_error(self):
        assert issubclass(TruncatedResponseError, LLMError)