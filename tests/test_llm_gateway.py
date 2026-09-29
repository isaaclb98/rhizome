"""Tests for the LLM gateway transport and JSON extraction."""

import json

import pytest

from rhizome.llm.base import LLMError
from rhizome.llm.gateway import GatewayEmbedder, GatewayLLM, extract_json


class TestExtractJson:
    def test_bare_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped_object(self):
        text = 'Sure! Here is the JSON you asked for:\n{"a": [1, 2]}\nHope that helps.'
        assert extract_json(text) == {"a": [1, 2]}

    def test_top_level_array(self):
        assert extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]

    def test_braces_inside_strings(self):
        text = '{"claim": "the set {x} is open", "n": 1}'
        assert extract_json(text) == {"claim": "the set {x} is open", "n": 1}

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, {"deep": true}]}}'
        assert extract_json(text) == {"outer": {"inner": [1, {"deep": True}]}}

    def test_prose_with_decoy_then_real(self):
        text = 'Consider {"not": "this"}... actually: {"real": 1}'
        result = extract_json(text)
        assert isinstance(result, dict)

    def test_empty_raises(self):
        with pytest.raises(LLMError):
            extract_json("")

    def test_unparseable_raises(self):
        with pytest.raises(LLMError):
            extract_json("no json here at all")

    def test_unbalanced_raises(self):
        with pytest.raises(LLMError):
            extract_json('{"a": 1')


class TestGatewayLLM:
    def test_complete_sends_payload(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "hello"}}]}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            return FakeResponse()

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        client = GatewayLLM(base_url="http://gw.test", model="m1")
        result = client.complete([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=99)

        assert result == "hello"
        assert captured["url"] == "http://gw.test/v1/chat/completions"
        assert captured["payload"]["model"] == "m1"
        assert captured["payload"]["temperature"] == 0.3
        assert captured["payload"]["max_tokens"] == 99

    def test_complete_sets_auth_header_when_key_present(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, **kwargs):
            captured["headers"] = kwargs["headers"]
            return FakeResponse()

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        GatewayLLM(base_url="http://gw.test", api_key="secret").complete(
            [{"role": "user", "content": "hi"}]
        )
        assert captured["headers"]["Authorization"] == "Bearer secret"

    def test_complete_no_auth_header_without_key(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, **kwargs):
            captured["headers"] = kwargs["headers"]
            return FakeResponse()

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        GatewayLLM(base_url="http://gw.test").complete([{"role": "user", "content": "hi"}])
        assert "Authorization" not in captured["headers"]

    def test_http_error_raises(self, monkeypatch):
        class FakeResponse:
            status_code = 500
            text = "boom"

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(LLMError, match="500"):
            GatewayLLM(base_url="http://gw.test").complete([{"role": "user", "content": "hi"}])

    def test_empty_completion_raises(self, monkeypatch):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "   "}}]}

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(LLMError, match="empty"):
            GatewayLLM(base_url="http://gw.test").complete([{"role": "user", "content": "hi"}])

    def test_malformed_response_shape_raises(self, monkeypatch):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"unexpected": True}

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(LLMError):
            GatewayLLM(base_url="http://gw.test").complete([{"role": "user", "content": "hi"}])

    def test_complete_json_reprompts_on_bad_json(self, monkeypatch):
        calls = []

        class FakeResponse:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, **kwargs):
            calls.append(kwargs["json"]["messages"])
            return FakeResponse("not json at all" if len(calls) == 1 else '{"a": 1}')

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        result = GatewayLLM(base_url="http://gw.test").complete_json(
            [{"role": "user", "content": "give json"}], retries=1
        )
        assert result == {"a": 1}
        assert len(calls) == 2
        assert "not valid JSON" in calls[1][-1]["content"]

    def test_complete_json_gives_up_after_retries(self, monkeypatch):
        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "prose only"}}]}

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(LLMError, match="never returned valid JSON"):
            GatewayLLM(base_url="http://gw.test").complete_json(
                [{"role": "user", "content": "give json"}], retries=1
            )


class TestGatewayEmbedder:
    def test_embed_orders_by_index(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "data": [
                        {"index": 1, "embedding": [0.2, 0.2]},
                        {"index": 0, "embedding": [0.1, 0.1]},
                    ]
                }

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            return FakeResponse()

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        vectors = GatewayEmbedder(base_url="http://gw.test", model="emb1").embed(["a", "b"])

        assert vectors == [[0.1, 0.1], [0.2, 0.2]]
        assert captured["url"] == "http://gw.test/v1/embeddings"
        assert captured["payload"]["model"] == "emb1"

    def test_embed_batches(self, monkeypatch):
        calls = []

        class FakeResponse:
            status_code = 200

            def __init__(self, n):
                self._n = n

            def json(self):
                return {"data": [{"index": i, "embedding": [0.0]} for i in range(self._n)]}

        def fake_post(url, **kwargs):
            calls.append(len(kwargs["json"]["input"]))
            return FakeResponse(len(kwargs["json"]["input"]))

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        vectors = GatewayEmbedder(base_url="http://gw.test", batch_size=2).embed(
            ["a", "b", "c", "d", "e"]
        )
        assert len(vectors) == 5
        assert calls == [2, 2, 1]

    def test_embed_empty_list_skips_request(self, monkeypatch):
        def fake_post(*a, **k):
            raise AssertionError("should not call the API for an empty list")

        monkeypatch.setattr("rhizome.llm.gateway.requests.post", fake_post)
        assert GatewayEmbedder(base_url="http://gw.test").embed([]) == []

    def test_embed_count_mismatch_raises(self, monkeypatch):
        from rhizome.embedder.base import EmbeddingError

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"index": 0, "embedding": [0.1]}]}

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(EmbeddingError, match="expected 2"):
            GatewayEmbedder(base_url="http://gw.test").embed(["a", "b"])

    def test_embed_http_error_raises(self, monkeypatch):
        from rhizome.embedder.base import EmbeddingError

        class FakeResponse:
            status_code = 400
            text = "bad model"

        monkeypatch.setattr(
            "rhizome.llm.gateway.requests.post", lambda *a, **k: FakeResponse()
        )
        with pytest.raises(EmbeddingError, match="400"):
            GatewayEmbedder(base_url="http://gw.test").embed(["a"])