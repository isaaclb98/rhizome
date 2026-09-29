"""Tests for the gateway transport and the `rhizome idea` command."""

import json

import pytest
from click.testing import CliRunner

from rhizome.cli.commands.idea import (
    build_prompt,
    format_fragments,
    material_path,
    render_material,
)
from rhizome.traversal.config import TraversalConfig
from rhizome.cli.main import main
from rhizome.embedder.base import EmbeddingError
from rhizome.gateway import GatewayEmbedder, GatewayError, GatewayLLM, strip_thinking
from rhizome.traversal.engine import TraversalStep


class RecordingPost:
    """requests.post double yielding queued (content, finish_reason) pairs."""

    def __init__(self, responses):
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
    recorder = RecordingPost(responses)
    monkeypatch.setattr("rhizome.gateway.requests.post", recorder)
    return recorder


class TestStripThinking:
    def test_removes_closed_block(self):
        assert strip_thinking("<think>reasoning</think>thesis") == "thesis"

    def test_removes_unterminated_block(self):
        assert strip_thinking("<think>cut off mid-thought").strip() == ""

    def test_keeps_plain_text(self):
        assert strip_thinking("plain") == "plain"

    def test_multiline_block(self):
        assert strip_thinking("<think>\na\nb\n</think>answer") == "answer"

    def test_none_and_empty(self):
        assert strip_thinking(None) == ""
        assert strip_thinking("") == ""


class TestGatewayLLM:
    def test_complete_returns_content(self, monkeypatch):
        fake = install(monkeypatch, [("a thesis", "stop")])
        result = GatewayLLM(base_url="http://gw.test", model="m1").complete(
            [{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=100
        )
        assert result == "a thesis"
        assert fake.calls[0]["model"] == "m1"
        assert fake.calls[0]["temperature"] == 0.5
        assert fake.calls[0]["max_tokens"] == 100

    def test_strips_thinking_before_returning(self, monkeypatch):
        install(monkeypatch, [("<think>pondering\nmore\n</think>the thesis", "stop")])
        result = GatewayLLM(base_url="http://gw.test").complete(
            [{"role": "user", "content": "hi"}]
        )
        assert result == "the thesis"

    def test_truncation_doubles_budget_and_retries(self, monkeypatch):
        fake = install(monkeypatch, [("only reasoning", "length"), ("thesis", "stop")])
        result = GatewayLLM(base_url="http://gw.test").complete(
            [{"role": "user", "content": "hi"}], max_tokens=4096
        )
        assert result == "thesis"
        assert [c["max_tokens"] for c in fake.calls] == [4096, 8192]

    def test_budget_capped_at_32768(self, monkeypatch):
        fake = install(monkeypatch, [("reasoning", "length")])
        with pytest.raises(GatewayError, match="ceiling"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}], max_tokens=30000, retries=5
            )
        assert [c["max_tokens"] for c in fake.calls] == [30000, 32768]

    def test_empty_completion_raises(self, monkeypatch):
        install(monkeypatch, [("", "stop")])
        with pytest.raises(GatewayError, match="empty completion"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}]
            )

    def test_http_error_raises(self, monkeypatch):
        class Fail:
            status_code = 500
            text = "boom"

        monkeypatch.setattr("rhizome.gateway.requests.post", lambda *a, **k: Fail())
        with pytest.raises(GatewayError, match="500"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}]
            )

    def test_bad_response_shape_raises(self, monkeypatch):
        class Weird:
            status_code = 200

            def json(self):
                return {"unexpected": True}

        monkeypatch.setattr("rhizome.gateway.requests.post", lambda *a, **k: Weird())
        with pytest.raises(GatewayError, match="response shape"):
            GatewayLLM(base_url="http://gw.test").complete(
                [{"role": "user", "content": "hi"}]
            )

    def test_auth_header_only_when_key_set(self, monkeypatch):
        headers_seen = []

        class OK:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]
                }

        def fake_post(url, **kwargs):
            headers_seen.append(kwargs["headers"])
            return OK()

        monkeypatch.setattr("rhizome.gateway.requests.post", fake_post)
        GatewayLLM(base_url="http://gw.test").complete([{"role": "user", "content": "hi"}])
        GatewayLLM(base_url="http://gw.test", api_key="secret").complete(
            [{"role": "user", "content": "hi"}]
        )
        assert "Authorization" not in headers_seen[0]
        assert headers_seen[1]["Authorization"] == "Bearer secret"


class TestGatewayEmbedder:
    def test_embed_orders_by_index(self, monkeypatch):
        captured = {}

        class OK:
            status_code = 200

            def json(self):
                return {
                    "data": [
                        {"index": 1, "embedding": [0.2]},
                        {"index": 0, "embedding": [0.1]},
                    ]
                }

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            return OK()

        monkeypatch.setattr("rhizome.gateway.requests.post", fake_post)
        vectors = GatewayEmbedder(base_url="http://gw.test", model="emb1").embed(["a", "b"])
        assert vectors == [[0.1], [0.2]]
        assert captured["url"] == "http://gw.test/v1/embeddings"
        assert captured["payload"]["model"] == "emb1"

    def test_embed_batches(self, monkeypatch):
        sizes = []

        class OK:
            status_code = 200

            def __init__(self, n):
                self._n = n

            def json(self):
                return {"data": [{"index": i, "embedding": [0.0]} for i in range(self._n)]}

        def fake_post(url, **kwargs):
            sizes.append(len(kwargs["json"]["input"]))
            return OK(len(kwargs["json"]["input"]))

        monkeypatch.setattr("rhizome.gateway.requests.post", fake_post)
        vectors = GatewayEmbedder(base_url="http://gw.test", batch_size=2).embed(
            ["a", "b", "c", "d", "e"]
        )
        assert len(vectors) == 5
        assert sizes == [2, 2, 1]

    def test_empty_list_skips_request(self, monkeypatch):
        monkeypatch.setattr(
            "rhizome.gateway.requests.post",
            lambda *a, **k: pytest.fail("should not call API for empty list"),
        )
        assert GatewayEmbedder(base_url="http://gw.test").embed([]) == []

    def test_count_mismatch_raises(self, monkeypatch):
        class OK:
            status_code = 200

            def json(self):
                return {"data": [{"index": 0, "embedding": [0.1]}]}

        monkeypatch.setattr("rhizome.gateway.requests.post", lambda *a, **k: OK())
        with pytest.raises(EmbeddingError, match="expected 2"):
            GatewayEmbedder(base_url="http://gw.test").embed(["a", "b"])

    def test_http_error_raises(self, monkeypatch):
        class Fail:
            status_code = 400
            text = "bad model"

        monkeypatch.setattr("rhizome.gateway.requests.post", lambda *a, **k: Fail())
        with pytest.raises(EmbeddingError, match="400"):
            GatewayEmbedder(base_url="http://gw.test").embed(["a"])


def make_step(title="Anti-Oedipus", text="Deterritorialization occurs.", jump=False):
    return TraversalStep(
        chunk_id=f"{title.replace(' ', '-')}-001",
        text=text,
        article_title=title,
        article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        depth=1,
        similarity=0.67,
        forced_jump=jump,
        candidates=[],
    )


class TestPromptBuilding:
    def test_fragments_numbered_in_order(self):
        out = format_fragments([make_step("A"), make_step("B")])
        assert out.index("[1] A") < out.index("[2] B")
        assert "---" in out

    def test_forced_jump_annotated(self):
        out = format_fragments([make_step("A", jump=True)])
        assert "forced jump" in out

    def test_urls_included_for_citation(self):
        out = format_fragments([make_step("Henri Bergson")])
        assert "https://en.wikipedia.org/wiki/Henri_Bergson" in out

    def test_prompt_contains_seed_and_text(self):
        prompt = build_prompt("structure and event", [make_step(text="the fragment body")])
        assert "structure and event" in prompt
        assert "the fragment body" in prompt
        assert "one thesis" in prompt

    def test_prompt_instructs_against_listing(self):
        prompt = build_prompt("seed", [make_step()])
        assert "Do not list ideas" in prompt


class _StubConfig:
    qdrant_url = "http://localhost:6333"
    qdrant_api_key = None
    qdrant_collection = "rhizome"
    llm_gateway_url = "http://gateway.test"
    llm_gateway_api_key = None
    llm_model = "agy/claude-opus-4-6-thinking-high"
    embedding_model = "openai/text-embedding-3-small"
    llm_temperature = 0.9
    default_depth = 8
    epsilon = 0.1
    top_k = 20
    temperature = 1.0
    max_same_article_consecutive = 2


@pytest.fixture
def stubbed(monkeypatch):
    """Stub config, traversal, collection check and LLM; capture what reached them."""
    captured = {}

    monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
    monkeypatch.setattr(
        "rhizome.cli.commands.idea.CollectionManager",
        lambda **kwargs: type("CM", (), {"collection_exists": lambda self, name: True})(),
    )

    def fake_traverse(self, concept):
        captured["concept"] = concept
        captured["config"] = self.config
        return [make_step("Anti-Oedipus"), make_step("Henri Bergson", jump=True)]

    monkeypatch.setattr("rhizome.cli.commands.idea.TraversalEngine.traverse", fake_traverse)

    class StubLLM:
        def __init__(self, base_url, model=None, api_key=None, timeout=600):
            self.model = model
            captured["llm_model"] = model
            captured["llm_base_url"] = base_url

        def complete(self, messages, temperature=None, max_tokens=8192):
            captured["prompt"] = messages[0]["content"]
            captured["llm_temperature"] = temperature
            return "The thesis text."

    monkeypatch.setattr("rhizome.cli.commands.idea.GatewayLLM", StubLLM)
    monkeypatch.setattr(
        "rhizome.cli.commands.idea.GatewayEmbedder", lambda **kwargs: object()
    )
    return captured


class TestMaterialSidecar:
    def test_path_replaces_md_suffix(self):
        assert material_path("/tmp/thesis.md") == "/tmp/thesis.material.md"

    def test_path_appends_when_no_suffix(self):
        assert material_path("/tmp/thesis") == "/tmp/thesis.material.md"

    def test_path_handles_nested_dirs(self):
        assert material_path("out/runs/a.md") == "out/runs/a.material.md"

    def test_path_keeps_dots_in_name(self):
        assert material_path("my.draft.md") == "my.draft.material.md"

    def test_renders_seed_and_knobs(self):
        config = TraversalConfig(depth=12, epsilon=0.4, top_k=30, temperature=1.5,
                                 max_same_article_consecutive=2)
        out = render_material("structure and event", [make_step(), make_step(jump=True)], config)
        assert "# Material: structure and event" in out
        assert "depth=12" in out
        assert "epsilon=0.4" in out
        assert "top_k=30" in out
        assert "2 fragment(s)" in out
        assert "1 forced jump(s)" in out

    def test_renders_each_step_with_provenance(self):
        config = TraversalConfig()
        out = render_material("seed", [make_step("Anti-Oedipus", "the body text")], config)
        assert "[1] Anti-Oedipus" in out
        assert "https://en.wikipedia.org/wiki/Anti-Oedipus" in out
        assert "0.670" in out
        assert "the body text" in out

    def test_marks_forced_jumps(self):
        out = render_material("seed", [make_step("A", jump=True)], TraversalConfig())
        assert "forced jump" in out


class TestIdeaCommand:
    def test_registered(self):
        assert "idea" in CliRunner().invoke(main, ["--help"]).output

    def test_help_lists_traversal_knobs(self):
        out = CliRunner().invoke(main, ["idea", "--help"]).output
        for flag in ("--depth", "--epsilon", "--top-k", "--temperature",
                     "--max-same-article-consecutive"):
            assert flag in out

    def test_prints_thesis_to_stdout(self, stubbed):
        result = CliRunner().invoke(main, ["idea", "structure and event"])
        assert result.exit_code == 0
        assert "# structure and event" in result.output
        assert "The thesis text." in result.output

    def test_progress_goes_to_stderr(self, stubbed):
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert "Walking:" in result.output
        assert "Walking:" not in result.stdout

    def test_writes_output_file(self, stubbed, tmp_path):
        out = tmp_path / "thesis.md"
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(out)])
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert content.startswith("# seed")
        assert "The thesis text." in content
        assert result.stdout == ""

    def test_knob_overrides_reach_engine(self, stubbed):
        CliRunner().invoke(
            main,
            ["idea", "seed", "--depth", "12", "--epsilon", "0.4",
             "--top-k", "30", "--temperature", "1.5",
             "--max-same-article-consecutive", "3"],
        )
        config = stubbed["config"]
        assert config.depth == 12
        assert config.epsilon == 0.4
        assert config.top_k == 30
        assert config.temperature == 1.5
        assert config.max_same_article_consecutive == 3
        assert config.collection_name == "rhizome"

    def test_config_defaults_reach_engine(self, stubbed):
        CliRunner().invoke(main, ["idea", "seed"])
        config = stubbed["config"]
        assert config.depth == 8
        assert config.epsilon == 0.1
        assert config.temperature == 1.0

    def test_model_override_reaches_llm(self, stubbed):
        CliRunner().invoke(main, ["idea", "seed", "--model", "auto/best-fast"])
        assert stubbed["llm_model"] == "auto/best-fast"

    def test_default_model_used(self, stubbed):
        CliRunner().invoke(main, ["idea", "seed"])
        assert stubbed["llm_model"] == "agy/claude-opus-4-6-thinking-high"

    def test_llm_temperature_passed(self, stubbed):
        CliRunner().invoke(main, ["idea", "seed", "--llm-temperature", "0.3"])
        assert stubbed["llm_temperature"] == 0.3

    def test_prompt_carries_walk_material(self, stubbed):
        CliRunner().invoke(main, ["idea", "seed"])
        assert "Anti-Oedipus" in stubbed["prompt"]
        assert "Henri Bergson" in stubbed["prompt"]

    def test_material_off_by_default(self, stubbed, tmp_path):
        out = tmp_path / "thesis.md"
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert not (tmp_path / "thesis.material.md").exists()
        assert "Material written" not in result.output

    def test_save_material_writes_sidecar(self, stubbed, tmp_path):
        out = tmp_path / "thesis.md"
        result = CliRunner().invoke(
            main, ["idea", "seed", "-o", str(out), "--save-material"]
        )
        assert result.exit_code == 0
        side = tmp_path / "thesis.material.md"
        assert side.exists()
        content = side.read_text(encoding="utf-8")
        assert "# Material: seed" in content
        assert "Anti-Oedipus" in content
        assert "Henri Bergson" in content
        assert "Material written to:" in result.output

    def test_save_material_records_knobs_used(self, stubbed, tmp_path):
        out = tmp_path / "thesis.md"
        CliRunner().invoke(
            main,
            ["idea", "seed", "-o", str(out), "--save-material",
             "--depth", "12", "--epsilon", "0.4", "--temperature", "1.5"],
        )
        content = (tmp_path / "thesis.material.md").read_text(encoding="utf-8")
        assert "depth=12" in content
        assert "epsilon=0.4" in content
        assert "temperature=1.5" in content

    def test_no_save_material_flag_disables_it(self, stubbed, tmp_path):
        out = tmp_path / "thesis.md"
        result = CliRunner().invoke(
            main, ["idea", "seed", "-o", str(out), "--no-save-material"]
        )
        assert result.exit_code == 0
        assert not (tmp_path / "thesis.material.md").exists()

    def test_save_material_without_output_aborts(self, stubbed):
        result = CliRunner().invoke(main, ["idea", "seed", "--save-material"])
        assert result.exit_code != 0
        assert "--save-material requires -o" in result.output

    def test_material_written_before_llm_call(self, monkeypatch, tmp_path):
        """An LLM failure must not discard the expensive walk."""
        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.CollectionManager",
            lambda **kwargs: type("CM", (), {"collection_exists": lambda self, n: True})(),
        )
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.TraversalEngine.traverse",
            lambda self, c: [make_step("Anti-Oedipus")],
        )
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.GatewayEmbedder", lambda **kwargs: object()
        )

        class FailingLLM:
            model = "failing-model"

            def __init__(self, **kwargs):
                pass

            def complete(self, messages, **kwargs):
                raise GatewayError("gateway error 503")

        monkeypatch.setattr("rhizome.cli.commands.idea.GatewayLLM", FailingLLM)
        out = tmp_path / "thesis.md"
        result = CliRunner().invoke(
            main, ["idea", "seed", "-o", str(out), "--save-material"]
        )
        assert result.exit_code != 0
        assert (tmp_path / "thesis.material.md").exists()
        assert not out.exists()

    def test_help_documents_flag_as_off_by_default(self):
        out = CliRunner().invoke(main, ["idea", "--help"]).output
        assert "--save-material" in out
        assert "--no-save-material" in out

    def test_missing_gateway_url_aborts(self, monkeypatch):
        class NoGateway(_StubConfig):
            llm_gateway_url = None

        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: NoGateway())
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert result.exit_code != 0
        assert "LLM_GATEWAY_URL is not set" in result.output

    def test_missing_collection_aborts(self, monkeypatch):
        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.CollectionManager",
            lambda **kwargs: type("CM", (), {"collection_exists": lambda self, n: False})(),
        )
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_empty_path_aborts(self, monkeypatch):
        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.CollectionManager",
            lambda **kwargs: type("CM", (), {"collection_exists": lambda self, n: True})(),
        )
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.TraversalEngine.traverse", lambda self, c: []
        )
        monkeypatch.setattr(
            "rhizome.cli.commands.idea.GatewayEmbedder", lambda **kwargs: object()
        )
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert result.exit_code != 0
        assert "No path generated" in result.output

    def test_llm_error_aborts(self, monkeypatch, stubbed):
        class FailingLLM:
            model = "failing-model"

            def __init__(self, **kwargs):
                pass

            def complete(self, messages, **kwargs):
                raise GatewayError("gateway error 503: unavailable")

        monkeypatch.setattr("rhizome.cli.commands.idea.GatewayLLM", FailingLLM)
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert result.exit_code != 0
        assert "LLM error" in result.output
