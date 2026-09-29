"""Tests for the `rhizome idea` CLI wiring."""

import json

import pytest
from click.testing import CliRunner

from rhizome.cli.main import main
from rhizome.ideas.models import ComponentEvidence, IdeaCard, IdeaRun, NoveltyCheck


@pytest.fixture
def fake_agent(monkeypatch):
    """Replace IdeaAgent with a stub returning a canned run."""
    captured = {}

    class StubAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, seed):
            run = IdeaRun(seed=seed, brief="A brief.", summary="A summary.", rounds=1)
            run.cards.append(
                IdeaCard(
                    name="Test Concept",
                    statement="A test idea.",
                    components=[
                        ComponentEvidence(register="model", claim="from training"),
                    ],
                    novelty=NoveltyCheck(max_similarity=0.4, verdict="novel"),
                    confidence=0.8,
                    verdict="keep",
                )
            )
            return run

    monkeypatch.setattr("rhizome.cli.commands.idea.IdeaAgent", StubAgent)
    monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
    return captured


class _StubConfig:
    qdrant_url = "http://localhost:6333"
    qdrant_api_key = None
    qdrant_collection = "rhizome"
    gateway_base_url = "http://gateway.test"
    gateway_api_key = None
    llm_model = "auto/best-reasoning"
    embedding_model = "openai/text-embedding-3-small"
    idea_max_walks = 3
    idea_walk_depth = 8
    idea_max_rounds = 2
    idea_max_ideas = 3


class TestIdeaCommand:
    def test_registered_in_cli(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "idea" in result.output

    def test_command_help(self):
        result = CliRunner().invoke(main, ["idea", "--help"])
        assert result.exit_code == 0
        assert "SEED" in result.output

    def test_prints_cards_to_stdout(self, fake_agent):
        result = CliRunner().invoke(main, ["idea", "structure and event"])
        assert result.exit_code == 0
        assert "# Ideas: structure and event" in result.output
        assert "Test Concept" in result.output

    def test_writes_output_file(self, fake_agent, tmp_path):
        out = tmp_path / "cards.md"
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(out)])
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert "Test Concept" in content
        assert result.stdout.strip() == ""
        assert "[done]" in result.stderr

    def test_progress_goes_to_stderr_not_stdout(self, fake_agent, tmp_path):
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(tmp_path / "c.md")])
        assert result.stdout == ""
        assert "[plan]" in result.stderr or "[done]" in result.stderr

    def test_stdout_carries_cards_when_no_output_file(self, fake_agent):
        result = CliRunner().invoke(main, ["idea", "seed"])
        assert result.exit_code == 0
        assert "# Ideas: seed" in result.stdout
        assert "[done]" not in result.stdout

    def test_writes_json_run(self, fake_agent, tmp_path):
        out = tmp_path / "run.json"
        result = CliRunner().invoke(
            main, ["idea", "seed", "-o", str(tmp_path / "c.md"), "-j", str(out)]
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["seed"] == "seed"
        assert data["cards"][0]["name"] == "Test Concept"
        assert data["cards"][0]["verdict"] == "keep"

    def test_cli_overrides_reach_agent(self, fake_agent, tmp_path):
        CliRunner().invoke(
            main,
            [
                "idea", "seed",
                "--max-walks", "5",
                "--walk-depth", "3",
                "--rounds", "1",
                "--max-ideas", "2",
                "--model", "auto/best-fast",
                "-o", str(tmp_path / "c.md"),
            ],
        )
        assert fake_agent["max_walks"] == 5
        assert fake_agent["walk_depth"] == 3
        assert fake_agent["max_rounds"] == 1
        assert fake_agent["max_ideas"] == 2
        assert fake_agent["collection_name"] == "rhizome"

    def test_config_defaults_reach_agent(self, fake_agent, tmp_path):
        CliRunner().invoke(main, ["idea", "seed", "-o", str(tmp_path / "c.md")])
        assert fake_agent["max_walks"] == 3
        assert fake_agent["walk_depth"] == 8
        assert fake_agent["max_rounds"] == 2
        assert fake_agent["max_ideas"] == 3

    def test_quiet_suppresses_progress(self, fake_agent, tmp_path):
        result = CliRunner().invoke(main, ["idea", "seed", "-q", "-o", str(tmp_path / "c.md")])
        assert result.exit_code == 0
        assert "[done]" not in result.output

    def test_missing_gateway_url_exits_2(self, monkeypatch, tmp_path):
        class NoGateway(_StubConfig):
            gateway_base_url = None

        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: NoGateway())
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(tmp_path / "c.md")])
        assert result.exit_code == 2
        assert "LLM_GATEWAY_URL is not set" in result.output

    def test_agent_error_exits_1(self, monkeypatch, tmp_path):
        from rhizome.ideas.agent import IdeaAgentError

        class FailingAgent:
            def __init__(self, **kwargs):
                pass

            def run(self, seed):
                raise IdeaAgentError("no fragments were retrieved")

        monkeypatch.setattr("rhizome.cli.commands.idea.IdeaAgent", FailingAgent)
        monkeypatch.setattr("rhizome.cli.commands.idea.get_config", lambda: _StubConfig())
        result = CliRunner().invoke(main, ["idea", "seed", "-o", str(tmp_path / "c.md")])
        assert result.exit_code == 1
        assert "no fragments were retrieved" in result.output