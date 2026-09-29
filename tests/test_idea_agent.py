"""Tests for the idea agent loop with a scripted LLM and fake vector store."""

import pytest

from rhizome.embedder.base import EmbeddingError
from rhizome.ideas.agent import (
    IdeaAgent,
    IdeaAgentError,
    _best_survivor,
    _best_quote,
    _validate_components,
)
from rhizome.ideas.models import (
    ComponentEvidence,
    Fragment,
    IdeaCard,
    IdeaRun,
    NoveltyCheck,
    WalkPlan,
)


class FakeEmbedder:
    """Returns a deterministic vector derived from the text length."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[(len(t) % 7 + 1) / 10.0] * self.dim for t in texts]


class FailingEmbedder:
    def embed(self, texts):
        raise EmbeddingError("embedder down")


class FakeVectorStore:
    """Search returns canned hits; scores configurable per call."""

    def __init__(self, scores: list[float] | None = None):
        self.scores = list(scores or [0.5])
        self.search_calls: list[dict] = []

    def search(self, query_vector, top_k=5, query_filter=None, with_vector=True):
        self.search_calls.append({"top_k": top_k, "with_vector": with_vector})
        score = self.scores[min(len(self.search_calls) - 1, len(self.scores) - 1)]
        return [
            {
                "id": f"hit-{i}",
                "score": score,
                "payload": {
                    "id": f"hit-{i}",
                    "text": "Nearest corpus text about the thing.",
                    "article_title": f"Nearest Article {i}",
                    "article_url": f"https://en.wikipedia.org/wiki/Nearest_{i}",
                },
                "vector": None,
            }
            for i in range(min(top_k, 3))
        ]

    def search_excluding(self, query_vector, exclude_ids, top_k=5, query_filter=None, with_vector=True):
        return self.search(query_vector, top_k=top_k, with_vector=with_vector)


STAGE_MARKERS = [
    ("plan_initial", "Propose the first round"),
    ("plan_refine", "Propose the next round"),
    ("walk_assess", "A traversal just completed"),
    ("synthesize", "Forge at most"),
    ("audit", "model-register claims were made"),
    ("critique", "Judge these forged concepts"),
    ("summarize", "the closing assessment"),
]


class ScriptedLLM:
    """Routes replies by agent stage rather than by call order.

    A positional queue is brittle: the refinement stage fires only when the
    walk budget is not yet spent, so any test that leaves headroom shifts every
    later stage. Dispatching on prompt content makes each stage independent and
    also verifies the agent reaches the intended prompt at all.
    """

    def __init__(self, replies: dict):
        self.replies = dict(replies)
        self.stages: list[str] = []

    def complete(self, messages, *, temperature=0.7, max_tokens=4096):
        raise AssertionError("agent should use complete_json")

    def complete_json(self, messages, *, temperature=0.7, max_tokens=4096, retries=1):
        stage = self._detect(messages)
        self.stages.append(stage)
        value = self.replies.get(stage, {})
        if isinstance(value, Exception):
            raise value
        return value

    @staticmethod
    def _detect(messages) -> str:
        user_text = messages[-1]["content"] if messages else ""
        for stage, marker in STAGE_MARKERS:
            if marker in user_text:
                return stage
        return "unknown"


def default_replies(**overrides):
    """Standard happy-path replies for every stage; override any stage."""
    replies = {
        "plan_initial": PLAN_REPLY,
        "plan_refine": {"walks": []},
        "walk_assess": ASSESS_REPLY,
        "synthesize": SYNTH_REPLY,
        "audit": AUDIT_REPLY,
        "critique": CRITIQUE_REPLY,
        "summarize": SUMMARY_REPLY,
    }
    replies.update(overrides)
    return replies


def make_step(chunk_id="c-1", title="Anti-Oedipus", text="Deterritorialization occurs.", jump=False):
    from rhizome.traversal.engine import TraversalStep

    return TraversalStep(
        chunk_id=chunk_id,
        text=text,
        article_title=title,
        article_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        depth=1,
        similarity=0.7,
        forced_jump=jump,
        candidates=[],
    )


PLAN_REPLY = {
    "brief": "The seed hides a tension between structure and event.",
    "walks": [
        {"seed": "structural causality", "rationale": "home turf", "depth": 4},
        {"seed": "Mongol mythology", "rationale": "deliberately foreign", "depth": 4},
    ],
}

ASSESS_REPLY = {"note": "Collision between Althusser and Bergson. Worth pursuing.", "next_seed_hints": ["duration"]}

SYNTH_REPLY = {
    "ideas": [
        {
            "name": "Structured Duration",
            "statement": "Duration as an ideological apparatus.",
            "composition": "Bergson's duration supplies the temporal component; Althusser supplies the structural one.",
            "components": [
                {
                    "register": "corpus",
                    "claim": "Deterritorialization is simultaneous with reterritorialization.",
                    "chunk_id": "c-1",
                    "quote": "Deterritorialization occurs.",
                },
                {"register": "model", "claim": "Bergson influenced Deleuze's reading of Spinoza."},
            ],
            "stakes": "Makes it possible to ask when a structure becomes an event.",
            "objection": "It may just restate process philosophy.",
            "walk_ids": [1],
        }
    ]
}

AUDIT_REPLY = {
    "audits": [
        {"id": "c0", "verdict": "corroborated", "note": "The passages confirm the Spinoza reading."}
    ]
}

CRITIQUE_REPLY = {
    "verdicts": [
        {
            "name": "Structured Duration",
            "is_tracing_of": "",
            "verdict": "keep",
            "confidence": 0.8,
            "reasoning": "Components require one another.",
        }
    ]
}

SUMMARY_REPLY = {"summary": "The run found one strong concept."}


def build_agent(replies=None, store=None, embedder=None, **kwargs):
    replies = default_replies() if replies is None else replies
    llm = ScriptedLLM(replies)
    store = store or FakeVectorStore()
    embedder = embedder or FakeEmbedder()
    agent = IdeaAgent(
        llm=llm,
        embedder=embedder,
        vector_store=store,
        collection_name="rhizome",
        **kwargs,
    )
    return agent, llm, store, embedder


@pytest.fixture
def fake_traverse(monkeypatch):
    """Replace TraversalEngine.traverse with a canned path."""
    calls = []

    def fake_traverse(self, starting_concept):
        calls.append((starting_concept, self.config.depth, self.config.epsilon))
        return [
            make_step("c-1", "Anti-Oedipus", "Deterritorialization occurs."),
            make_step("c-2", "Bergson", "Duration is heterogeneous multiplicity.", jump=True),
        ]

    monkeypatch.setattr("rhizome.ideas.agent.TraversalEngine.traverse", fake_traverse)
    return calls


class TestAgentRun:
    def test_full_run_produces_cards(self, fake_traverse):
        agent, llm, store, _ = build_agent(max_walks=2, max_rounds=1)
        run = agent.run("structure and event")

        assert isinstance(run, IdeaRun)
        assert run.seed == "structure and event"
        assert run.brief.startswith("The seed hides")
        assert len(run.walks) == 2
        assert len(run.cards) == 1
        assert run.cards[0].name == "Structured Duration"
        assert run.summary == "The run found one strong concept."

    def test_walks_use_planner_knobs(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        agent.run("structure and event")
        seeds = [c[0] for c in fake_traverse]
        assert "structural causality" in seeds
        assert "Mongol mythology" in seeds

    def test_fragments_attached_to_walks(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        walk = run.walks[0]
        assert len(walk.fragments) == 2
        assert walk.fragments[0].chunk_id == "c-1"
        assert walk.fragments[1].forced_jump is True
        assert walk.jumps == 1
        assert walk.fragments[0].walk_seed == "structural causality"

    def test_walk_budget_respected(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=1, max_rounds=1)
        run = agent.run("structure and event")
        assert len(run.walks) == 1
        assert len(fake_traverse) == 1

    def test_empty_seed_raises(self):
        agent, _, _, _ = build_agent()
        with pytest.raises(IdeaAgentError, match="must not be empty"):
            agent.run("   ")

    def test_planner_no_walks_falls_back_to_seed(self, fake_traverse):
        replies = default_replies(plan_initial={"brief": "b", "walks": []})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("the fold")
        assert len(run.walks) == 1
        assert fake_traverse[0][0] == "the fold"

    def test_no_fragments_anywhere_raises(self, monkeypatch):
        monkeypatch.setattr(
            "rhizome.ideas.agent.TraversalEngine.traverse", lambda self, s: []
        )
        agent, _, _, _ = build_agent(max_walks=1, max_rounds=1)
        with pytest.raises(IdeaAgentError, match="no fragments"):
            agent.run("anything")

    def test_walk_failure_is_recorded_not_fatal(self, monkeypatch):
        monkeypatch.setattr(
            "rhizome.ideas.agent.TraversalEngine.traverse",
            lambda self, s: (_ for _ in ()).throw(RuntimeError("qdrant down")),
        )
        replies = default_replies()
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        with pytest.raises(IdeaAgentError, match="no fragments"):
            agent.run("structure and event")

    def test_progress_callback_receives_stages(self, fake_traverse):
        seen: list[tuple[str, str]] = []
        agent, _, _, _ = build_agent(
            max_walks=2, max_rounds=1, on_progress=lambda s, m: seen.append((s, m))
        )
        agent.run("structure and event")
        stages = {s for s, _ in seen}
        assert {"plan", "walk", "synthesize", "verify", "critique", "summarize"} <= stages

    def test_synthesis_failure_yields_empty_cards(self, fake_traverse):
        from rhizome.llm.base import LLMError

        replies = default_replies(synthesize=LLMError("synthesis blew up"))
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards == []
        assert len(run.walks) == 2


class TestGrounding:
    def test_fabricated_chunk_id_is_demoted(self):
        fragments = {"real-1": Fragment(
            walk_id=1, chunk_id="real-1", text="abc", article_title="T",
            article_url="U", similarity=0.5, depth=1, forced_jump=False,
        )}
        components = [
            ComponentEvidence(register="corpus", claim="x", chunk_id="FAKE-9", quote="abc"),
            ComponentEvidence(register="corpus", claim="y", chunk_id="real-1", quote="abc"),
        ]
        validated = _validate_components(components, fragments)
        assert validated[0].register == "model"
        assert validated[0].audit == "uncorroborated"
        assert "not in any walk result" in validated[0].audit_note
        assert validated[0].quote == ""
        assert validated[1].register == "corpus"

    def test_corpus_component_inherits_source_metadata(self):
        fragments = {"real-1": Fragment(
            walk_id=1, chunk_id="real-1", text="abc", article_title="Anti-Oedipus",
            article_url="https://x", similarity=0.67, depth=2, forced_jump=False,
        )}
        components = [ComponentEvidence(
            register="corpus", claim="y", chunk_id="real-1",
            article_title="WRONG", similarity=0.0, quote="abc",
        )]
        validated = _validate_components(components, fragments)
        assert validated[0].article_title == "Anti-Oedipus"
        assert validated[0].similarity == 0.67

    def test_paraphrased_quote_is_replaced_with_verbatim(self):
        actual = "Deterritorialization and reterritorialization occur simultaneously in every assemblage."
        fragments = {"r": Fragment(
            walk_id=1, chunk_id="r", text=actual, article_title="T",
            article_url="U", similarity=0.5, depth=1, forced_jump=False,
        )}
        components = [ComponentEvidence(
            register="corpus", claim="y", chunk_id="r",
            quote="Deterritorialization and reterritorialization happen together",
        )]
        validated = _validate_components(components, fragments)
        assert validated[0].quote in actual

    def test_model_component_loses_corpus_fields(self):
        components = [ComponentEvidence(
            register="model", claim="z", chunk_id="should-clear", quote="should-clear"
        )]
        validated = _validate_components(components, {})
        assert validated[0].chunk_id == ""
        assert validated[0].quote == ""

    def test_best_quote_finds_offset(self):
        actual = "prefix words then the target phrase appears here at the end"
        assert _best_quote("the target phrase", actual).startswith("the target phrase")

    def test_best_quote_falls_back_to_head(self):
        assert _best_quote("nothing matches", "some other text entirely") == "some other text entirely"


class TestNovelty:
    def test_tracing_verdict_at_high_similarity(self, fake_traverse):
        agent, _, store, _ = build_agent(
            store=FakeVectorStore(scores=[0.95]), max_walks=2, max_rounds=1
        )
        run = agent.run("structure and event")
        card = run.cards[0]
        assert card.novelty.verdict == "tracing"
        assert card.novelty.max_similarity == 0.95
        assert card.novelty.nearest_title.startswith("Nearest Article")

    def test_novel_verdict_at_low_similarity(self, fake_traverse):
        agent, _, _, _ = build_agent(
            store=FakeVectorStore(scores=[0.30]), max_walks=2, max_rounds=1
        )
        run = agent.run("structure and event")
        assert run.cards[0].novelty.verdict == "novel"

    def test_borderline_verdict(self, fake_traverse):
        agent, _, _, _ = build_agent(
            store=FakeVectorStore(scores=[0.65]), max_walks=2, max_rounds=1
        )
        run = agent.run("structure and event")
        assert run.cards[0].novelty.verdict == "borderline"

    def test_embedder_failure_skips_novelty_without_crash(self, fake_traverse):
        agent, _, _, _ = build_agent(embedder=FailingEmbedder(), max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].novelty.verdict == ""


class TestAudit:
    def test_model_claim_gets_verdict(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        model_components = run.cards[0].model_components()
        assert len(model_components) == 1
        assert model_components[0].audit == "corroborated"
        assert "Spinoza" in model_components[0].audit_note

    def test_missing_audit_verdict_marks_uncorroborated(self, fake_traverse):
        replies = default_replies(audit={"audits": []})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].model_components()[0].audit == "uncorroborated"

    def test_contested_claim_flags_card(self, fake_traverse):
        replies = default_replies(audit={"audits": [{"id": "c0", "verdict": "contested", "note": "Corpus contradicts this."}]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].contested() is True

    def test_unknown_verdict_normalises(self, fake_traverse):
        replies = default_replies(audit={"audits": [{"id": "c0", "verdict": "banana", "note": "n"}]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].model_components()[0].audit == "uncorroborated"

    def test_no_model_claims_skips_audit(self, fake_traverse):
        synth = {
            "ideas": [{
                "name": "Pure Corpus", "statement": "s",
                "components": [{"register": "corpus", "claim": "c", "chunk_id": "c-1", "quote": "Deterritorialization occurs."}],
            }]
        }
        replies = default_replies(synthesize=synth)
        agent, llm, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].model_components() == []
        assert any(c.register == "corpus" for c in run.cards[0].components)


class TestCritique:
    def test_discard_caps_confidence(self, fake_traverse):
        replies = default_replies(critique={"verdicts": [{"name": "Structured Duration", "verdict": "discard", "confidence": 0.9, "reasoning": "weak"}]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        card = run.cards[0]
        assert card.confidence <= 0.2
        assert card.verdict == "discard"
        assert "ruled discard" in card.confidence_note

    def test_tracing_of_caps_confidence(self, fake_traverse):
        replies = default_replies(critique={"verdicts": [{"name": "Structured Duration", "is_tracing_of": "process philosophy", "verdict": "keep", "confidence": 0.95, "reasoning": "r"}]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        card = run.cards[0]
        assert card.confidence <= 0.3
        assert card.tracing_of == "process philosophy"
        assert card.is_tracing() is True

    def test_keep_with_tracing_records_both_signals(self, fake_traverse):
        """A referee 'keep' overridden by a tracing cap must stay auditable.

        Regression: confidence was silently capped at 0.3 with the referee's
        own verdict and score discarded, so a praised card looked rejected with
        no recorded reason.
        """
        replies = default_replies(critique={"verdicts": [{
            "name": "Structured Duration",
            "is_tracing_of": "process philosophy",
            "verdict": "keep",
            "confidence": 0.95,
            "reasoning": "worth keeping",
        }]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        card = agent.run("structure and event").cards[0]
        assert card.verdict == "keep"
        assert card.is_tracing()
        assert card.confidence == 0.3
        assert "referee scored 0.95" in card.confidence_note
        assert "ruled it a tracing" in card.confidence_note
        assert card.tracing_of == "process philosophy"

    def test_uncapped_card_has_empty_note(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        card = agent.run("structure and event").cards[0]
        assert card.verdict == "keep"
        assert card.confidence == 0.8
        assert card.confidence_note == ""
        assert card.is_tracing() is False

    def test_unknown_ruling_normalises_to_unjudged(self, fake_traverse):
        replies = default_replies(critique={"verdicts": [{
            "name": "Structured Duration", "verdict": "banana", "confidence": 0.7,
        }]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        assert agent.run("structure and event").cards[0].verdict == "unjudged"

    def test_unmatched_card_is_unjudged(self, fake_traverse):
        replies = default_replies(critique={"verdicts": [
            {"name": "Some Other Concept", "verdict": "keep", "confidence": 1.0}
        ]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        assert agent.run("structure and event").cards[0].verdict == "unjudged"

    def test_tracing_excluded_from_survivor_selection(self):
        run = IdeaRun(seed="s")
        run.cards = [
            IdeaCard(name="Tracing", statement="x", confidence=0.9,
                     verdict="keep", tracing_of="process philosophy"),
            IdeaCard(name="Genuine", statement="y", confidence=0.6, verdict="keep"),
        ]
        assert _best_survivor(run).name == "Genuine"

    def test_discarded_card_excluded_from_survivor_selection(self):
        run = IdeaRun(seed="s")
        run.cards = [
            IdeaCard(name="Rejected", statement="x", confidence=0.2, verdict="discard"),
            IdeaCard(name="Kept", statement="y", confidence=0.6, verdict="keep"),
        ]
        assert _best_survivor(run).name == "Kept"

    def test_referee_reasoning_appended_to_objection(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert "Referee:" in run.cards[0].objection
        assert "Components require one another." in run.cards[0].objection

    def test_unmatched_card_gets_neutral_confidence(self, fake_traverse):
        replies = default_replies(critique={"verdicts": [{"name": "Some Other Concept", "verdict": "keep", "confidence": 1.0}]})
        agent, _, _, _ = build_agent(replies=replies, max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        assert run.cards[0].confidence == 0.5


class TestRounds:
    def test_second_round_reseeds_from_survivor(self, monkeypatch):
        calls = []

        def fake_traverse(self, starting_concept):
            calls.append(starting_concept)
            return [make_step("c-1", "Anti-Oedipus", "Deterritorialization occurs.")]

        monkeypatch.setattr("rhizome.ideas.agent.TraversalEngine.traverse", fake_traverse)
        replies = default_replies()
        agent, _, _, _ = build_agent(replies=replies, max_walks=4, max_rounds=2)
        run = agent.run("structure and event")
        assert run.rounds == 2
        assert len(run.cards) == 2
        assert "Structured Duration" in calls

    def test_no_survivor_stops_early(self, monkeypatch):
        calls = []

        def fake_traverse(self, starting_concept):
            calls.append(starting_concept)
            return [make_step("c-1", "T", "text")]

        monkeypatch.setattr("rhizome.ideas.agent.TraversalEngine.traverse", fake_traverse)
        discard = {"verdicts": [{"name": "Structured Duration", "verdict": "discard", "confidence": 0.1}]}
        replies = default_replies(critique=discard)
        agent, llm, _, _ = build_agent(replies=replies, max_walks=4, max_rounds=3)
        run = agent.run("structure and event")
        assert run.rounds == 2
        assert len(run.cards) == 1
        assert llm.stages.count("synthesize") == 1

    def test_best_survivor_prefers_uncontested(self):
        run = IdeaRun(seed="s")
        contested = IdeaCard(name="A", statement="x", confidence=0.9,
                             components=[ComponentEvidence(register="model", claim="c", audit="contested")])
        clean = IdeaCard(name="B", statement="y", confidence=0.7)
        run.cards = [contested, clean]
        assert _best_survivor(run).name == "B"

    def test_best_survivor_none_when_all_rejected(self):
        run = IdeaRun(seed="s")
        run.cards = [IdeaCard(name="A", statement="x", confidence=0.2)]
        assert _best_survivor(run) is None


class TestModelParsing:
    def test_walk_plan_clamps_out_of_range(self):
        plan = WalkPlan.from_dict({"seed": "x", "depth": 9999, "epsilon": 5.0, "temperature": -1.0, "top_k": 1})
        assert plan.depth == 24
        assert plan.epsilon == 1.0
        assert plan.temperature == 0.0
        assert plan.top_k == 3

    def test_walk_plan_requires_seed(self):
        with pytest.raises(ValueError, match="missing a seed"):
            WalkPlan.from_dict({"depth": 5})

    def test_walk_plan_defaults_on_garbage(self):
        plan = WalkPlan.from_dict({"seed": "ok", "depth": "banana", "epsilon": None})
        assert plan.depth == 8
        assert plan.epsilon == 0.1

    def test_idea_card_clamps_confidence(self):
        card = IdeaCard.from_dict({"name": "n", "statement": "s", "confidence": 99})
        assert card.confidence == 1.0
        card = IdeaCard.from_dict({"name": "n", "statement": "s", "confidence": -5})
        assert card.confidence == 0.0

    def test_idea_card_defaults_name(self):
        assert IdeaCard.from_dict({"statement": "s"}).name == "unnamed concept"

    def test_component_rejects_unknown_register(self):
        component = ComponentEvidence.from_dict({"register": "vibes", "claim": "c"})
        assert component.register == "model"

    def test_component_rejects_unknown_audit(self):
        component = ComponentEvidence.from_dict({"register": "model", "claim": "c", "audit": "maybe"})
        assert component.audit == ""

    def test_idea_card_skips_non_int_walk_ids(self):
        card = IdeaCard.from_dict({"name": "n", "statement": "s", "walk_ids": [1, "x", None, 3]})
        assert card.walk_ids == [1, 3]

    def test_novelty_check_handles_bad_similarity(self):
        assert NoveltyCheck.from_dict({"max_similarity": "nope"}).max_similarity == 0.0

    def test_run_to_dict_roundtrips(self, fake_traverse):
        agent, _, _, _ = build_agent(max_walks=2, max_rounds=1)
        run = agent.run("structure and event")
        data = run.to_dict()
        assert data["seed"] == "structure and event"
        assert len(data["walks"]) == 2
        assert len(data["cards"]) == 1
        assert data["cards"][0]["components"][0]["register"] == "corpus"
        import json as _json
        _json.dumps(data)

    def test_kept_cards_threshold(self):
        run = IdeaRun(seed="s")
        run.cards = [
            IdeaCard(name="hi", statement="x", confidence=0.8),
            IdeaCard(name="lo", statement="y", confidence=0.2),
        ]
        assert [c.name for c in run.kept_cards()] == ["hi"]
