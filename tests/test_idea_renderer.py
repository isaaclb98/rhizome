"""Tests for markdown card rendering."""

from rhizome.ideas.models import (
    ComponentEvidence,
    Fragment,
    IdeaCard,
    IdeaRun,
    NoveltyCheck,
    WalkPlan,
    WalkResult,
)
from rhizome.ideas.renderer import render_cards


def make_run() -> IdeaRun:
    run = IdeaRun(seed="structure and event", brief="A tension worth pulling on.")
    run.walks.append(
        WalkResult(
            walk_id=1,
            plan=WalkPlan(seed="structural causality", rationale="home turf", depth=4, epsilon=0.2),
            fragments=[
                Fragment(
                    walk_id=1,
                    chunk_id="Anti-Oedipus-003",
                    text="Deterritorialization occurs.",
                    article_title="Anti-Oedipus",
                    article_url="https://en.wikipedia.org/wiki/Anti-Oedipus",
                    similarity=0.671,
                    depth=1,
                    forced_jump=False,
                ),
                Fragment(
                    walk_id=1,
                    chunk_id="Bergson-007",
                    text="Duration is heterogeneous.",
                    article_title="Henri Bergson",
                    article_url="https://en.wikipedia.org/wiki/Henri_Bergson",
                    similarity=0.312,
                    depth=2,
                    forced_jump=True,
                ),
            ],
            jumps=1,
            note="A real collision between Althusser and Bergson.",
        )
    )
    run.cards.append(
        IdeaCard(
            name="Structured Duration",
            statement="Duration as an ideological apparatus.",
            composition="Bergson supplies time; Althusser supplies structure.",
            components=[
                ComponentEvidence(
                    register="corpus",
                    claim="Deterritorialization is simultaneous with reterritorialization.",
                    chunk_id="Anti-Oedipus-003",
                    article_title="Anti-Oedipus",
                    article_url="https://en.wikipedia.org/wiki/Anti-Oedipus",
                    similarity=0.671,
                    quote="Deterritorialization occurs.",
                ),
                ComponentEvidence(
                    register="model",
                    claim="Bergson influenced Deleuze's reading of Spinoza.",
                    audit="corroborated",
                    audit_note="Corpus confirms the Spinoza reading.",
                ),
            ],
            stakes="Makes it possible to ask when a structure becomes an event.",
            objection="It may restate process philosophy.\n\n**Referee:** Components require one another.",
            novelty=NoveltyCheck(
                max_similarity=0.42,
                nearest_title="Henri Bergson",
                nearest_url="https://en.wikipedia.org/wiki/Henri_Bergson",
                verdict="novel",
            ),
            walk_ids=[1],
            confidence=0.8,
            verdict="keep",
        )
    )
    run.summary = "One strong concept survived."
    run.rounds = 1
    return run


class TestRenderCards:
    def test_contains_all_sections(self):
        out = render_cards(make_run())
        for heading in ["# Ideas:", "## Brief", "## Concepts", "## Assessment", "## Traversal trail"]:
            assert heading in out

    def test_card_name_and_statement_present(self):
        out = render_cards(make_run())
        assert "Structured Duration" in out
        assert "Duration as an ideological apparatus." in out

    def test_corpus_component_cited_with_link_and_chunk_id(self):
        out = render_cards(make_run())
        assert "[Anti-Oedipus](https://en.wikipedia.org/wiki/Anti-Oedipus)" in out
        assert "`Anti-Oedipus-003`" in out
        assert "sim 0.671" in out

    def test_registers_labelled_distinctly(self):
        out = render_cards(make_run())
        assert "*(corpus)*" in out
        assert "*(model [corroborated])*" in out

    def test_quote_rendered_as_blockquote(self):
        out = render_cards(make_run())
        assert "> Deterritorialization occurs." in out

    def test_audit_note_shown(self):
        out = render_cards(make_run())
        assert "audit: Corpus confirms the Spinoza reading." in out

    def test_novelty_verdict_in_flags(self):
        out = render_cards(make_run())
        assert "novelty: novel (0.42)" in out
        assert "confidence: 0.80" in out

    def test_contested_flag_appears(self):
        run = make_run()
        run.cards[0].components[1].audit = "contested"
        out = render_cards(run)
        assert "contested" in out

    def test_tracing_flag_appears(self):
        run = make_run()
        run.cards[0].novelty.verdict = "tracing"
        assert "novelty: tracing" in render_cards(run)

    def test_forced_jump_marker_in_trail(self):
        out = render_cards(make_run())
        assert "JUMP" in out

    def test_walk_knobs_in_trail(self):
        out = render_cards(make_run())
        assert "epsilon=0.2" in out
        assert "seed: *structural causality*" in out
        assert "Intent: home turf" in out

    def test_walk_note_in_trail(self):
        assert "A real collision between Althusser and Bergson." in render_cards(make_run())

    def test_objection_includes_referee(self):
        out = render_cards(make_run())
        assert "**Objection.**" in out
        assert "**Referee:**" in out

    def test_stakes_rendered(self):
        assert "**Stakes.** Makes it possible to ask when a structure becomes an event." in render_cards(make_run())

    def test_cards_sorted_by_confidence_descending(self):
        run = make_run()
        run.cards.append(IdeaCard(name="Weaker", statement="w", confidence=0.3))
        run.cards.append(IdeaCard(name="Stronger", statement="s", confidence=0.9))
        out = render_cards(run)
        assert out.index("Stronger") < out.index("Structured Duration") < out.index("Weaker")

    def test_empty_run_renders_without_crash(self):
        out = render_cards(IdeaRun(seed="nothing"))
        assert "# Ideas: nothing" in out
        assert "## Concepts" not in out

    def test_walk_with_no_fragments_noted(self):
        run = IdeaRun(seed="s")
        run.walks.append(WalkResult(walk_id=1, plan=WalkPlan(seed="x")))
        assert "*no fragments returned*" in render_cards(run)

    def test_header_counts(self):
        out = render_cards(make_run())
        assert "1 walk(s), 1 concept(s), 1 kept" in out

    def test_verdict_shown_in_flags(self):
        assert "verdict: keep" in render_cards(make_run())

    def test_tracing_target_shown(self):
        run = make_run()
        run.cards[0].tracing_of = "process philosophy"
        out = render_cards(run)
        assert "tracing of: process philosophy" in out

    def test_confidence_note_shown(self):
        run = make_run()
        run.cards[0].confidence_note = "referee scored 0.95; capped at 0.3: tracing"
        assert "*referee scored 0.95; capped at 0.3: tracing*" in render_cards(run)

    def test_unjudged_default_verdict(self):
        run = make_run()
        run.cards[0].verdict = "unjudged"
        assert "verdict: unjudged" in render_cards(run)

    def test_nearest_corpus_text_section(self):
        out = render_cards(make_run())
        assert "**Nearest corpus text.**" in out
        assert "[Henri Bergson](https://en.wikipedia.org/wiki/Henri_Bergson) at 0.420" in out
