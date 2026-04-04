"""Tests for the stitching module."""

import pytest
from rhizome.stitching.formatter import stitch_to_markdown
from rhizome.stitching.citation import format_citation
from rhizome.traversal.engine import TraversalStep


class TestCitation:
    """Tests for citation formatting."""

    def test_format_citation(self):
        """Citations are formatted as [Source: Title]."""
        citation = format_citation("Modernism", "https://en.wikipedia.org/wiki/Modernism")
        assert citation == "[Source: Modernism](https://en.wikipedia.org/wiki/Modernism)"

    def test_format_citation_special_characters(self):
        """Citations handle special characters in titles."""
        citation = format_citation("Post-modernism (philosophy)", "https://en.wikipedia.org/wiki/Post-modernism")
        assert "Post-modernism" in citation


class TestFormatter:
    """Tests for markdown formatter."""

    def test_stitch_empty_path(self):
        """Empty path produces a note, not broken markdown."""
        result = stitch_to_markdown("Test Concept", [])
        assert "No traversal path" in result
        assert "# Test Concept" in result

    def test_stitch_single_chunk(self):
        """Single chunk is formatted correctly."""
        path = [
            TraversalStep(
                chunk_id="modernism-001",
                text="Modernism is a philosophical movement.",
                article_title="Modernism",
                article_url="https://en.wikipedia.org/wiki/Modernism",
                depth=0,
                similarity=0.95,
                forced_jump=False,
                candidates=[],
            )
        ]
        result = stitch_to_markdown("Modernism", path)

        assert "# Modernism" in result
        assert "Modernism is a philosophical movement." in result
        assert "[Source: Modernism]" in result
        assert "https://en.wikipedia.org/wiki/Modernism" in result

    def test_stitch_multiple_chunks(self):
        """Multiple chunks are concatenated in order."""
        path = [
            TraversalStep(
                chunk_id="modernism-001",
                text="First paragraph.",
                article_title="Modernism",
                article_url="https://en.wikipedia.org/wiki/Modernism",
                depth=0,
                similarity=0.95,
                forced_jump=False,
                candidates=[],
            ),
            TraversalStep(
                chunk_id="postmodernism-001",
                text="Second paragraph.",
                article_title="Postmodernism",
                article_url="https://en.wikipedia.org/wiki/Postmodernism",
                depth=1,
                similarity=0.88,
                forced_jump=False,
                candidates=[],
            ),
        ]
        result = stitch_to_markdown("Concept", path)

        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert result.index("First paragraph.") < result.index("Second paragraph.")
        assert "Modernism" in result
        assert "Postmodernism" in result

    def test_stitch_footer(self):
        """Footer includes chunk count and tool name."""
        path = [
            TraversalStep(
                chunk_id="c1",
                text="Text.",
                article_title="T",
                article_url="https://example.com/t",
                depth=0,
                similarity=0.95,
                forced_jump=False,
                candidates=[],
            )
        ]
        result = stitch_to_markdown("Concept", path)

        assert "1 chunks" in result or "1 chunks" in result
        assert "Rhizome" in result
