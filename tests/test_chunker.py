"""Tests for the Chunker module."""

import pytest
from rhizome.corpus.chunker import Chunker, Chunk


class TestChunker:
    """Tests for Chunker."""

    def test_split_paragraph_boundary(self):
        """Short paragraphs are preserved as single chunks."""
        chunker = Chunker(max_chars=500)
        chunks = chunker.chunk_article(
            article_title="Test Article",
            article_url="https://en.wikipedia.org/wiki/Test_Article",
            article_text="This is a short paragraph.\n\nThis is another paragraph.",
        )

        assert len(chunks) == 2
        assert chunks[0].article_title == "Test Article"
        assert "short paragraph" in chunks[0].text

    def test_long_paragraph_split_at_sentence_boundary(self):
        """Paragraphs > max_chars are split at sentence boundaries."""
        chunker = Chunker(max_chars=50)
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = chunker.chunk_article(
            article_title="Test",
            article_url="https://en.wikipedia.org/wiki/Test",
            article_text=text,
        )

        # Each chunk should be a subset of sentences
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 100  # rough upper bound after rejoining

    def test_duplicate_chunk_deduplication(self):
        """Same text across articles would produce same IDs but are tracked."""
        # Note: deduplication happens at ingest time, not in chunker itself
        # This tests that chunk IDs are stable and unique per article+position
        chunker = Chunker()
        chunks_a = chunker.chunk_article("Article A", "https://example.com/a", "Same text here.")
        chunks_b = chunker.chunk_article("Article B", "https://example.com/b", "Same text here.")

        # IDs should differ because article slug differs
        assert chunks_a[0].id != chunks_b[0].id
        assert chunks_a[0].id.startswith("article-a")
        assert chunks_b[0].id.startswith("article-b")

    def test_slugify(self):
        """Article titles are slugified into chunk IDs."""
        chunker = Chunker()
        chunks = chunker.chunk_article(
            article_title="Modernism and Postmodernism",
            article_url="https://en.wikipedia.org/wiki/Modernism_and_Postmodernism",
            article_text="A short paragraph.",
        )
        assert chunks[0].id.startswith("modernism-and-postmodernism")

    def test_empty_paragraphs_ignored(self):
        """Empty paragraphs do not produce chunks."""
        chunker = Chunker()
        chunks = chunker.chunk_article(
            article_title="Test",
            article_url="https://example.com/test",
            article_text="Paragraph one.\n\n\n\nParagraph two.",
        )
        assert len(chunks) == 2
        assert all(c.text.strip() for c in chunks)

    def test_bibliography_truncation(self):
        """Bibliography and reference sections are stripped before chunking."""
        chunker = Chunker()
        text = (
            "Modernism is characterized by a break with traditional ways of writing. "
            "It experiment with form and technique.\n\n"
            "See also\nPostmodernism\nReferences\n"
            "This reference section should be stripped."
        )
        chunks = chunker.chunk_article(
            article_title="Modernism",
            article_url="https://en.wikipedia.org/wiki/Modernism",
            article_text=text,
        )
        # The text should be truncated before "See also"
        full_text = " ".join(c.text for c in chunks)
        assert "See also" not in full_text
        assert "References" not in full_text
        assert "Modernism is characterized" in full_text
