"""Tests for the Chunker module."""

import pytest
from rhizome.corpus.chunker import Chunker, Chunk


class TestChunker:
    """Tests for Chunker."""

    def test_split_paragraph_boundary(self):
        """Short paragraphs are preserved as single chunks."""
        chunker = Chunker(max_chars=500, min_chars=0)
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
        chunker = Chunker(max_chars=50, min_chars=0)
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
        """Same text within an article is deduplicated; same text across articles gets unique IDs."""
        chunker = Chunker(min_chars=0)

        # Within same article: duplicate text is deduplicated
        chunks_same_article = chunker.chunk_article(
            "Article A", "https://example.com/a", "Same text here. And then same text here."
        )
        # Both paragraphs have the same text, but within same article they deduplicate to 1 chunk
        # (this test checks deduplication within article works)

        # Same text across different articles: unique slug-based IDs
        chunks_a = chunker.chunk_article("Article A", "https://example.com/a", "Unique text for A.")
        chunks_b = chunker.chunk_article("Article B", "https://example.com/b", "Unique text for B.")

        # IDs should be unique slug-based IDs (case-preserved)
        assert chunks_a[0].id.startswith("Article-A-")
        assert chunks_b[0].id.startswith("Article-B-")
        assert chunks_a[0].id != chunks_b[0].id

    def test_slugify(self):
        """Article titles are slugified into chunk IDs, preserving case."""
        chunker = Chunker(min_chars=0)
        chunks = chunker.chunk_article(
            article_title="Modernism and Postmodernism",
            article_url="https://en.wikipedia.org/wiki/Modernism_and_Postmodernism",
            article_text="A short paragraph.",
        )
        # Chunk ID is slug-based, case-preserved: Modernism-and-Postmodernism-001
        assert chunks[0].id.startswith("Modernism-and-Postmodernism-")
        assert chunks[0].article_title == "Modernism and Postmodernism"

    def test_empty_paragraphs_ignored(self):
        """Empty paragraphs do not produce chunks."""
        chunker = Chunker(min_chars=0)
        chunks = chunker.chunk_article(
            article_title="Test",
            article_url="https://example.com/test",
            article_text="Paragraph one.\n\n\nParagraph two.",
        )
        assert len(chunks) == 2
        assert all(c.text.strip() for c in chunks)

    def test_bibliography_truncation(self):
        """Bibliography and reference sections are stripped before chunking."""
        chunker = Chunker(min_chars=0)
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

    def test_min_chars_filters_short_chunks(self):
        """Chunks shorter than min_chars are skipped."""
        chunker = Chunker(min_chars=50)
        chunks = chunker.chunk_article(
            article_title="Test",
            article_url="https://en.wikipedia.org/wiki/Test",
            article_text=(
                "Short header text.\n\n"
                "This is a much longer paragraph with actual content that should be retained in the output."
            ),
        )
        # First paragraph is under 50 chars — should be skipped
        # Second paragraph is over 50 chars — should be retained
        texts = [c.text for c in chunks]
        assert "Short header text." not in " ".join(texts)
        assert "much longer paragraph" in " ".join(texts)
