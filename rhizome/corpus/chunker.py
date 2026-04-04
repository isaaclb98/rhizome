"""Splits Wikipedia articles into chunks for embedding and retrieval."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text from a Wikipedia article."""

    id: str          # Slug-based ID: {article-slug}-{ordinal}
    text: str        # chunk text content
    article_title: str
    article_url: str


_HEADER_MARKUP_RE = re.compile(r'^=+\s*(.+?)\s*=+$')


_STOP_HEADERS = [
    "See also",
    "References",
    "Further reading",
    "External links",
    "Notes",
    "Bibliography",
    "Explanatory footnotes",
    "General and cited references",
]


def _truncate_before_bibliography(text: str) -> str:
    """Truncate text at the first bibliography section header.

    Wikipedia articles contain References, See also, External links and other
    sections after the main prose. These sections add noise to embeddings.
    This function finds the earliest bibliography header and cuts there.

    Handles both plain text headers (e.g. "\nExternal links\n") and
    Wikipedia markup headers (e.g. "\n== External links ==\n").
    """
    earliest = len(text)
    for header in _STOP_HEADERS:
        # Plain text with trailing newline (paragraph separator after header)
        plain = f"\n{header}\n"
        # Wikipedia markup: == Header == (with optional trailing spaces inside markup)
        markup = f"\n== {header} =="
        for variant in [plain, markup]:
            pos = text.find(variant)
            if pos != -1 and pos < earliest:
                earliest = pos
    return text[:earliest]


class Chunker:
    """Splits articles into chunks at paragraph boundaries.

    Paragraphs longer than max_chars are split at sentence boundaries.
    Bibliography and reference sections are stripped before chunking.
    """

    def __init__(self, max_chars: int = 500, min_chars: int = 50):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk_article(
        self,
        article_title: str,
        article_url: str,
        article_text: str,
    ) -> list[Chunk]:
        """Split a single article into chunks.

        Args:
            article_title: Wikipedia article title.
            article_url: Full Wikipedia article URL.
            article_text: Raw article text (no markup).

        Returns:
            List of Chunks with stable slug-based IDs.
        """
        slug = self._slugify(article_title)
        article_text = _truncate_before_bibliography(article_text)
        paragraphs = self._split_paragraphs(article_text)

        chunks = []
        ordinal = 0
        seen_text_hashes: set[str] = set()

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Skip chunks shorter than min_chars — these are usually headers,
            # single-line notes, or other noise that shouldn't be embedded.
            if len(para) < self.min_chars:
                continue

            if len(para) <= self.max_chars:
                text_hash = hashlib.md5(para.encode()).hexdigest()[:8]
                if text_hash in seen_text_hashes:
                    continue  # Deduplicate identical text within same article
                seen_text_hashes.add(text_hash)
                ordinal += 1
                chunks.append(Chunk(
                    id=f"{slug}-{ordinal:03d}",
                    text=para,
                    article_title=article_title,
                    article_url=article_url,
                ))
            else:
                sub_chunks = self._split_at_sentences(para, self.max_chars)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if sub and len(sub) >= self.min_chars:
                        text_hash = hashlib.md5(sub.encode()).hexdigest()[:8]
                        if text_hash in seen_text_hashes:
                            continue  # Deduplicate identical text within same article
                        seen_text_hashes.add(text_hash)
                        ordinal += 1
                        chunks.append(Chunk(
                            id=f"{slug}-{ordinal:03d}",
                            text=sub,
                            article_title=article_title,
                            article_url=article_url,
                        ))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text at paragraph boundaries, skipping bare Wikipedia headers."""
        # Split on double newlines first (paragraph breaks)
        parts = re.split(r'\n\s*\n', text)
        result = []
        for part in parts:
            part_stripped = part.strip()
            # Skip paragraphs that are purely Wikipedia section headers
            # (explaintext sometimes leaves bare markup like "=== Section Name ===").
            # A bare header has no period after stripping markup — it's a title/caption,
            # not prose, so it should not be embedded as a standalone chunk.
            if _HEADER_MARKUP_RE.match(part_stripped):
                header_text = _HEADER_MARKUP_RE.sub(r'\1', part_stripped)
                if not header_text.strip() or '.' not in header_text:
                    continue  # bare header, skip

            # Further split on single newlines if paragraphs are too long
            lines = part.split('\n')
            current = []
            current_len = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if current_len + len(line) + 1 <= self.max_chars:
                    current.append(line)
                    current_len += len(line) + 1
                else:
                    if current:
                        result.append(' '.join(current))
                    current = [line]
                    current_len = len(line)
            if current:
                result.append(' '.join(current))
        return result

    def _split_at_sentences(self, text: str, max_chars: int) -> list[str]:
        """Split long text at sentence boundaries, targeting max_chars per chunk."""
        # Simple sentence-end pattern: period, question, exclamation followed by space or end
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_len + len(sentence) + 1 <= max_chars:
                current.append(sentence)
                current_len += len(sentence) + 1
            else:
                if current:
                    chunks.append(' '.join(current))
                # If single sentence exceeds max_chars, hard-split it
                if len(sentence) > max_chars:
                    # Split at word boundary closest to max_chars
                    words = sentence.split()
                    current = []
                    current_len = 0
                    for word in words:
                        if current_len + len(word) + 1 <= max_chars:
                            current.append(word)
                            current_len += len(word) + 1
                        else:
                            if current:
                                chunks.append(' '.join(current))
                            current = [word]
                            current_len = len(word)
                    if current:
                        chunks.append(' '.join(current))
                else:
                    current = [sentence]
                    current_len = len(sentence)

        if current:
            chunks.append(' '.join(current))

        return chunks

    def _slugify(self, text: str) -> str:
        """Convert article title to a URL-safe slug.

        Preserves case to avoid collisions between articles whose titles
        differ only by case (e.g. 'Modernism' vs 'modernism' on Wikipedia).
        """
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
