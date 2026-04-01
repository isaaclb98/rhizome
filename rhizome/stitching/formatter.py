"""Converts a traversal path into a markdown document with citations."""

from rhizome.stitching.citation import format_citation
from rhizome.traversal.engine import TraversalStep


def stitch_to_markdown(
    starting_concept: str,
    path: list[TraversalStep],
) -> str:
    """Convert a traversal path into a readable markdown document.

    The document consists of:
    1. A title (the starting concept)
    2. Each chunk in traversal order, followed by its citation
    3. A footer note

    Args:
        starting_concept: The concept that started the traversal.
        path: Ordered list of TraversalSteps.

    Returns:
        Markdown-formatted string.
    """
    lines = [f"# {starting_concept}\n"]

    if not path:
        lines.append("*No traversal path generated. Try a different starting concept.*\n")
        return "".join(lines)

    for step in path:
        citation = format_citation(step.article_title, step.article_url)
        lines.append(f"{step.text}\n\n*{citation}*\n\n")

    # Footer
    lines.append("---\n")
    lines.append(f"*Traversal path: {len(path)} chunks | ")
    lines.append("Rhizome — Wikipedia Rhizomatic Traversal Tool*\n")

    return "".join(lines)
