"""Citation formatting utilities."""


def format_citation(article_title: str, article_url: str) -> str:
    """Format an inline citation for a chunk.

    Args:
        article_title: Wikipedia article title.
        article_url: Full Wikipedia article URL.

    Returns:
        Formatted citation string, e.g. "[Source: Modernism]"
    """
    return f"[Source: {article_title}]({article_url})"
