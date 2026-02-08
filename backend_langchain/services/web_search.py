"""External web search helpers (DuckDuckGo)."""

from __future__ import annotations

from typing import List, Dict

try:  # Prefer the renamed package
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ddgs package is required for web search. Install with 'pip install ddgs'."
        ) from exc

from ..core.logger import setup_logger

logger = setup_logger(__name__)


def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return DuckDuckGo search snippets for the given query."""
    results: List[Dict[str, str]] = []
    if not query:
        return results

    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                query,
                region="in-en",
                safesearch="moderate",
                timelimit="y",
                max_results=max_results,
            )
            for item in raw_results or []:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    }
                )
    except Exception as exc:
        logger.warning("duckduckgo_search_failed | query=%s | error=%s", query[:80], exc)

    filtered = [r for r in results if r.get("title") and r.get("url")]
    return filtered[:max_results]
