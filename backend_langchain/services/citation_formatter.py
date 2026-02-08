"""Helpers for normalizing and formatting citation metadata."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def format_resources_section(citations: Sequence[Dict[str, Any]]) -> str:
    """Return a markdown Resources block from structured citations."""
    if not citations:
        return ""

    doc_entries = [c for c in citations if c.get("type") == "document"]
    web_entries = [c for c in citations if c.get("type") == "web"]

    if not doc_entries and not web_entries:
        return ""

    lines: List[str] = ["**Resources**"]

    for entry in doc_entries:
        title = entry.get("title") or entry.get("display_name") or entry.get("doc_id") or "Document"
        download_url = entry.get("download_url") or entry.get("source_url")
        page_text = _format_page_text(entry.get("pages"), entry.get("has_unknown_page"))
        suffix = f" ({page_text})" if page_text else ""
        if download_url:
            lines.append(f"- {title} ([Download PDF]({download_url})) {suffix}".rstrip())
        else:
            lines.append(f"- {title}{suffix}")

    for entry in web_entries:
        title = entry.get("title") or entry.get("display_name") or entry.get("url") or "Web source"
        url = entry.get("url") or entry.get("source_url")
        favicon = entry.get("favicon_url")
        prefix = f"![{entry.get('display_name', 'site')}]({favicon}) " if favicon else ""
        if url:
            lines.append(f"- {prefix}[{title}]({url})")
        else:
            lines.append(f"- {prefix}{title}")

    return "\n".join(lines)


def append_resources_to_answer(answer: str, resources_text: str) -> str:
    """Ensure the Resources block appears exactly once at the end of the answer."""
    if not resources_text:
        return answer

    answer = answer or ""
    normalized_resources = resources_text.strip()
    if not normalized_resources:
        return answer

    if normalized_resources in answer:
        return answer

    if "Resources" in answer and "Resources" in normalized_resources:
        # Heuristic: avoid double-heading
        return answer

    separator = "\n\n" if answer.strip() else ""
    return f"{answer.rstrip()}{separator}{resources_text}".strip()


def _format_page_text(pages: Iterable[int] | None, has_unknown: Any) -> str:
    numbers = sorted(set(int(p) for p in pages or []))
    parts: List[str] = []
    if numbers:
        ranges = _condense_page_ranges(numbers)
        parts.append(f"Pages {', '.join(ranges)}")
    if has_unknown:
        parts.append("Page unknown")
    return "; ".join(parts)


def _condense_page_ranges(numbers: List[int]) -> List[str]:
    if not numbers:
        return []
    ranges: List[str] = []
    start = prev = numbers[0]
    for num in numbers[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append(_format_range(start, prev))
        start = prev = num
    ranges.append(_format_range(start, prev))
    return ranges


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}" if start != end else str(start)
