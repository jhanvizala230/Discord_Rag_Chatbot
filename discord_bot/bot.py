"""Discord bot client for DocIntel backend."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import discord
import httpx
from discord.ext import commands

from config import Settings

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOGS_DIR = Path(__file__).resolve().parent / "output" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "discord_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("docintel.discord")

settings = Settings.load()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}


async def query_backend(
    message_content: str,
    session_id: str,
    client: httpx.AsyncClient,
    *,
    agent_hint: Optional[str] = None,
    image_attachment: Optional[Tuple[str, bytes, str]] = None,
) -> Optional[dict]:
    url = f"{settings.api_base_url}/routed_query"
    try:
        logger.info(
            "discord_request | session_id=%s | has_image=%s | agent_hint=%s",
            session_id,
            bool(image_attachment),
            agent_hint or "",
        )
        if image_attachment:
            filename, content, content_type = image_attachment
            logger.info(
                "discord_image_payload | filename=%s | size_bytes=%d | content_type=%s",
                filename,
                len(content or b""),
                content_type or "",
            )
            files = {
                "image_file": (filename, content, content_type or "application/octet-stream"),
            }
            data = {
                "query": message_content,
                "session_id": session_id,
            }
            if agent_hint:
                data["agent_hint"] = agent_hint
            response = await client.post(url, data=data, files=files, timeout=settings.request_timeout)
        else:
            payload = {"query": message_content, "session_id": session_id}
            if agent_hint:
                payload["agent_hint"] = agent_hint
            response = await client.post(url, json=payload, timeout=settings.request_timeout)
        response.raise_for_status()
        return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error("API request failed: %s", exc)
        return None


async def upload_attachment(
    attachment: discord.Attachment,
    author_name: str,
    client: httpx.AsyncClient,
) -> Tuple[str, bool, str]:
    filename = attachment.filename or "document"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in DOCUMENT_EXTENSIONS:
        return filename, False, "Unsupported file type"

    try:
        file_response = await client.get(attachment.url, timeout=settings.request_timeout)
        file_response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error("Failed to download attachment %s: %s", filename, exc)
        return filename, False, "Could not download attachment"

    files = {
        "file": (
            filename,
            file_response.content,
            attachment.content_type or "application/octet-stream",
        )
    }
    data = {
        "title": filename,
        "author": author_name,
    }

    try:
        upload_response = await client.post(
            f"{settings.api_base_url}/upload",
            data=data,
            files=files,
            timeout=settings.request_timeout,
        )
        upload_response.raise_for_status()
        return filename, True, f"document {filename} is saved successfully you can ask me anything from it"
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error("DocIntel upload failed for %s: %s", filename, exc)
        return filename, False, "DocIntel upload failed"


async def handle_attachments(
    message: discord.Message, client: httpx.AsyncClient
) -> Tuple[Optional[Tuple[str, bytes, str]], List[str], bool]:
    image_payload: Optional[Tuple[str, bytes, str]] = None
    status_lines: List[str] = []

    if not message.attachments:
        return image_payload, status_lines, False

    has_document = any(
        (att.filename or "").rsplit(".", 1)[-1].lower() in DOCUMENT_EXTENSIONS
        for att in message.attachments
    )

    for attachment in message.attachments:
        filename = attachment.filename or "attachment"
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        is_image = _is_image_attachment(extension, attachment.content_type)
        logger.info(
            "discord_attachment | filename=%s | ext=%s | content_type=%s | is_image=%s",
            filename,
            extension,
            attachment.content_type or "",
            is_image,
        )
        if extension in DOCUMENT_EXTENSIONS:
            name, success, detail = await upload_attachment(attachment, message.author.display_name, client)
            prefix = "✅" if success else "⚠️"
            status_lines.append(f"{prefix} {name}: {detail}")
        elif has_document and is_image:
            status_lines.append(
                "⚠️ Images ignored. Please upload documents and images separately."
            )
        elif has_document:
            status_lines.append(f"⚠️ {filename}: Unsupported attachment type")
        elif is_image and image_payload is None:
            try:
                file_response = await client.get(attachment.url, timeout=settings.request_timeout)
                file_response.raise_for_status()
                image_payload = (
                    filename,
                    file_response.content,
                    attachment.content_type or "image/png",
                )
                status_lines.append(f"🖼️ {filename}: Ready for visual analysis")
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                logger.error("Failed to download image %s: %s", filename, exc)
                status_lines.append(f"⚠️ {filename}: Could not download image")
        else:
            status_lines.append(f"⚠️ {filename}: Unsupported attachment type")

    return image_payload, status_lines, has_document


@bot.event
async def on_ready():
    logger.info("Bot connected as %s", bot.user)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or (not message.content and not message.attachments):
        return

    async with httpx.AsyncClient() as client:
        image_payload, status_lines, has_document = await handle_attachments(message, client)

        if status_lines:
            await message.channel.send("\n".join(status_lines))

        if has_document:
            return

        content_text = (message.content or "").strip()
        if not content_text and not image_payload:
            return

        cleaned_query, agent_hint = _extract_agent_hint(content_text)
        if image_payload:
            agent_hint = "image_qa"

        data = await query_backend(
            cleaned_query,
            str(message.channel.id),
            client,
            agent_hint=agent_hint,
            image_attachment=image_payload,
        )

    if not data:
        await message.channel.send(
            "Sorry, I'm having trouble reaching DocIntel right now. Please try again later."
        )
        return

    answer_text = data.get("answer") or "I couldn't generate an answer this time."
    metadata = data.get("metadata") or {}
    response_text = _format_discord_response(answer_text, metadata)
    if image_payload and not metadata.get("used_image"):
        logger.warning("Backend ignored uploaded image for session %s", message.channel.id)

    for chunk in _split_discord_message(response_text):
        await message.channel.send(chunk)


def _format_discord_response(answer: str, metadata: dict) -> str:
    metadata = metadata or {}
    citations = metadata.get("citations") or []
    if citations:
        resources_block = _render_resources_block(citations)
        if resources_block:
            return _append_block(answer, resources_block)
    return answer


def _split_discord_message(text: str, *, limit: int = 1990) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _append_block(answer: str, block: str) -> str:
    answer = (answer or "").strip()
    block = (block or "").strip()
    if not block:
        return answer
    if block in answer:
        return answer
    separator = "\n\n" if answer else ""
    return f"{answer}{separator}{block}".strip()


def _render_resources_block(citations: List[dict]) -> str:
    if not citations:
        return ""

    doc_entries = [c for c in citations if c.get("type") == "document"]
    web_entries = [c for c in citations if c.get("type") == "web"]
    other_entries = [c for c in citations if c.get("type") not in {"document", "web"}]

    ordered: List[dict] = doc_entries + web_entries + other_entries
    if not ordered:
        ordered = citations

    lines: List[str] = ["**Resources**"]
    counter = 1
    for entry in ordered:
        line = _format_resource_line(entry, counter)
        if line:
            lines.append(line)
            counter += 1

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_resource_line(entry: dict, idx: int) -> str:
    entry = entry or {}
    entry_type = entry.get("type")
    if entry_type == "document" or (entry_type is None and entry.get("doc_id")):
        return _format_document_line(entry, idx)
    return _format_web_line(entry, idx)


def _format_document_line(entry: dict, idx: int) -> str:
    title = entry.get("title") or entry.get("display_name") or entry.get("doc_id") or f"Document {idx}"
    link = _resolve_document_link(entry)
    label = f"[{title}]({link})" if link else title
    page_text = _summarize_pages(entry)
    suffix = f" – {page_text}" if page_text else ""
    return f"{idx}. {label}{suffix}"


def _resolve_document_link(entry: dict) -> str:
    link = entry.get("download_url") or entry.get("source_url")
    if link:
        return link
    doc_id = entry.get("doc_id")
    if doc_id:
        return f"{settings.api_base_url}/documents/{doc_id}/file"
    return ""


def _summarize_pages(entry: dict) -> str:
    pages = entry.get("pages") or []
    has_unknown = entry.get("has_unknown_page")

    normalized = []
    for value in pages:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    normalized = sorted(set(normalized))

    segments: List[str] = []
    if normalized:
        segments.append(_condense_ranges(normalized))
    if has_unknown:
        segments.append("Page unknown")

    if segments:
        return "; ".join(segments)

    label = entry.get("page_label")
    if label:
        return label
    page_number = entry.get("page_number")
    if page_number:
        return f"Page {page_number}"
    return ""


def _condense_ranges(numbers: List[int]) -> str:
    if not numbers:
        return ""
    ranges: List[str] = []
    start = prev = numbers[0]
    for num in numbers[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append(_format_range(start, prev))
        start = prev = num
    ranges.append(_format_range(start, prev))
    return f"Pages {', '.join(ranges)}"


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}" if start != end else str(start)


def _format_web_line(entry: dict, idx: int) -> str:
    title = entry.get("title") or entry.get("display_name") or entry.get("url") or f"Link {idx}"
    url = entry.get("url") or entry.get("source_url")
    label = f"[{title}]({url})" if url else title
    snippet = entry.get("snippet")
    snippet_text = f" – {snippet.strip()}" if snippet else ""
    return f"{idx}. {label}{snippet_text}"


def _is_image_attachment(extension: str, content_type: Optional[str]) -> bool:
    if extension in IMAGE_EXTENSIONS:
        return True
    if content_type and content_type.lower().startswith("image/"):
        return True
    return False


def _extract_agent_hint(message_content: str) -> Tuple[str, Optional[str]]:
    text = (message_content or "").strip()
    lowered = text.lower()
    if lowered.startswith("/document"):
        return text[len("/document"):].strip(), "doc"
    if lowered.startswith("/internet"):
        return text[len("/internet"):].strip(), "finance_info"
    if lowered.startswith("/general"):
        return text[len("/general"):].strip(), "smalltalk"

    doc_terms = ("use document", "from document", "from documents", "uploaded doc", "uploaded document")
    web_terms = ("use internet", "search web", "search the web", "use web", "from internet", "from web")

    doc_match = any(term in lowered for term in doc_terms)
    web_match = any(term in lowered for term in web_terms)

    if doc_match and not web_match:
        return text, "doc"
    if web_match and not doc_match:
        return text, "finance_info"
    return text, None


def main() -> None:
    bot.run(settings.discord_token)


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.sleep(0))
    except RuntimeError:
        pass
    main()
