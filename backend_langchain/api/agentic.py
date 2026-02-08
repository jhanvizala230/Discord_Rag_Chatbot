
"""Agentic API router: single entry point for routed questions."""

import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError

from backend_langchain.agents import get_agent_orchestrator
from backend_langchain.core.config import (
    IMAGES_DIR,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_UPLOAD_BYTES,
    MAX_IMAGE_UPLOAD_MB,
)
from backend_langchain.core.logger import setup_logger
from backend_langchain.services.citation_formatter import (
    append_resources_to_answer,
    format_resources_section,
)

logger = setup_logger(__name__)
router = APIRouter()


class RoutedQueryRequest(BaseModel):
    query: str = Field(..., description="User question to route.")
    session_id: str = Field(..., description="Session id for conversation memory (required).")
    agent_hint: str | None = Field(None, description="Optional hint to force a specific agent.")


class RoutedQueryResponse(BaseModel):
    answer: str = Field(..., description="Final routed answer.")
    metadata: dict = Field(..., description="Metadata including query time, agent type, and citations.")

_orchestrator = get_agent_orchestrator()

JSON_REQUEST_SCHEMA = {
    "requestBody": {
        "content": {
            "application/json": {
                "schema": RoutedQueryRequest.model_json_schema(),
                "example": {
                    "query": "Summarize the uploaded LIC policy",
                    "session_id": "user-123",
                    "agent_hint": None,
                },
            }
        }
    }
}


@router.post("/routed_query", response_model=RoutedQueryResponse, openapi_extra=JSON_REQUEST_SCHEMA)
async def routed_query(request: Request):
    """Classify the incoming query and delegate to the appropriate specialist agent."""
    start_time = time.time()
    content_type = (request.headers.get("content-type") or "").lower()
    query: str | None = None
    session_id: str | None = None
    agent_hint: str | None = None
    image_path: str | None = None
    has_image_upload = False

    if "multipart/form-data" in content_type:
        form = await request.form()
        query = (form.get("query") or "").strip()
        session_id = (form.get("session_id") or "").strip() or None
        raw_hint = form.get("agent_hint")
        agent_hint = raw_hint.strip() if isinstance(raw_hint, str) else raw_hint
        upload = form.get("image_file")
        has_image_upload = bool(upload)
        logger.info(
            "API multipart | session_id=%s | agent_hint=%s | has_image=%s",
            session_id,
            agent_hint or "",
            has_image_upload,
        )
        upload_has_reader = bool(upload) and hasattr(upload, "read")
        if upload:
            logger.info(
                "API multipart upload_type | type=%s | has_reader=%s",
                type(upload).__name__,
                upload_has_reader,
            )
        if upload_has_reader:
            logger.info(
                "API multipart file | filename=%s | content_type=%s",
                getattr(upload, "filename", "") or "",
                getattr(upload, "content_type", "") or "",
            )
            image_path = await _persist_image(upload)
    else:
        try:
            payload_dict = await request.json()
        except Exception as exc:
            logger.error("API ERROR /routed_query | invalid_json | error=%s", exc)
            raise HTTPException(400, "Invalid request payload")
        try:
            payload = RoutedQueryRequest(**payload_dict)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors())
        query = payload.query
        session_id = payload.session_id
        agent_hint = payload.agent_hint
        image_path = None

    if query is None:
        logger.error("API ERROR /routed_query | missing_query")
        raise HTTPException(400, "Missing 'query' parameter")
    if not query.strip() and not (image_path or has_image_upload or agent_hint == "image_qa"):
        logger.error("API ERROR /routed_query | empty_query")
        raise HTTPException(400, "Query cannot be empty")
    if not session_id:
        logger.error("API ERROR /routed_query | missing_session_id")
        raise HTTPException(400, "Missing 'session_id' parameter")

    logger.info(
        "API INPUT /routed_query | session_id=%s | agent_hint=%s | image=%s",
        session_id,
        agent_hint,
        bool(image_path),
    )

    try:
        agent_type, result_payload = _orchestrator.run(
            query,
            session_id,
            agent_hint=agent_hint,
            image_path=image_path,
        )
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(500, f"Agent error: {exc}")

    if isinstance(result_payload, dict):
        answer_text = result_payload.get("output", "I don't know.")
        citations = result_payload.get("citations", []) or []
        resources_text = result_payload.get("resources_text", "") or ""
    else:
        answer_text = str(result_payload)
        citations = []
        resources_text = ""

    if citations and not resources_text:
        resources_text = format_resources_section(citations)
    answer_text = append_resources_to_answer(answer_text, resources_text)

    elapsed = round(time.time() - start_time, 2)
    logger.info(
        "API OUTPUT /routed_query | agent_type=%s | elapsed=%ss | used_image=%s",
        agent_type,
        elapsed,
        bool(image_path),
    )
    return {
        "answer": answer_text,
        "metadata": {
            "query_time_seconds": elapsed,
            "agent_type": agent_type,
            "citations": citations,
            "used_image": bool(image_path),
            "resources_text": resources_text,
        },
    }


async def _persist_image(upload: UploadFile) -> str:
    filename = upload.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower().lstrip(".")
    if not suffix:
        suffix = _extension_from_content_type(upload.content_type) or "png"
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        mapped = _extension_from_content_type(upload.content_type)
        if mapped and mapped in ALLOWED_IMAGE_EXTENSIONS:
            logger.warning(
                "image_suffix_overridden | filename=%s | suffix=%s | mapped=%s",
                filename,
                suffix,
                mapped,
            )
            suffix = mapped
        else:
            logger.warning("image_rejected_extension | filename=%s | suffix=%s", filename, suffix)
            raise HTTPException(415, f"Unsupported image type: {suffix}")

    data = await upload.read()
    if not data:
        logger.warning("image_rejected_empty | filename=%s", filename)
        raise HTTPException(400, "Uploaded image is empty")
    if len(data) > MAX_IMAGE_UPLOAD_BYTES:
        logger.warning(
            "image_rejected_size | filename=%s | size_bytes=%d | max_bytes=%d",
            filename,
            len(data),
            MAX_IMAGE_UPLOAD_BYTES,
        )
        raise HTTPException(
            413,
            f"Image exceeds the maximum size of {int(MAX_IMAGE_UPLOAD_MB)} MB",
        )

    target_path = IMAGES_DIR / f"{uuid4().hex}.{suffix}"
    with open(target_path, "wb") as handle:
        handle.write(data)
    logger.info(
        "image_saved | original=%s | path=%s | size_bytes=%d",
        filename,
        target_path,
        len(data),
    )
    return str(target_path)


def _extension_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    normalized = content_type.lower().strip()
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }
    return mapping.get(normalized)