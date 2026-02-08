"""Image question-answering agent powered by Ollama LLaVA-Phi3."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List

import requests

from ..core.logger import setup_logger
from ..services.chat_service import get_agent_memory
from ..core.config import OLLAMA_BASE_URL, VISION_MODEL

logger = setup_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "core" / "prompts_txt" / "image_qa_prompt.txt"


def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read().strip()


IMAGE_QA_PROMPT = _load_prompt()


class ImageQAAgent:
    """Runs multimodal inference using LLaVA-Phi3 served by Ollama."""

    def __init__(self, session_id: str, window_size: int = 4) -> None:
        self.model = VISION_MODEL
        self.endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        self.session_id = session_id
        self.memory = get_agent_memory(session_id, k=window_size)
        self._warned_model = False

    def _format_history(self, messages: List[object]) -> str:
        lines = []
        for msg in messages:
            role = getattr(msg, "type", None)
            content = getattr(msg, "content", "")
            if not content:
                continue
            prefix = "User" if role == "human" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        image_path = inputs.get("image_path")
        question = inputs.get("input", "")
        if not image_path:
            logger.warning("image_qa_missing_path | question_length=%d", len(question or ""))
            return {"output": "I didn't receive an image to analyze.", "citations": [], "resources_text": ""}

        path_obj = Path(image_path)
        if not path_obj.exists():
            logger.warning("image_qa_missing_file | path=%s", image_path)
            return {
                "output": "The referenced image is missing on the server.",
                "citations": [],
                "resources_text": "",
            }

        try:
            encoded = base64.b64encode(path_obj.read_bytes()).decode("utf-8")
        except Exception as exc:
            logger.exception("image_qa_read_failed | path=%s", image_path)
            return {
                "output": "I could not read the uploaded image.",
                "citations": [],
                "resources_text": "",
            }

        model_lower = (self.model or "").lower()
        if not self._warned_model:
            vision_tokens = (
                "llava",
                "vision",
                "qwen-vl",
                "qwenvl",
                "minicpm",
                "moondream",
            )
            if not any(token in model_lower for token in vision_tokens):
                logger.warning("image_qa_model_maybe_text_only | model=%s", self.model)
            self._warned_model = True

        history_msgs = self.memory.load_memory_variables({}).get("chat_history", [])
        history_text = self._format_history(history_msgs)
        if history_text:
            prompt = (
                f"{IMAGE_QA_PROMPT}\n\nConversation history:\n{history_text}"
                f"\n\nUser question: {question}"
            )
        else:
            prompt = f"{IMAGE_QA_PROMPT}\n\nUser question: {question}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
        }

        logger.info("image_qa_request | model=%s | image_bytes=%d", self.model, len(encoded))

        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            answer = data.get("response") or data.get("message", {}).get("content")
            if not answer:
                answer = "I couldn't interpret the response from the vision model."
            output_text = answer.strip()
            self.memory.save_context({"input": question}, {"output": output_text})
            return {"output": output_text, "citations": [], "resources_text": ""}
        except Exception as exc:
            logger.exception("image_qa_inference_failed")
            return {
                "output": "Vision model inference failed. Please try again.",
                "citations": [],
                "resources_text": "",
            }
