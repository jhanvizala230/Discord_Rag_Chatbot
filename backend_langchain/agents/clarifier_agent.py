"""Clarifier agent to disambiguate doc vs web answers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from ..core.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..core.logger import setup_logger

logger = setup_logger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "core" / "prompts_txt" / "clarifier_prompt.txt"


def _load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read().strip()


CLARIFIER_PROMPT = _load_prompt()


class ClarifierAgent:
    """Generates a single follow-up question asking for doc vs internet."""

    def __init__(self) -> None:
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=min(LLM_MAX_TOKENS, 256),
        )
        self.prompt = ChatPromptTemplate.from_template(
            "{instructions}\n\nOriginal question: {original_question}"
        )
        self.parser = StrOutputParser()

    def ask(self, original_question: str) -> Dict[str, str]:
        try:
            message = self.prompt | self.llm | self.parser
            output_text = message.invoke(
                {
                    "instructions": CLARIFIER_PROMPT,
                    "original_question": original_question.strip(),
                }
            )
        except Exception as exc:
            logger.exception("clarifier_generation_failed")
            output_text = (
                "Would you like me to rely on your uploaded documents or search the internet "
                "for this answer?"
            )
        return {"output": output_text.strip(), "citations": [], "resources_text": ""}
