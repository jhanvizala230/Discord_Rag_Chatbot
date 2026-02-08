"""Friendly conversational agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from ..core.logger import setup_logger
from ..core.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..services.chat_service import get_agent_memory

logger = setup_logger(__name__)
PROMPT_DIR = Path(__file__).resolve().parent.parent / "core" / "prompts_txt"


def _load_prompt(filename: str) -> str:
    with open(PROMPT_DIR / filename, "r", encoding="utf-8") as handle:
        return handle.read().strip()


SMALLTALK_PROMPT_TEXT = _load_prompt("smalltalk_prompt.txt")


class SmalltalkAgent:
    """Keeps casual, motivational conversations flowing."""

    def __init__(self, session_id: str, window_size: int = 6) -> None:
        self.session_id = session_id
        self.memory = get_agent_memory(session_id, k=window_size)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
        )
        self.chain = self._build_chain()
        logger.info("smalltalk_agent_initialized | session_id=%s", session_id)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SMALLTALK_PROMPT_TEXT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_query}"),
            ]
        )
        return (
            RunnablePassthrough.assign(
                chat_history=lambda _: self.memory.load_memory_variables({})["chat_history"]
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        user_query = inputs.get("input", "")
        try:
            response_text = self.chain.invoke({"user_query": user_query})
            self.memory.save_context({"input": user_query}, {"output": response_text})
            return {"output": response_text, "citations": [], "resources_text": ""}
        except Exception as exc:
            logger.exception("smalltalk_agent_error")
            return {
                "output": "I ran into a hiccup, could you try again?",
                "citations": [],
                "resources_text": "",
            }
