"""Finance information agent leveraging DuckDuckGo search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from ..core.logger import setup_logger
from ..core.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from ..services.chat_service import get_agent_memory
from ..services.citation_formatter import (
    append_resources_to_answer,
    format_resources_section,
)
from ..services.web_search import duckduckgo_search

logger = setup_logger(__name__)
PROMPT_DIR = Path(__file__).resolve().parent.parent / "core" / "prompts_txt"


def _load_prompt(filename: str) -> str:
    with open(PROMPT_DIR / filename, "r", encoding="utf-8") as handle:
        return handle.read().strip()


FINANCE_PROMPT_TEXT = _load_prompt("finance_agent_prompt.txt")


class FinanceAgent:
    """Fetches up-to-date Indian financial information."""

    def __init__(self, session_id: str, window_size: int = 4) -> None:
        self.session_id = session_id
        self.memory = get_agent_memory(session_id, k=window_size)
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FINANCE_PROMPT_TEXT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    "Question: {user_query}\nPreference note: {preference_note}\n\nWeb snippets:\n{search_context}",
                ),
            ]
        )
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda _: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        user_query = inputs.get("input", "")
        preference_note = inputs.get(
            "preference_note",
            "No explicit preference provided; leverage the freshest reliable web sources.",
        )
        snippets = duckduckgo_search(user_query, max_results=5)
        if not snippets:
            message = (
                "I couldn't retrieve reliable financial information right now. "
                "Please try again in a moment."
            )
            return {"output": message, "citations": [], "resources_text": ""}

        context_text = self._format_snippets(snippets)

        try:
            response_text = self.chain.invoke(
                {
                    "user_query": user_query,
                    "search_context": context_text,
                    "preference_note": preference_note,
                }
            )
            citations = self._build_citations(snippets)
            resources_text = format_resources_section(citations)
            final_output = append_resources_to_answer(response_text, resources_text)
            self.memory.save_context({"input": user_query}, {"output": final_output})
            return {
                "output": final_output,
                "citations": citations,
                "resources_text": resources_text,
            }
        except Exception as exc:
            logger.exception("finance_agent_error")
            return {
                "output": "I ran into an issue fetching financial details. Could you retry shortly?",
                "citations": [],
                "resources_text": "",
            }

    def _format_snippets(self, snippets: List[Dict[str, str]]) -> str:
        blocks = []
        for idx, item in enumerate(snippets, start=1):
            title = item.get("title", "Unknown")
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            blocks.append(f"[{idx}] {title}\nSnippet: {snippet}\nURL: {url}")
        return "\n\n".join(blocks)

    def _build_citations(self, snippets: List[Dict[str, str]]) -> List[Dict[str, str]]:
        citations: List[Dict[str, str]] = []
        for item in snippets[:3]:
            url = item.get("url", "")
            parsed = urlparse(url) if url else None
            domain = (parsed.netloc or "").lower() if parsed else ""
            favicon_url = f"https://www.google.com/s2/favicons?sz=64&domain={domain}" if domain else None
            citations.append(
                {
                    "type": "web",
                    "title": item.get("title") or domain or "Web source",
                    "display_name": domain or item.get("title") or "Web",
                    "url": url,
                    "favicon_url": favicon_url,
                    "snippet": item.get("snippet", ""),
                }
            )
        return citations
