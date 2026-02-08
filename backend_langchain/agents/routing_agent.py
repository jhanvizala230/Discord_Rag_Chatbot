"""Routing agent - lightweight classifier for specialist agents."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from ..core.logger import setup_logger
from ..core.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = setup_logger(__name__)

AgentType = Literal["doc", "smalltalk", "finance_info", "image_qa", "clarify"]

_VALID_HINTS = {"doc", "smalltalk", "finance_info", "image_qa", "clarify"}
_SMALLTALK_KEYWORDS = (
    "hello",
    "hi",
    "hey",
    "how are you",
    "good morning",
    "good evening",
    "motivate",
    "motivation",
    "encourage",
    "joke",
    "fun fact",
    "tell me something",
    "inspire",
)
_DOC_KEYWORDS = (
    "document",
    "documents",
    "doc",
    "pdf",
    "contract",
    "report",
    "page",
    "pages",
    "table",
    "tables",
    "spreadsheet",
    "uploaded",
    "upload",
    "file",
    "files",
    "clause",
    "handbook",
    "manual",
    "policy wording",
    "lease",
    "wordings",
    "appendix",
)
_AMBIGUOUS_KEYWORDS = (
    "benefits",
    "options",
    "plans",
    "details",
    "overview",
    "information",
    "explain",
    "compare",
)
_FINANCE_KEYWORDS = (
    "policy",
    "policies",
    "insurance",
    "investment",
    "mutual fund",
    "sip",
    "stock",
    "finance",
    "financial",
    "loan",
    "tax",
    "pf",
    "ppf",
    "elss",
    "pension",
    "scheme",
    "government scheme",
    "lic",
    "health cover",
    "medical cover",
    "insurance policy",
    "market",
    "latest rate",
)
_FRESH_DATA_KEYWORDS = (
    "current",
    "today",
    "latest",
    "live",
    "updated",
    "trend",
    "news",
    "market",
)


class RoutingAgent:
    """Rule-enhanced router with optional agent hints."""

    def __init__(self) -> None:
        prompt_path = Path(__file__).resolve().parent.parent / "core" / "prompts_txt" / "routing_agent_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as handle:
            prompt_text = handle.read().strip()

        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=min(256, LLM_MAX_TOKENS),
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("routing_agent_initialized | mode=heuristic+llm")

    def decide(
        self,
        query: str,
        session_id: str,
        *,
        agent_hint: Optional[str] = None,
        has_image: bool = False,
        history: Optional[str] = None,
        previous_agent: Optional[str] = None,
        previous_query: Optional[str] = None,
        has_prior_image: bool = False,
    ) -> AgentType:
        """Return best-fit agent using hints, attachments, and heuristics."""

        if has_image:
            logger.debug("routing_decision | session_id=%s | signal=image | route=image_qa", session_id)
            return "image_qa"

        normalized_hint = self._normalize_hint(agent_hint)
        if normalized_hint:
            logger.debug(
                "routing_decision | session_id=%s | hint=%s | route=%s",
                session_id,
                agent_hint,
                normalized_hint,
            )
            return normalized_hint

        query_text = (query or "").lower()

        llm_choice = self._decide_with_llm(
            query=query,
            agent_hint=agent_hint,
            history=history or "",
            previous_agent=previous_agent or "",
            previous_query=previous_query or "",
            has_prior_image=has_prior_image,
        )
        if llm_choice:
            logger.debug(
                "routing_decision | session_id=%s | mode=llm | route=%s",
                session_id,
                llm_choice,
            )
            return llm_choice
        history_text = (history or "").lower()
        doc_context_signal = self._history_mentions_docs(history_text)
        doc_like = doc_context_signal or self._is_doc_query(query_text)
        finance_like = self._is_finance(query_text) and not doc_like and not doc_context_signal
        ambiguous = self._needs_clarification(query_text, doc_like, finance_like)

        if self._is_smalltalk(query_text):
            logger.debug("routing_decision | session_id=%s | route=smalltalk", session_id)
            return "smalltalk"

        if doc_like:
            logger.debug("routing_decision | session_id=%s | route=doc", session_id)
            return "doc"

        if finance_like:
            logger.debug("routing_decision | session_id=%s | route=finance_info", session_id)
            return "finance_info"

        if ambiguous:
            logger.debug("routing_decision | session_id=%s | ambiguous_keyword | route=clarify", session_id)
            return "clarify"

        logger.debug("routing_decision | session_id=%s | route=doc", session_id)
        return "doc"

    def _decide_with_llm(
        self,
        *,
        query: str,
        agent_hint: Optional[str],
        history: str,
        previous_agent: str,
        previous_query: str,
        has_prior_image: bool,
    ) -> Optional[AgentType]:
        try:
            response = self.chain.invoke(
                {
                    "history": history,
                    "context": "",
                    "query": query,
                    "agent_hint": agent_hint or "",
                    "previous_agent": previous_agent,
                    "previous_query": previous_query,
                    "has_prior_image": str(has_prior_image).lower(),
                }
            )
        except Exception:
            logger.exception("routing_llm_failed")
            return None

        if not response:
            return None
        normalized = response.strip().lower()
        if normalized in _VALID_HINTS:
            return normalized  # type: ignore[return-value]
        return None

    def _normalize_hint(self, agent_hint: Optional[str]) -> Optional[AgentType]:
        if not agent_hint:
            return None
        hint = agent_hint.strip().lower()
        return hint if hint in _VALID_HINTS else None

    def _is_smalltalk(self, query: str) -> bool:
        return any(keyword in query for keyword in _SMALLTALK_KEYWORDS)

    def _is_finance(self, query: str) -> bool:
        if any(keyword in query for keyword in _DOC_KEYWORDS):
            return False
        return any(keyword in query for keyword in _FINANCE_KEYWORDS)

    def _is_doc_query(self, query: str) -> bool:
        return any(keyword in query for keyword in _DOC_KEYWORDS)

    def _needs_clarification(self, query: str, doc_like: bool, finance_like: bool) -> bool:
        if doc_like or finance_like:
            return False
        return any(keyword in query for keyword in _AMBIGUOUS_KEYWORDS)

    def _history_mentions_docs(self, history: str) -> bool:
        if not history:
            return False
        doc_markers = ("uploaded", "saved", "document", "pdf", "handbook", "policy", "lease")
        return any(marker in history for marker in doc_markers)
