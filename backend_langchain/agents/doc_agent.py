"""Document Q&A agent using LangChain chains."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from ..core.logger import setup_logger
from ..core.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DOCUMENT_DOWNLOAD_BASE_URL,
    TOP_K,
    FINAL_K,
)
from ..services.chat_service import get_agent_memory
from ..services.citation_formatter import (
    append_resources_to_answer,
    format_resources_section,
)
from ..vector_and_db.embeddings import Embedder
from ..vector_and_db.vectorstore import query_top_k
from ..services.reranker import rerank
from ..vector_and_db.db import get_document

logger = setup_logger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "core" / "prompts_txt"


def _load_prompt(filename: str) -> str:
    with open(PROMPT_DIR / filename, "r", encoding="utf-8") as handle:
        return handle.read().strip()


QUERY_REFINEMENT_PROMPT_TEXT = _load_prompt("query_refinement_prompt.txt")
ANSWER_SYNTHESIS_PROMPT_TEXT = _load_prompt("answer_synthesis_prompt.txt")


class DocAgent:
    """Document Q&A using LangChain chains with memory."""
    
    def __init__(
        self,
        session_id: str,
        window_size: int = 6,
        top_k: int = TOP_K,
        final_k: int = FINAL_K,
    ):
        self.session_id = session_id
        self.top_k = top_k
        self.final_k = final_k
        self.embedder = Embedder()
        self._doc_metadata_cache: Dict[str, dict] = {}
        
        # Get conversation memory (LangChain ConversationBufferWindowMemory)
        self.memory = get_agent_memory(session_id, k=window_size)
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
        )
        
        # Build the two chains
        self.query_refinement_chain = self._build_query_refinement_chain()
        self.answer_synthesis_chain = self._build_answer_synthesis_chain()
        
        logger.info(f"doc_agent_initialized | session_id={session_id}")
    
    def _build_query_refinement_chain(self) -> Runnable:
        """CHAIN 1: Query refinement using ChatPromptTemplate + memory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_REFINEMENT_PROMPT_TEXT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_query}")
        ])
        
        # Chain: input → prompt → llm → parse output
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _build_answer_synthesis_chain(self) -> Runnable:
        """CHAIN 2: Answer synthesis using ChatPromptTemplate + memory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYNTHESIS_PROMPT_TEXT),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Context:\n{retrieved_context}\n\nPreference note: {preference_note}\n\nQuestion: {user_query}",
            ),
        ])
        
        # Chain: input → prompt → llm → parse output
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _parse_refined_queries(self, refinement_output: str, original_query: str) -> List[str]:
        """Parse numbered list from query refinement output."""
        refined_queries = [original_query]  # Always include original
        
        lines = refinement_output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering: "1. query" or "- query"
                query = line.split(".", 1)[-1].split("-", 1)[-1].strip()
                if query and query not in refined_queries:
                    refined_queries.append(query)
        
        return refined_queries[:3]  # Max 3 queries
    
    def _retrieve_documents(self, queries: List[str]) -> List[dict]:
        """Retrieve and deduplicate documents from multiple queries."""
        all_chunks = []
        seen_chunk_ids = set()
        
        for query in queries:
            query_embedding = self.embedder.embed_text(query)
            candidates = query_top_k(query_embedding, top_k=self.top_k)
            
            # Deduplicate by chunk_id
            for item in candidates:
                chunk_id = item.get("meta", {}).get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunks.append({
                        "text": item.get("text"),
                        "meta": item.get("meta", {}),
                        "distance": item.get("distance")
                    })
        
        # Rerank combined results
        all_chunks = rerank(queries[0], all_chunks)
        
        logger.info(f"retrieved_documents | total={len(all_chunks)}")
        return all_chunks[:self.final_k]
    
    def _parse_page(self, raw_page) -> tuple[None | int, str]:
        """Return numeric page (if available) and a human label."""
        if raw_page in (None, "", "None"):
            return None, "Page unknown"
        try:
            page_num = int(raw_page)
            if page_num < 1:
                page_num = 1
            return page_num, f"Page {page_num}"
        except (ValueError, TypeError):
            label = str(raw_page).strip()
            if not label:
                return None, "Page unknown"
            return None, f"Page {label}"

    def _build_source_url(self, doc_id: str | None, page_number: int | None) -> str:
        """Compose a deep link to the document file if doc_id is known."""
        base = DOCUMENT_DOWNLOAD_BASE_URL.rstrip("/")
        if not doc_id:
            return base
        url = f"{base}/documents/{doc_id}/file"
        if page_number is not None:
            return f"{url}#page={page_number}"
        return url

    def _build_context_text(self, chunks: List[dict]) -> str:
        """Format chunks as context string with doc + page metadata."""
        if not chunks:
            return "No relevant documents found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            meta = chunk.get("meta", {}) or {}
            doc_id = meta.get("doc_id", "unknown")
            title = meta.get("title") or meta.get("source") or doc_id
            page_number, page_label = self._parse_page(meta.get("page"))
            header = f"[Document {i} - {title} | {page_label} | ID: {doc_id}]"
            context_parts.append(f"{header}\n{text}")

        return "\n\n".join(context_parts)

    def _build_citations(self, chunks: List[dict]) -> List[dict]:
        """Aggregate chunk metadata into unique document citations."""
        grouped: Dict[str, dict] = {}

        for chunk in chunks:
            meta = chunk.get("meta", {}) or {}
            doc_id = meta.get("doc_id") or "unknown"
            entry = grouped.get(doc_id)
            if entry is None:
                download_target = None if doc_id == "unknown" else self._build_source_url(doc_id, None)
                entry = {
                    "type": "document",
                    "doc_id": doc_id,
                    "title": self._resolve_doc_title(doc_id, meta),
                    "download_url": download_target,
                    "pages": set(),
                    "has_unknown_page": False,
                }
                grouped[doc_id] = entry
            page_number, _ = self._parse_page(meta.get("page"))
            if page_number is not None:
                entry["pages"].add(page_number)
            else:
                entry["has_unknown_page"] = True

        citations: List[dict] = []
        for entry in grouped.values():
            pages = sorted(entry.get("pages", set()))
            entry["pages"] = pages
            entry["page_label"] = self._summarize_page_label(pages, entry.get("has_unknown_page"))
            citations.append(entry)

        return citations

    def _summarize_page_label(self, pages: List[int], has_unknown: bool) -> str:
        if not pages and not has_unknown:
            return ""
        parts: List[str] = []
        if pages:
            parts.append(f"Pages {', '.join(self._condense_ranges(pages))}")
        if has_unknown:
            parts.append("Page unknown")
        return "; ".join(parts)

    def _condense_ranges(self, pages: List[int]) -> List[str]:
        if not pages:
            return []
        results: List[str] = []
        start = prev = pages[0]
        for num in pages[1:]:
            if num == prev + 1:
                prev = num
                continue
            results.append(self._format_range(start, prev))
            start = prev = num
        results.append(self._format_range(start, prev))
        return results

    @staticmethod
    def _format_range(start: int, end: int) -> str:
        return f"{start}-{end}" if start != end else str(start)

    def _resolve_doc_title(self, doc_id: Optional[str], meta: Optional[dict]) -> str:
        meta = meta or {}
        if not doc_id or doc_id == "unknown":
            return meta.get("title") or meta.get("source") or "Unknown document"
        if doc_id not in self._doc_metadata_cache:
            try:
                self._doc_metadata_cache[doc_id] = get_document(doc_id) or {}
            except Exception:
                logger.debug("doc_metadata_lookup_failed | doc_id=%s", doc_id)
                self._doc_metadata_cache[doc_id] = {}
        doc_meta = self._doc_metadata_cache.get(doc_id) or {}
        return (
            doc_meta.get("title")
            or doc_meta.get("filename")
            or meta.get("title")
            or meta.get("source")
            or doc_id
        )
    
    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Execute the two-chain workflow: refine → retrieve → synthesize."""
        user_query = inputs.get("input", "")
        preference_note = inputs.get(
            "preference_note",
            "No explicit preference provided; rely on uploaded documents by default.",
        )
        logger.info(f"doc_agent_invoke | query={user_query}")
        
        try:
            # STEP 1: Refine query using first chain
            refinement_output = self.query_refinement_chain.invoke({
                "user_query": user_query
            })
            refined_queries = self._parse_refined_queries(refinement_output, user_query)
            logger.info(f"refined_queries | count={len(refined_queries)}")
            
            # STEP 2: Retrieve documents
            chunks = self._retrieve_documents(refined_queries)
            context_text = self._build_context_text(chunks)
            citations = self._build_citations(chunks)
            resources_text = format_resources_section(citations)
            
            # STEP 3: Synthesize answer using second chain
            answer = self.answer_synthesis_chain.invoke({
                "user_query": user_query,
                "retrieved_context": context_text,
                "preference_note": preference_note,
            })

            # STEP 4: Save to memory
            final_answer = append_resources_to_answer(answer, resources_text)
            self.memory.save_context(
                {"input": user_query},
                {"output": final_answer}
            )

            logger.info(f"answer_generated | length={len(final_answer)} | citations={len(citations)}")
            return {
                "output": final_answer,
                "citations": citations,
                "resources_text": resources_text,
            }
            
        except Exception as exc:
            logger.exception("doc_agent_error")
            return {
                "output": "I encountered an error. Please try again.",
                "citations": [],
                "resources_text": "",
            }
