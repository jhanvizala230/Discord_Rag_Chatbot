"""Agent orchestrator with routing capabilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..services.chat_service import get_window_messages

from ..core.logger import setup_logger
from ..core.config import TOP_K, FINAL_K
from .doc_agent import DocAgent
from .routing_agent import RoutingAgent
from .smalltalk_agent import SmalltalkAgent
from .finance_agent import FinanceAgent
from .image_qa_agent import ImageQAAgent
from .clarifier_agent import ClarifierAgent

logger = setup_logger(__name__)


class AgentOrchestrator:
    """Orchestrates routing between different agents."""
    
    def __init__(self, use_routing: bool = False) -> None:
        """
        Initialize orchestrator.
        
        Args:
            use_routing: If True, use RoutingAgent to classify queries.
                        If False, always route to doc agent.
        """
        self.use_routing = use_routing
        self._doc_agents: Dict[str, DocAgent] = {}
        self._smalltalk_agents: Dict[str, SmalltalkAgent] = {}
        self._finance_agents: Dict[str, FinanceAgent] = {}
        self._image_agents: Dict[str, ImageQAAgent] = {}
        self._clarifier = ClarifierAgent()
        self._clarification_state: Dict[str, Dict[str, str]] = {}
        self._session_state: Dict[str, Dict[str, Optional[str]]] = {}
        
        # Initialize routing agent if enabled
        self._routing_agent = RoutingAgent() if use_routing else None
        
        # Future: Add more agent types here
        # self._advisor_agents: Dict[str, AdvisorAgent] = {}
        
        logger.info(f"orchestrator_initialized | use_routing={use_routing}")
    
    def _route(
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
    ) -> str:
        """
        Determine which agent to use.
        
        If use_routing=False: Always returns 'doc' unless image forces image_qa
        If use_routing=True: Uses RoutingAgent to classify with hints
        """
        normalized_hint = self._normalize_hint(agent_hint)
        if has_image:
            return "image_qa"
        if normalized_hint:
            return normalized_hint
        if not self.use_routing:
            return "doc"
        return self._routing_agent.decide(
            query,
            session_id,
            agent_hint=agent_hint,
            has_image=has_image,
            history=history,
            previous_agent=previous_agent,
            previous_query=previous_query,
            has_prior_image=has_prior_image,
        )
    
    def _get_doc_agent(self, session_id: str) -> DocAgent:
        """Get or create doc agent for session."""
        if session_id not in self._doc_agents:
            self._doc_agents[session_id] = DocAgent(
                session_id=session_id,
                window_size=6,
                top_k=TOP_K,
                final_k=FINAL_K,
            )
        return self._doc_agents[session_id]

    def _get_smalltalk_agent(self, session_id: str) -> SmalltalkAgent:
        if session_id not in self._smalltalk_agents:
            self._smalltalk_agents[session_id] = SmalltalkAgent(session_id=session_id)
        return self._smalltalk_agents[session_id]

    def _get_finance_agent(self, session_id: str) -> FinanceAgent:
        if session_id not in self._finance_agents:
            self._finance_agents[session_id] = FinanceAgent(session_id=session_id)
        return self._finance_agents[session_id]

    def _get_image_agent(self, session_id: str) -> ImageQAAgent:
        if session_id not in self._image_agents:
            self._image_agents[session_id] = ImageQAAgent(session_id=session_id)
        return self._image_agents[session_id]

    def _get_session_state(self, session_id: str) -> Dict[str, Optional[str]]:
        if session_id not in self._session_state:
            self._session_state[session_id] = {
                "last_agent": None,
                "last_user_query": None,
                "last_image_path": None,
            }
        return self._session_state[session_id]

    def _render_history(self, session_id: str, k: int = 6) -> str:
        messages = get_window_messages(session_id, k=k)
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def _normalize_hint(self, agent_hint: Optional[str]) -> Optional[str]:
        if not agent_hint:
            return None
        hint = agent_hint.strip().lower()
        if hint in {"doc", "smalltalk", "finance_info", "image_qa"}:
            return hint
        return None
    
    def run(
        self,
        query: str,
        session_id: str,
        *,
        agent_hint: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Execute query through appropriate agent.
        
        Returns:
            Tuple of (agent_type, answer)
        """
        logger.info(f"orchestrator_run | session_id={session_id}")
        
        try:
            # Route to appropriate agent
            has_image = image_path is not None
            state = self._get_session_state(session_id)
            last_agent = state.get("last_agent")
            last_user_query = state.get("last_user_query")
            last_image_path = state.get("last_image_path")
            has_prior_image = bool(last_image_path)
            resolved = self._consume_clarification(session_id, query)
            forced_route = None
            preference_note = ""
            if resolved:
                resolved_route, resolved_query, preference_note = resolved
                if resolved_route == "clarify":
                    response = self._clarifier.ask(resolved_query or query)
                    return "clarify", response
                forced_route = resolved_route
                query = resolved_query or query

            if forced_route:
                route = forced_route
            elif has_image:
                route = "image_qa"
            else:
                history_text = self._render_history(session_id, k=6)
                route = self._route(
                    query,
                    session_id,
                    agent_hint=agent_hint,
                    has_image=has_image,
                    history=history_text,
                    previous_agent=last_agent,
                    previous_query=last_user_query,
                    has_prior_image=has_prior_image,
                )
            logger.info(f"routing_decision | route={route}")
            
            # Execute based on route
            if route == "doc":
                agent = self._get_doc_agent(session_id)
                response = agent.invoke({
                    "input": query,
                    "preference_note": preference_note,
                }) or {}
            elif route == "smalltalk":
                agent = self._get_smalltalk_agent(session_id)
                response = agent.invoke({"input": query}) or {}
            elif route == "finance_info":
                agent = self._get_finance_agent(session_id)
                response = agent.invoke({
                    "input": query,
                    "preference_note": preference_note,
                }) or {}
            elif route == "image_qa":
                if not image_path and last_image_path:
                    image_path = last_image_path
                if not image_path:
                    response = {
                        "output": "I couldn't find the image attachment. Please resend the picture.",
                        "citations": [],
                        "resources_text": "",
                    }
                else:
                    logger.info("image_route | session_id=%s | image_path=%s", session_id, image_path)
                    agent = self._get_image_agent(session_id)
                    response = agent.invoke({
                        "input": query,
                        "image_path": image_path,
                    }) or {}
            elif route == "clarify":
                self._clarification_state[session_id] = {"original_query": query}
                response = self._clarifier.ask(query)
            else:
                response = {"output": "I don't know.", "citations": [], "resources_text": ""}

            if not isinstance(response, dict):
                response = {"output": response}
            response.setdefault("citations", [])
            response.setdefault("resources_text", "")

            output = response.get("output", "I don't know.")
            logger.info(f"orchestrator_complete | response_length={len(output)}")
            state["last_agent"] = route
            state["last_user_query"] = query
            if route == "image_qa" and image_path:
                state["last_image_path"] = image_path
            elif route != "image_qa":
                state["last_image_path"] = None
            return route, response
            
        except Exception as exc:
            logger.exception("orchestrator_error")
            return "doc", {"output": "I don't know.", "citations": [], "resources_text": ""}

    def _consume_clarification(
        self,
        session_id: str,
        latest_message: str,
    ) -> Optional[Tuple[str, Optional[str], str]]:
        state = self._clarification_state.get(session_id)
        if not state:
            return None

        preference = self._detect_preference(latest_message)
        if not preference:
            return ("clarify", state.get("original_query"), "")

        original_query = state.get("original_query") or latest_message
        if len(latest_message.split()) > 4:
            resolved_query = f"{original_query}\nClarification detail: {latest_message}"
        else:
            resolved_query = original_query

        preference_note = (
            "User explicitly requested an answer from uploaded documents."
            if preference == "doc"
            else "User explicitly requested an answer based on internet research."
        )

        del self._clarification_state[session_id]
        return preference, resolved_query, preference_note

    def _detect_preference(self, message: str) -> Optional[str]:
        text = (message or "").lower()
        doc_terms = ("document", "documents", "doc", "pdf", "file", "files", "upload")
        web_terms = ("internet", "online", "web", "google", "search")

        doc_match = any(term in text for term in doc_terms)
        web_match = any(term in text for term in web_terms)

        if doc_match and not web_match:
            return "doc"
        if web_match and not doc_match:
            return "finance_info"
        if "both" in text or (doc_match and web_match):
            return None
        return None
