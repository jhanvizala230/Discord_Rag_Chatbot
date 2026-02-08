"""Agent module - modular agent architecture."""

from .orchestrator import AgentOrchestrator

# Singleton instance
_ORCHESTRATOR: AgentOrchestrator | None = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = AgentOrchestrator(use_routing=True)
    return _ORCHESTRATOR


__all__ = ["AgentOrchestrator", "get_agent_orchestrator"]
