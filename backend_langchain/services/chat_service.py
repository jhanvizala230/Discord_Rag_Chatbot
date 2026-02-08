"""Chat conversation service - unified interface for LangChain memory."""

from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from ..core.config import SQLITE_DB
from ..core.logger import setup_logger
from ..vector_and_db.db import ensure_chat_history_table

logger = setup_logger(__name__)
engine = create_engine(f"sqlite:///{SQLITE_DB}", connect_args={"check_same_thread": False})


def get_agent_memory(session_id: str, k: int = 6) -> ConversationBufferWindowMemory:
    """
    Get or create ConversationBufferWindowMemory backed by SQLite.
    Used by agents for conversation context.
    
    Args:
        session_id: Conversation identifier
        k: Window size (number of exchanges to remember)
    
    Returns:
        LangChain ConversationBufferWindowMemory instance
    """
    logger.debug("creating_agent_memory | session_id=%s | window_size=%s", session_id, k)
    
    try:
        ensure_chat_history_table()
        
        chat_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{SQLITE_DB}",
        )
        
        memory = ConversationBufferWindowMemory(
            k=k,
            memory_key="chat_history",
            return_messages=True,
            chat_memory=chat_history,
        )
        
        logger.info("agent_memory_initialized | session_id=%s | k=%s", session_id, k)
        return memory
        
    except Exception as exc:
        logger.error("agent_memory_init_failed | session_id=%s | error=%s", session_id, exc)
        raise


def get_chat_history(session_id: str) -> SQLChatMessageHistory:
    """
    Get SQLChatMessageHistory for direct message access.
    
    Args:
        session_id: Conversation identifier
        
    Returns:
        LangChain SQLChatMessageHistory instance
    """
    logger.debug("getting_chat_history | session_id=%s", session_id)
    try:
        ensure_chat_history_table()
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{SQLITE_DB}"
        )
        logger.debug("chat_history_initialized | session_id=%s", session_id)
        return history
    except Exception as exc:
        logger.error("chat_history_init_failed | session_id=%s | error=%s", session_id, exc)
        raise


def add_message(session_id: str, role: str, text: str) -> None:
    """
    Add message to conversation history.
    
    Args:
        session_id: Conversation identifier
        role: "user" or "assistant"
        text: Message content
    """
    logger.info("adding_message | session_id=%s | role=%s | text_length=%d", session_id, role, len(text))
    try:
        history = get_chat_history(session_id)
        if role == "user":
            history.add_user_message(text)
            logger.debug("user_message_added | session_id=%s", session_id)
        else:
            history.add_ai_message(text)
            logger.debug("ai_message_added | session_id=%s", session_id)
    except Exception as exc:
        logger.error("add_message_failed | session_id=%s | role=%s | error=%s", session_id, role, exc)
        raise


def get_messages(session_id: str) -> List[Dict[str, str]]:
    """
    Get all messages for a conversation.
    
    Args:
        session_id: Conversation identifier
        
    Returns:
        List of {"role": "user"/"assistant", "text": "..."}
    """
    logger.info("retrieving_messages | session_id=%s", session_id)
    try:
        history = get_chat_history(session_id)
        msgs = []
        for m in history.messages:
            if hasattr(m, "type"):
                role = "user" if m.type == "human" else "assistant"
            else:
                role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
            msgs.append({"role": role, "text": m.content})
        
        logger.info("messages_retrieved | session_id=%s | message_count=%d", session_id, len(msgs))
        return msgs
    except Exception as exc:
        logger.error("get_messages_failed | session_id=%s | error=%s", session_id, exc)
        return []


def get_window_messages(session_id: str, k: int = 6) -> List[Dict[str, str]]:
    """
    Get last k exchanges (2k messages) from conversation.
    
    Args:
        session_id: Conversation identifier
        k: Number of exchanges to retrieve
        
    Returns:
        List of {"role": "user"/"assistant", "text": "..."}
    """
    logger.info("retrieving_window_messages | session_id=%s | k=%d", session_id, k)
    try:
        history = get_chat_history(session_id)
        memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            chat_memory=history,
        )
        window = memory.load_memory_variables({}).get("history", [])
        msgs = []
        for m in window:
            if hasattr(m, "type"):
                role = "user" if m.type == "human" else "assistant"
                content = m.content
            else:
                role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
                content = getattr(m, "content", str(m))
            msgs.append({"role": role, "text": content})
        logger.info("window_messages_retrieved | session_id=%s | count=%d", session_id, len(msgs))
        return msgs
    except Exception as exc:
        logger.error("get_window_messages_failed | session_id=%s | error=%s", session_id, exc)
        return []


def get_all_sessions() -> List[Dict[str, any]]:
    """
    Get list of all conversation sessions.
    
    Returns:
        List of {"session_id": "...", "message_count": N, "last_message_id": N}
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT session_id, COUNT(*) as message_count, MAX(id) as last_message_id "
                     "FROM message_store GROUP BY session_id ORDER BY last_message_id DESC")
            )
            sessions = []
            for row in result:
                sessions.append({
                    "session_id": row[0],
                    "message_count": row[1],
                    "last_message_id": row[2]
                })
        logger.info("all_sessions_retrieved | count=%d", len(sessions))
        return sessions
    except Exception as exc:
        logger.error("get_all_sessions_failed | error=%s", exc)
        return []


def clear_conversation(session_id: str) -> None:
    """
    Clear all messages for a conversation.
    
    Args:
        session_id: Conversation identifier to clear
    """
    logger.info("clearing_conversation | session_id=%s", session_id)
    try:
        ensure_chat_history_table()
        chat_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{SQLITE_DB}",
        )
        chat_history.clear()
        logger.info("conversation_cleared | session_id=%s", session_id)
    except Exception as exc:
        logger.error("clear_conversation_failed | session_id=%s | error=%s", session_id, exc)
        raise
