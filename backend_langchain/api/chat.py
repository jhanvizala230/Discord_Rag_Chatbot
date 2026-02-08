from fastapi import APIRouter, HTTPException
from backend_langchain.services.chat_service import add_message, get_messages, get_all_sessions
from backend_langchain.core.logger import setup_logger
logger = setup_logger(__name__)

router = APIRouter()

@router.get("/chat/conversations")
def get_conversations():
    try:
        conversations = get_all_sessions()
        logger.info(f"Conversations retrieved | count={len(conversations)}")
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Failed to retrieve conversations: {str(e)}")
        return {"conversations": []}

@router.get("/chat/history/{session_id}")
def get_chat(session_id: str):
    try:
        history = get_messages(session_id)
        logger.info(f"Chat history retrieved | session_id={session_id} | messages={len(history)}")
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {str(e)}")
        return {"session_id": session_id, "history": []}