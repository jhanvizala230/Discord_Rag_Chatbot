

# Main FastAPI entrypoint: imports and includes all modular routers from api/ subfolder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend_langchain.api.document import router as document_router
from backend_langchain.api.chat import router as chat_router
from backend_langchain.api.agentic import router as agentic_router
from backend_langchain.core.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="DocIntel - RAG System",
    description="Document Intelligence System",
    version="2.0.0"
)
logger.info("fastapi_app_initialized | title=%s", app.title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all modular routers
app.include_router(document_router)
app.include_router(chat_router)
app.include_router(agentic_router)
logger.debug("routers_registered | document_router=1 | chat_router=1 | agentic_router=1")

@app.get("/")
def root():
    logger.debug("root_endpoint_hit")
    return {
        "service": "DocIntel RAG System",
        "version": "2.0.0",
        "status": "operational"
    }

@app.get("/health")
def health():
    logger.debug("health_endpoint_hit")
    return {
        "status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("starting_uvicorn_server | host=0.0.0.0 | port=8000")
    uvicorn.run("backend_langchain.app:app", host="0.0.0.0", port=8000, reload=True)

