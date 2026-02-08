# DocIntel Backend (FastAPI + LangChain)

The backend exposes document ingestion and routed Q&A endpoints backed by Chroma, SQLite, and Ollama.

## Run with Docker Compose

From the repository root:
```bash
docker compose up --build
```

The API is available at `http://localhost:8000` and docs at `http://localhost:8000/docs`.

## Run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data:
```bash
python -m nltk.downloader averaged_perceptron_tagger_eng
```

3. Start the API server:
```bash
uvicorn backend_langchain.app:app --reload --port 8000
```

## API endpoints

- POST /upload
- GET /getdocs
- GET /getdoc_id/{doc_id}
- DELETE /doc/{doc_id}
- POST /routed_query
- GET /chat/conversations
- GET /chat/history/{session_id}
- GET /health
- GET /docs (OpenAPI UI)

## Storage paths

- backend_langchain/output/chroma_store_langchain
- backend_langchain/output/sql/metadata_langchain.db
- backend_langchain/output/uploaded_docs_langchain
- backend_langchain/output/images
- backend_langchain/output/tables_data

## Environment variables

Common backend settings:

- CHUNK_SIZE, CHUNK_OVERLAP
- EMBEDDING_MODEL, EMBEDDING_PROVIDER
- RERANKER_MODEL
- TOP_K, FINAL_K, SIMILARITY_THRESHOLD, RETRIEVAL_DISTANCE_THRESHOLD
- OLLAMA_BASE_URL, OLLAMA_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
- DOCINTEL_API_BASE, DOCUMENT_DOWNLOAD_BASE_URL
- DRIFT_SIM_THRESHOLD, ENABLE_GUARDRAILS, GUARDRAILS_ON_FAIL

## Notes

- Uses Chroma for vector storage and SQLite for metadata and chat history.
- The routed query endpoint supports optional image upload via multipart form data.