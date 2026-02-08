# DocIntel - Document Intelligence RAG System

DocIntel is a document-based retrieval-augmented generation (RAG) system built on FastAPI, LangChain, and Ollama. It ships with a Discord bot that forwards user questions to the backend.

## Architecture

```
                                  +------------------+
                                  |   Ollama LLM     |
                                  |  http://ollama   |
                                  +---------+--------+
                                            |
                                            |
  +---------------+        +----------------v-----------------+
  |  Discord Bot  |        |            Backend API            |
  | (discord_bot) +------->|  FastAPI + LangChain Agents       |
  +-------+-------+        |  /upload, /routed_query, /chat/*  |
          |                +-----------+-----------------------+
          |                            |
          |                            v
          |                 +-------------------------+
          |                 | Vector + SQL Storage    |
          |                 | Chroma + SQLite + Files |
          |                 +-------------------------+
          |
          |  (client uploads and queries)
          v
  +------------------+
  | External Clients |
  +------------------+
```

## Repository layout

- backend_langchain: FastAPI app, agents, data extraction, and storage logic
- discord_bot: Discord client that calls the backend API
- docker-compose.yml: Multi-service local deployment (backend + bot + Ollama)

## Quick start (Docker)

1. Build and start services:
```bash
docker compose up --build
```

2. (Optional) Pull the default model inside Ollama:
```bash
docker exec -it document_analyzer-ollama-1 ollama pull qwen2.5:7b-instruct-q4_K_M
```

3. Open the API docs:
```text
http://localhost:8000/docs
```

## Local development (no Docker)

Backend:
```bash
pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng
uvicorn backend_langchain.app:app --reload --port 8000
```

Discord bot:
```bash
cd discord_bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python bot.py
```

## Environment variables

Backend (common):

- OLLAMA_BASE_URL: `http://ollama:11434` in Docker, `http://localhost:11434` locally
- OLLAMA_URL: `http://ollama:11434/api/generate` in Docker
- LLM_MODEL: `qwen2.5:7b-instruct-q4_K_M`
- EMBEDDING_MODEL: `sentence-transformers/all-MiniLM-L12-v2`
- RERANKER_MODEL: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- DOCINTEL_API_BASE: used to build citation download links
- DOCUMENT_DOWNLOAD_BASE_URL: overrides the citation download base URL

Discord bot:

- DISCORD_BOT_TOKEN: required
- DOCINTEL_API_BASE: `http://backend:8000` inside Docker, `http://localhost:8000` locally
- DOCINTEL_REQUEST_TIMEOUT: request timeout in seconds (default 15)

## Data persistence

Backend data is stored under `backend_langchain/output` and is persisted via a Docker volume when running with compose.

## Troubleshooting

- Build error: missing snapshot in BuildKit cache
  - Run `docker builder prune -af`, restart Docker, then rebuild.
- Ollama not reachable
  - Ensure the `ollama` service is running and `OLLAMA_BASE_URL` points to `http://ollama:11434` in Docker.
```json
{
  "answer": "According to the uploaded documents, employee termination policies include...",
  "metadata": {
    "query_time_seconds": 2.34,
    "agent_type": "doc"
  }
}
```

**Workflow** (DocAgent):
```
User Query
    ↓
┌───────────────────────────────────────────┐
│ 1. Query Refinement Chain                 │
│    • Loads last 6 conversation exchanges  │
│    • Generates 2-3 optimized queries      │
│    • Example: "termination policy" →      │
│      ["employee dismissal grounds",       │
│       "contract termination reasons"]     │
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ 2. Document Retrieval                     │
│    • Embeds each refined query            │
│    • Queries ChromaDB (top 30 chunks)     │
│    • Deduplicates by chunk_id             │
│    • Reranks using CrossEncoder           │
│    • Returns top 15 most relevant chunks  │
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ 3. Answer Synthesis Chain                 │
│    • Loads conversation history           │
│    • Injects retrieved context            │
│    • Generates answer using LLM           │
│    • Cites document sources               │
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ 4. Save to Conversation History           │
│    • Saves user query to message_store    │
│    • Saves AI answer to message_store     │
│    • Maintains 6-exchange window          │
└───────────────────────────────────────────┘
```

**Conversation Persistence**:
- All conversations saved to SQLite `message_store` table
- Each session maintains independent history
- 6-turn conversation window (12 messages: 6 user + 6 AI)
- Retrieve history via `/chat/history/{session_id}`

**Example Multi-Turn Conversation**:
```bash
# Turn 1
POST /routed_query
{"query": "What is the vacation policy?", "session_id": "user-123"}
→ "Employees get 15 days paid vacation..."

# Turn 2 (references previous context)
POST /routed_query
{"query": "How do I request it?", "session_id": "user-123"}
→ "To request vacation, submit form HR-204..." (knows "it" = vacation)

# Turn 3
POST /routed_query
{"query": "What about sick leave?", "session_id": "user-123"}
→ "Sick leave policy allows..." (maintains conversation flow)
```

---

## 📚 Additional API Endpoints

### Document Management

**List Documents**
```bash
GET /getdocs
```
Returns all uploaded documents with metadata.

**Get Document Chunks**
```bash
GET /getdoc_id/{doc_id}
```
Returns all chunks for a specific document.

**Delete Document**
```bash
DELETE /doc/{doc_id}
```
Removes document from both ChromaDB and SQLite.

**Delete All Documents**
```bash
DELETE /docs
```
Clears entire vector database and document metadata.

### Conversation Management

**Get Chat History**
```bash
GET /chat/history/{session_id}
```
Returns all messages for a conversation.

**List All Conversations**
```bash
GET /chat/conversations
```
Returns all session IDs with message counts.

### Utilities

**Health Check**
```bash
GET /health
```
Returns API status.

**Drift Detection**
```bash
GET /drift/{doc_id}
```
Analyzes document drift from baseline.

---

## 🗄️ Database Schema

### SQLite Tables

#### 1. `message_store` (Conversations)
Managed by LangChain `SQLChatMessageHistory`.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `session_id` | TEXT | Conversation identifier |
| `message` | BLOB | JSON message (HumanMessage/AIMessage) |

#### 2. `documents` (Document Metadata)
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `doc_id` | TEXT | Unique document ID (UUID) |
| `filename` | TEXT | Original filename |
| `file_type` | TEXT | pdf, docx, txt |
| `file_size` | INTEGER | Size in bytes |
| `title` | TEXT | Document title |
| `author` | TEXT | Document author |
| `num_chunks` | INTEGER | Number of text chunks |
| `metadata` | TEXT | Additional metadata (JSON) |
| `created_at` | DATETIME | Upload timestamp |

#### 3. `raw_tables` (PDF Table Data)
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `doc_id` | TEXT | Document ID |
| `page` | INTEGER | Page number |
| `csv` | TEXT | Table data as CSV string |
| `created_at` | DATETIME | Extraction timestamp |

### ChromaDB Collections

#### `text_collection` (Vector Store)
Stores embeddings + metadata for all document chunks.

**Metadata per chunk**:
```json
{
  "doc_id": "a1b2c3d4-...",
  "chunk_id": 0,
  "chunk_type": "text",
  "chunk_size": 782,
  "title": "Company Policy",
  "author": "HR Department",
  "file_type": "pdf",
  "page": 5,
  "entity_type": "text",
  "source_id": "a1b2c3d4-...",
  "created_at": "2026-02-01T10:30:00"
}
```

---

## 🧩 Project Structure

```
backend/
├── backend_langchain/
│   ├── app.py                      # FastAPI application
│   ├── agents/                     # Agent system (modular)
│   │   ├── __init__.py            # Module exports
│   │   ├── doc_agent.py           # Document Q&A agent
│   │   ├── orchestrator.py        # Agent routing
│   │   ├── routing_agent.py       # Query classification (future)
│   │   └── ARCHITECTURE.md        # Agent architecture docs
│   ├── api/                        # API endpoints
│   │   ├── document.py            # Upload, list, delete
│   │   ├── chat.py                # Chat history
│   │   └── agentic.py             # /routed_query endpoint
│   ├── services/                   # Business logic
│   │   ├── chat_service.py        # Conversation management
│   │   └── reranker.py            # CrossEncoder reranking
│   ├── vector_and_db/              # Data layer
│   │   ├── db.py                  # SQLite operations
│   │   ├── vectorstore.py         # ChromaDB operations
│   │   ├── embeddings.py          # Embedding models
│   │   └── table_file_store.py    # Table storage
│   ├── data_extraction/            # Document processing
│   │   ├── pdf_text_extraction.py # PyMuPDF4LLM
│   │   ├── pdf_table_extraction.py# Table extraction
│   │   └── extractor.py           # Generic extractor
│   ├── core/                       # Configuration
│   │   ├── config.py              # Environment variables
│   │   ├── logger.py              # Logging setup
│   │   └── prompts_txt/           # LLM prompts
│   │       ├── query_refinement_prompt.txt
│   │       └── answer_synthesis_prompt.txt
│   ├── validations/                # Quality checks
│   │   ├── drift.py               # Evidently drift detection
│   │   └── langsmith_client.py    # LangSmith integration
│   └── output/                     # Generated data
│       ├── chroma_store_langchain/ # Vector DB
│       ├── sql/                    # SQLite DB
│       ├── uploaded_docs_langchain/# Original files
│       └── logs/                   # Application logs
└── requirements.txt
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `qwen2.5:7b-instruct-q4_K_M` | LLM model name |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_TOKENS` | `1024` | Max output length |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L12-v2` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `TOP_K` | `30` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `600` | Chunk size for text splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

---

## 🧪 Testing

### 1. Upload a document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test.pdf" \
  -F "title=Test Document"
```

### 2. Ask a question
```bash
curl -X POST "http://localhost:8000/routed_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "session_id": "test-1"}'
```

### 3. Check conversation history
```bash
curl "http://localhost:8000/chat/history/test-1"
```

### 4. List documents
```bash
curl "http://localhost:8000/getdocs"
```

---

## 🐛 Troubleshooting

### ChromaDB Migration Error
If you see "deprecated configuration" error:
```bash
pip install chroma-migrate
chroma-migrate
```

### Import Errors
If IDE shows import errors for `langchain_core`, `langchain_community`, etc., these are false positives. Packages are installed correctly.

### Ollama Connection Failed
Ensure Ollama is running:
```bash
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Conversation Not Saving
Check `message_store` table exists:
```python
import sqlite3
conn = sqlite3.connect("backend_langchain/output/sql/docintel.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(tables)  # Should include message_store
```

---

## 📦 Dependencies

### Backend
- **FastAPI** - API framework
- **LangChain** - LLM orchestration
- **ChromaDB** - Vector database
- **Sentence-Transformers** - Embeddings
- **PyMuPDF4LLM** - PDF extraction
- **Ollama** - Local LLM inference
- **SQLAlchemy** - SQLite ORM
- **Evidently** - Drift detection

---

## 🚀 Production Deployment

```bash
# Use production ASGI server
pip install gunicorn
gunicorn backend_langchain.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

