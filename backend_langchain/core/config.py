from pathlib import Path
import os

from .logger import setup_logger

logger = setup_logger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PACKAGE_ROOT / "output"
DB_DIR = OUTPUT_DIR / "chroma_store_langchain"
DOCS_DIR = OUTPUT_DIR / "uploaded_docs_langchain"
SQLITE_DB = OUTPUT_DIR / "sql" / "metadata_langchain.db"
TABLES_BASE_DIR = OUTPUT_DIR / "tables_data"
TABLE_JSON_DIR = TABLES_BASE_DIR / "json"
TABLE_CSV_DIR = TABLES_BASE_DIR / "csv"
IMAGES_DIR = OUTPUT_DIR / "images"

# Create directories
for path in (
	OUTPUT_DIR,
	DB_DIR,
	DOCS_DIR,
	SQLITE_DB.parent,
	TABLES_BASE_DIR,
	TABLE_JSON_DIR,
	TABLE_CSV_DIR,
	IMAGES_DIR,
):
	path.mkdir(parents=True, exist_ok=True)

# Chunking
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

# Models
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval
TOP_K = int(os.environ.get("TOP_K", 30))
FINAL_K = int(os.environ.get("FINAL_K", 15))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.65))
RETRIEVAL_DISTANCE_THRESHOLD = float(os.environ.get("RETRIEVAL_DISTANCE_THRESHOLD", 0.6))

# LLM
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 1024))
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:4b")

# Download links (used for citations)
DOCUMENT_DOWNLOAD_BASE_URL = os.environ.get(
	"DOCUMENT_DOWNLOAD_BASE_URL",
	os.environ.get("DOCINTEL_API_BASE", "http://localhost:8000"),
).rstrip("/")

ALLOWED_IMAGE_EXTENSIONS = os.environ.get(
	"ALLOWED_IMAGE_EXTENSIONS",
	"png,jpg,jpeg,bmp,gif,webp"
).lower().split(",")
ALLOWED_IMAGE_EXTENSIONS = {ext.strip() for ext in ALLOWED_IMAGE_EXTENSIONS if ext.strip()}

MAX_IMAGE_UPLOAD_MB = float(os.environ.get("MAX_IMAGE_UPLOAD_MB", "8"))
MAX_IMAGE_UPLOAD_BYTES = int(MAX_IMAGE_UPLOAD_MB * 1024 * 1024)

# Drift Detection
DRIFT_SIM_THRESHOLD = float(os.environ.get("DRIFT_SIM_THRESHOLD",0.6))

# Guardrails
ENABLE_GUARDRAILS = os.environ.get("ENABLE_GUARDRAILS", "1") == "1"
GUARDRAILS_ON_FAIL = os.environ.get("GUARDRAILS_ON_FAIL", "fix")  # fix, exception, filter, refrain

# LangSmith
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")

logger.debug(
	"config_initialized | DB_DIR=%s | DOCS_DIR=%s | SQLITE_DB=%s | CHUNK_SIZE=%s | CHUNK_OVERLAP=%s | TOP_K=%s | FINAL_K=%s | SIMILARITY_THRESHOLD=%s",
	DB_DIR,
	DOCS_DIR,
	SQLITE_DB,
	CHUNK_SIZE,
	CHUNK_OVERLAP,
	TOP_K,
	FINAL_K,
	SIMILARITY_THRESHOLD,
)
logger.info(
	"model_config | LLM_MODEL=%s | EMBEDDING_MODEL=%s | RERANKER_MODEL=%s | LANGSMITH_ENABLED=%s",
	LLM_MODEL,
	EMBEDDING_MODEL,
	RERANKER_MODEL,
	LANGSMITH_ENABLED,
)
