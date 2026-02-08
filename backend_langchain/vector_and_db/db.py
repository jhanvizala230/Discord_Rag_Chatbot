"""SQLite database operations for DocIntel."""

from typing import Optional

from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    MetaData,
    create_engine,
    select,
    func,
    Index,
    update,
)
from datetime import datetime
from langchain_community.chat_message_histories import SQLChatMessageHistory

from ..core.config import SQLITE_DB
from ..core.logger import setup_logger

logger = setup_logger(__name__)

engine = create_engine(f"sqlite:///{SQLITE_DB}", connect_args={"check_same_thread": False})
metadata = MetaData()


# ============================================================================
# TABLE 1: CONVERSATIONS (message_store - managed by LangChain)
# ============================================================================

def ensure_chat_history_table() -> None:
    """
    Ensure LangChain's message_store table exists.
    Table schema managed by LangChain SQLChatMessageHistory.
    """
    try:
        SQLChatMessageHistory(
            session_id="__bootstrap__",
            connection_string=f"sqlite:///{SQLITE_DB}",
        )
        logger.debug("chat_history_table_ready")
    except Exception as exc:
        logger.error("chat_history_table_init_failed | error=%s", exc)


# ============================================================================
# TABLE 2: DOCUMENTS (documents + metadata)
# ============================================================================

documents_table = Table(
    "documents", metadata,
    Column("id", Integer, primary_key=True),
    Column("doc_id", String, unique=True, index=True, nullable=False),
    Column("filename", String, nullable=False),
    Column("file_type", String),
    Column("file_size", Integer),
    Column("title", String),
    Column("author", String),
    Column("num_chunks", Integer, default=0),
    Column("metadata", Text),  # JSON string for additional metadata
    Column("created_at", DateTime, default=func.now()),
)
Index("ix_documents_created", documents_table.c.created_at)


# ============================================================================
# TABLE 2b: DOCUMENT PROCESSING STATUS
# ============================================================================

document_status_table = Table(
    "document_status",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("doc_id", String, unique=True, index=True, nullable=False),
    Column("status", String, nullable=False),
    Column("detail", Text),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
)
Index("ix_document_status_doc", document_status_table.c.doc_id)


# ============================================================================
# TABLE 3: RAW TABLES (for PDF table extraction)
# ============================================================================

raw_tables_table = Table(
    "raw_tables", metadata,
    Column("id", Integer, primary_key=True),
    Column("doc_id", String, index=True),
    Column("page", Integer),
    Column("csv", Text),
    Column("created_at", DateTime, default=func.now())
)

# Create all tables
metadata.create_all(engine)


# ============================================================================
# DOCUMENT OPERATIONS
# ============================================================================

def register_doc(doc_id: str, filename: str, num_chunks: int, 
                 file_type: str = None, file_size: int = None,
                 title: str = None, author: str = None,
                 metadata_json: str = None) -> None:
    """
    Register a new document in the database.
    
    Args:
        doc_id: Unique document identifier
        filename: Original filename
        num_chunks: Number of text chunks extracted
        file_type: Document type (pdf, docx, txt)
        file_size: File size in bytes
        title: Document title
        author: Document author
        metadata_json: Additional metadata as JSON string
    """
    logger.info("registering_doc | doc_id=%s | filename=%s | chunks=%d", doc_id, filename, num_chunks)
    try:
        with engine.connect() as conn:
            conn.execute(documents_table.insert().values(
                doc_id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                title=title or filename,
                author=author or "Unknown",
                num_chunks=num_chunks,
                metadata=metadata_json
            ))
            conn.commit()
        logger.debug("doc_registered | doc_id=%s", doc_id)
    except Exception as exc:
        logger.error("register_doc_failed | doc_id=%s | error=%s", doc_id, exc)
        raise


def set_document_status(doc_id: str, status: str, detail: Optional[str] = None) -> None:
    """Create or update document processing status."""
    now = datetime.utcnow()
    payload = {
        "doc_id": doc_id,
        "status": status,
        "detail": detail,
        "updated_at": now,
    }
    with engine.begin() as conn:
        existing = conn.execute(
            select(document_status_table.c.doc_id).where(document_status_table.c.doc_id == doc_id)
        ).first()
        if existing:
            conn.execute(
                update(document_status_table)
                .where(document_status_table.c.doc_id == doc_id)
                .values(status=status, detail=detail, updated_at=now)
            )
        else:
            conn.execute(document_status_table.insert().values(payload))
    logger.debug("document_status_set | doc_id=%s | status=%s", doc_id, status)


def get_document_status(doc_id: str) -> Optional[dict]:
    """Fetch processing status for a document."""
    with engine.connect() as conn:
        row = conn.execute(
            select(document_status_table).where(document_status_table.c.doc_id == doc_id)
        ).mappings().first()
        if not row:
            return None
        result = dict(row)
        if result.get("updated_at"):
            result["updated_at"] = result["updated_at"].isoformat()
        return result


def get_document(doc_id: str) -> dict:
    """
    Get document metadata by ID.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Dictionary with document metadata or None if not found
    """
    logger.info("fetching_document | doc_id=%s", doc_id)
    with engine.connect() as conn:
        stmt = select(documents_table).where(documents_table.c.doc_id == doc_id)
        row = conn.execute(stmt).mappings().first()
        if not row:
            logger.warning("document_not_found | doc_id=%s", doc_id)
            return None
        
        doc = dict(row)
        if doc.get("created_at"):
            doc["created_at"] = doc["created_at"].isoformat()
        logger.debug("document_retrieved | doc_id=%s", doc_id)
        return doc


def list_documents() -> list:
    """
    List all documents.
    
    Returns:
        List of document metadata dictionaries
    """
    logger.info("listing_documents")
    with engine.connect() as conn:
        stmt = select(documents_table).order_by(documents_table.c.created_at.desc())
        rows = conn.execute(stmt).mappings().all()
        
        docs = []
        for row in rows:
            doc = dict(row)
            if doc.get("created_at"):
                doc["created_at"] = doc["created_at"].isoformat()
            docs.append(doc)
        
        logger.debug("documents_listed | count=%d", len(docs))
        return docs


def delete_document(doc_id: str) -> bool:
    """
    Delete document from database.
    Note: This does NOT delete from vector store - use vectorstore.delete_doc() for that.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        True if deleted, False if not found
    """
    logger.info("deleting_document | doc_id=%s", doc_id)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                documents_table.delete().where(documents_table.c.doc_id == doc_id)
            )
            conn.commit()
            deleted = result.rowcount > 0
            if deleted:
                logger.info("document_deleted | doc_id=%s", doc_id)
            else:
                logger.warning("document_not_found_for_deletion | doc_id=%s", doc_id)
            return deleted
    except Exception as exc:
        logger.error("delete_document_failed | doc_id=%s | error=%s", doc_id, exc)
        raise


def delete_all_documents() -> int:
    """
    Delete all documents from database.
    Note: This does NOT delete from vector store.
    
    Returns:
        Number of documents deleted
    """
    logger.warning("deleting_all_documents")
    try:
        with engine.connect() as conn:
            result = conn.execute(documents_table.delete())
            conn.commit()
            count = result.rowcount
            logger.info("all_documents_deleted | count=%d", count)
            return count
    except Exception as exc:
        logger.error("delete_all_documents_failed | error=%s", exc)
        raise


# ============================================================================
# RAW TABLE OPERATIONS (for PDF table extraction)
# ============================================================================

def insert_raw_table(doc_id: str, page: int, csv_str: str) -> None:
    """
    Store raw table data extracted from PDF.
    
    Args:
        doc_id: Document identifier
        page: Page number where table was found
        csv_str: Table data as CSV string
    """
    logger.info("inserting_raw_table | doc_id=%s | page=%s", doc_id, page)
    with engine.connect() as conn:
        conn.execute(raw_tables_table.insert().values(
            doc_id=doc_id,
            page=page,
            csv=csv_str
        ))
        conn.commit()
    logger.debug("raw_table_inserted | doc_id=%s | page=%s", doc_id, page)


def fetch_raw_table(doc_id: str, page: int) -> str:
    """
    Retrieve raw table CSV string.
    
    Args:
        doc_id: Document identifier
        page: Page number
        
    Returns:
        CSV string or empty string if not found
    """
    logger.info("fetching_raw_table | doc_id=%s | page=%d", doc_id, page)
    with engine.connect() as conn:
        stmt = select(raw_tables_table.c.csv).where(
            (raw_tables_table.c.doc_id == doc_id) & (raw_tables_table.c.page == page)
        )
        result = conn.execute(stmt).fetchone()
    return result[0] if result else ""


# ============================================================================
# LEGACY COMPATIBILITY (for existing code)
# ============================================================================

# Aliases for backward compatibility
def fetch_raw_table_db(doc_id: str, page: int) -> str:
    """Legacy alias for fetch_raw_table."""
    return fetch_raw_table(doc_id, page)
