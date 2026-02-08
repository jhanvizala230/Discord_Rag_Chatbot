from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import uuid, time
from backend_langchain.core.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend_langchain.vector_and_db.vectorstore import add_chunks, list_docs, get_doc_entries, delete_doc, delete_all_docs
from backend_langchain.vector_and_db.db import register_doc, insert_raw_table, get_document
from backend_langchain.data_extraction.pdf_text_extraction import PyMuPDF4LLMPDFExtractor
from backend_langchain.data_extraction.extractor import extract_and_chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend_langchain.validations.drift import detect_drift_for_new_doc
from backend_langchain.core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize heavy components once
pymupdf4llm_pdf_extractor = PyMuPDF4LLMPDFExtractor(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


@router.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), title: str = None, author: str = None):
    logger.info(f"API INPUT /upload | filename={file.filename}, title={title}, author={author}")
    start_time = time.time()
    doc_id = str(uuid.uuid4())
    content = await file.read()
    file_size = len(content)
    ext = Path(file.filename).suffix.lower()
    file_type = ext.lstrip('.')
    if file_type not in {"pdf", "docx", "txt"}:
        raise HTTPException(415, f"Unsupported file type: {file_type}")
    metadata = {
        'title': title or Path(file.filename).stem,
        'author': author or 'Unknown',
        'file_type': file_type,
        'file_size': file_size,
        'original_filename': file.filename,
        'Category_Document': file_type
    }
    out_path = DOCS_DIR / f"{doc_id}{ext}"
    with open(out_path, "wb") as f:
        f.write(content)

    def process_upload_document():
        try:
            if file_type == "pdf":
                result = pymupdf4llm_pdf_extractor.extract(out_path)
            elif file_type == "docx":
                result = extract_and_chunk(str(out_path), file_type)
            else:
                text = out_path.read_text(encoding="utf-8", errors="ignore")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                text_chunks = [
                    {
                        "text": chunk,
                        "metadata": {
                            "content_type": "text",
                            "page": None,
                            "source": out_path.name,
                        },
                    }
                    for chunk in splitter.split_text(text)
                    if chunk.strip()
                ]
                result = {
                    "text_chunks": text_chunks,
                    "table_chunks": [],
                    "image_chunks": [],
                    "raw_tables": [],
                }
            text_chunks = result.get("text_chunks", [])
            table_chunks = result.get("table_chunks", [])
            image_chunks = result.get("image_chunks", [])
            raw_tables = result.get("raw_tables", [])
            logger.info(f"Extracted chunks | text={len(text_chunks)}, \
                         tables={len(table_chunks)}, images={len(image_chunks)}")
            # Persist raw tables to SQLite with the generated doc_id
            for raw_table in raw_tables:
                try:
                    insert_raw_table(
                        doc_id=doc_id,
                        page=raw_table.get("metadata", {}).get("page"),
                        csv_str=raw_table.get("csv", ""),
                    )
                except Exception as exc:
                    logger.warning("raw_table_insert_failed | doc_id=%s | error=%s", doc_id, exc)

            def _normalize_chunk(chunk, default_type: str):
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if text is None:
                        text = chunk.get("csv", "")
                    metadata = dict(chunk.get("metadata") or {})
                    metadata.setdefault("content_type", default_type)
                    metadata.setdefault("page", metadata.get("page"))
                    if "page" not in metadata:
                        metadata["page"] = None
                    return text or "", metadata
                return str(chunk), {"content_type": default_type}

            chunk_texts = []
            chunk_metas = []
            chunk_types = []
            chunk_ids = []

            for idx, chunk in enumerate(text_chunks):
                text, meta = _normalize_chunk(chunk, "text")
                chunk_id = f"txt_{doc_id}_{idx}"
                chunk_texts.append(text)
                chunk_ids.append(chunk_id)
                chunk_types.append("text")
                chunk_metas.append(
                    {
                        "entity_type": "text",
                        "source_id": doc_id,
                        "page": meta.get("page"),
                        "chunk_index": meta.get("chunk_index", idx),
                        "references": {"tables": [], "images": []},
                        "text": text,
                        "created_at": meta.get("created_at"),
                        "Category_Document": metadata.get("Category_Document"),
                        **{k: v for k, v in meta.items() if k not in {"content_type"}},
                    }
                )

            logger.info(f"Total chunks prepared for storage: {len(chunk_texts)}")
            metas = metadata.copy()
            logger.info(f"Meta prepared for storage: {metas}")          
            metas["chunk_types"] = chunk_types
            add_chunks(doc_id, chunk_texts, metas=metas, chunk_metadata=chunk_metas, ids=chunk_ids)
            
            # Register document in SQLite with metadata
            register_doc(
                doc_id=doc_id,
                filename=metas.get('title', 'Untitled'),
                num_chunks=len(chunk_texts),
                file_type=metadata.get('file_type'),
                file_size=metadata.get('file_size'),
                title=metadata.get('title'),
                author=metadata.get('author')
            )
            
            try:
                drift_result = detect_drift_for_new_doc(chunk_texts)
            except Exception as e:
                logger.warning(f"Drift detection failed: {str(e)}")
            logger.info(f"API OUTPUT /upload | doc_id={doc_id}, chunks={len(chunk_texts)}, metadata={metadata}")
        except Exception as e:
            logger.error(f"Background upload failed: {str(e)}")

    background_tasks.add_task(process_upload_document)
    processing_time = time.time() - start_time
    return {
        "doc_id": doc_id,
        "metadata": metadata,
        "chunks": None,
        "processing_time_seconds": round(processing_time, 2),
        "drift_detection": None,
        "background": True
    }

@router.get("/getdocs")
def get_docs():
    logger.info("API INPUT /getdocs")
    try:
        docs = list_docs()
        logger.info(f"API OUTPUT /getdocs | docs_count={len(docs)}")
        return JSONResponse(content={"status": "success", "docs": docs}, status_code=200)
    except Exception as e:
        logger.error(f"API ERROR /getdocs | {str(e)}")
        return JSONResponse(content={"status": "error", "message": "Failed to fetch documents"}, status_code=500)

@router.get("/getdoc_id/{doc_id}")
def get_doc(doc_id: str):
    logger.info(f"API INPUT /getdoc_id/{doc_id}")
    entries = get_doc_entries(doc_id)
    logger.info(f"API OUTPUT /getdoc_id/{doc_id} | entries_count={len(entries)}")
    if not entries:
        logger.warning(f"API WARNING /getdoc_id/{doc_id} | Document not found")
        raise HTTPException(404, f"Document not found: {doc_id}")
    return {"doc_id": doc_id, "entries": entries}

@router.delete("/doc/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    logger.info(f"Request to delete document: {doc_id}")
    def process_delete():
        try:
            success = delete_doc(doc_id)
            deleted_file = None
            for file in DOCS_DIR.glob(f"{doc_id}.*"):
                try:
                    file.unlink()
                    deleted_file = str(file)
                    logger.info(f"Deleted file from disk: {deleted_file}")
                except Exception as fe:
                    logger.warning(f"Failed to delete file {file}: {fe}")
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
    background_tasks.add_task(process_delete)
    return JSONResponse(content={"status": "success", "message": f"Document {doc_id} deletion started in background"}, status_code=200)

@router.delete("/docs")
def delete_all_documents():
    logger.warning("Request to delete all documents")
    try:
        delete_all_docs()
        deleted_files = []
        for file in DOCS_DIR.glob("*"):
            try:
                file.unlink()
                deleted_files.append(str(file))
                logger.info(f"Deleted file from disk: {file}")
            except Exception as fe:
                logger.warning(f"Failed to delete file {file}: {fe}")
        return JSONResponse(content={"status": "success", "message": "All documents deleted", "deleted_files": deleted_files}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to delete all documents: {str(e)}")
        raise HTTPException(500, f"Error deleting documents: {str(e)}")

@router.get("/drift/{doc_id}")
def check_drift(doc_id: str):
    logger.info(f"API INPUT /drift/{doc_id}")
    from backend_langchain.vector_and_db.vectorstore import get_doc_entries
    try:
        entries = get_doc_entries(doc_id)
        if not entries:
            logger.warning(f"API WARNING /drift/{doc_id} | Document not found")
            raise HTTPException(404, f"Document not found: {doc_id}")
        chunks = [e["text"] for e in entries]
        drift_result = detect_drift_for_new_doc(chunks)
        logger.info(f"API OUTPUT /drift/{doc_id} | drift_result={drift_result}")
        return {"doc_id": doc_id, "drift_detection": drift_result}
    except Exception as e:
        logger.error(f"API ERROR /drift/{doc_id} | {str(e)}")
        raise HTTPException(500, f"Error during drift detection: {str(e)}")


@router.get("/documents/{doc_id}/file")
def download_document_file(doc_id: str):
    """Download the original uploaded document for citation deep links."""
    file_path = next(DOCS_DIR.glob(f"{doc_id}.*"), None)
    if not file_path:
        logger.warning("document_file_not_found | doc_id=%s", doc_id)
        raise HTTPException(404, f"Document not found: {doc_id}")

    doc_meta = None
    try:
        doc_meta = get_document(doc_id)
    except Exception as exc:
        logger.warning("document_metadata_lookup_failed | doc_id=%s | error=%s", doc_id, exc)

    download_name = (doc_meta or {}).get("filename") or file_path.name
    return FileResponse(
        path=file_path,
        filename=download_name,
        media_type="application/octet-stream",
    )