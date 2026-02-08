# Retrieve relevant document chunks from vector DB
# from .embeddings import embed_texts

import os
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from ..core.config import (
    DB_DIR,
    TOP_K,
    FINAL_K,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_DISTANCE_THRESHOLD,
)
from ..core.logger import setup_logger
from .embeddings import get_langchain_embeddings

DB_PATH = str(DB_DIR)
os.makedirs(DB_PATH, exist_ok=True)
TEXT_COLLECTION = "text_collection"
TABLE_COLLECTION = "table_collection"
IMAGE_COLLECTION = "image_collection"
logger = setup_logger(__name__)

_VECTORSTORES: dict[str, Chroma] = {}


def get_vectorstore(collection_name: str) -> Chroma:
    """Get or create the LangChain Chroma vector store."""
    if collection_name not in _VECTORSTORES:
        embeddings = get_langchain_embeddings()
        _VECTORSTORES[collection_name] = Chroma(
            collection_name=collection_name,
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )
        logger.info("LangChain Chroma initialized | collection=%s", collection_name)
    return _VECTORSTORES[collection_name]


def ensure_collections() -> None:
    """Ensure text/table/image collections exist (table/image are schema-only)."""
    for name in (TEXT_COLLECTION, TABLE_COLLECTION, IMAGE_COLLECTION):
        get_vectorstore(name)

def get_collection():
    """Get the underlying Chroma collection for text storage."""
    collection = get_vectorstore(TEXT_COLLECTION)._collection
    count = collection.count()
    logger.info("Accessed collection '%s' at '%s', doc_count=%s", TEXT_COLLECTION, DB_PATH, count)
    return collection

def add_chunks(doc_id, chunks, embeddings=None, metas=None, chunk_metadata=None, ids=None):
    """Add document chunks to the vector store"""
    logger.info(f"Adding {len(chunks)} chunks for document {doc_id}")

    metas = metas or {}
    
    # Add basic document metadata to all chunks
    base_meta = {
        'doc_id': doc_id,
        'title': metas.get('title', 'Untitled'),
        'author': metas.get('author', 'Unknown'),
        'file_type': metas.get('file_type', 'unknown'),
        'file_size': metas.get('file_size', 0),
        'uploaded_at': datetime.now().isoformat(),
        'total_chunks': len(chunks),
        'avg_chunk_length': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
        'Category_Document': metas.get('Category_Document')
    }
    
    # Determine chunk types if provided in metas, else default to 'text'
    chunk_types = metas.get('chunk_types', ['text'] * len(chunks))
    ids = ids or [f"txt_{doc_id}_{i}" for i in range(len(chunks))]
    chunk_metas = []
    for i in range(len(chunks)):
        meta = base_meta.copy()
        if chunk_metadata and i < len(chunk_metadata) and isinstance(chunk_metadata[i], dict):
            meta.update(chunk_metadata[i])
        meta.update({
            'chunk_id': i,
            'chunk_size': len(chunks[i]),
            'chunk_type': chunk_types[i] if i < len(chunk_types) else 'text'
        })
        meta.setdefault("entity_type", "text")
        meta.setdefault("source_id", doc_id)
        meta.setdefault("references", {"tables": [], "images": []})
        meta.setdefault("created_at", datetime.now().isoformat())
        meta.setdefault("text", chunks[i])
        chunk_metas.append(meta)
    
    try:
        vectorstore = get_vectorstore(TEXT_COLLECTION)
        sanitized_metas = []
        for meta in chunk_metas:
            cleaned = {}
            for key, value in meta.items():
                if isinstance(value, (dict, list)):
                    cleaned[key] = json.dumps(value)
                else:
                    cleaned[key] = value
            sanitized_metas.append(cleaned)
        documents = [Document(page_content=chunks[i], metadata=sanitized_metas[i]) for i in range(len(chunks))]
        documents = filter_complex_metadata(documents)
        vectorstore.add_documents(documents, ids=ids)
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
        logger.debug(f"Successfully added {len(chunks)} chunks for document {doc_id}")
    except Exception as e:
        logger.error(f"Failed to add chunks for document {doc_id}: {e}")
        raise

def search_docs(query=None, filters=None):
    """Search documents by metadata filters"""
    logger.info(f"Searching documents with filters: {filters}")
    
    try:
        c = get_collection()
        
        # Log collection count for debugging
        count = c.count()
        logger.debug(f"Collection stats: total_items={count}")
        
        if count == 0:
            logger.info("Empty collection")
            return []
            
        # For no filters, just get all documents without a where clause
        if not filters:
            logger.debug("Fetching all documents")
            results = c.get(include=['metadatas', 'documents'])
        else:
            # Build where clause only when filters are provided
            where = None
            if 'doc_id' in filters:
                where = {"doc_id": {"$eq": filters['doc_id']}}
            elif 'title' in filters:
                where = {"title": {"$eq": filters['title']}}
            elif 'author' in filters:
                where = {"author": {"$eq": filters['author']}}
            
            logger.debug(f"Executing filtered search: where_clause={where}")
            results = c.get(
                where=where,
                include=['metadatas', 'documents']
            )
        
        # Group by doc_id and extract unique document info
        docs = {}
        metadatas = results.get('metadatas', [])
        
        # Debug the structure of returned metadatas
        logger.debug("processing_metadatas", 
                 metadatas_type=str(type(metadatas)),
                 first_item_type=str(type(metadatas[0])) if metadatas else "empty")
        
        # Handle both nested and flat metadata structures
        if metadatas:
            if isinstance(metadatas[0], list):
                # Nested structure
                for metadata_list in metadatas:
                    for meta in metadata_list:
                        if not isinstance(meta, dict):
                            continue
                        doc_id = meta.get('doc_id')
                        if doc_id and doc_id not in docs:
                            docs[doc_id] = {
                                'doc_id': doc_id,
                                'title': meta.get('title', 'Untitled'),
                                'author': meta.get('author', 'Unknown'),
                                'file_type': meta.get('file_type', 'unknown'),
                                'file_size': meta.get('file_size', 0),
                                'uploaded_at': meta.get('uploaded_at', ''),
                                'total_chunks': meta.get('total_chunks', 0)
                            }
            else:
                # Flat structure
                for meta in metadatas:
                    if not isinstance(meta, dict):
                        continue
                    doc_id = meta.get('doc_id')
                    if doc_id and doc_id not in docs:
                        docs[doc_id] = {
                            'doc_id': doc_id,
                            'title': meta.get('title', 'Untitled'),
                            'author': meta.get('author', 'Unknown'),
                            'file_type': meta.get('file_type', 'unknown'),
                            'file_size': meta.get('file_size', 0),
                            'uploaded_at': meta.get('uploaded_at', ''),
                            'total_chunks': meta.get('total_chunks', 0)
                        }
        
        logger.debug(f"Documents found: total_documents={len(docs)}, document_ids={list(docs.keys())}")
        return list(docs.values())
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return []

def list_docs():
    """List all documents with metadata"""
    logger.info("Listing all documents")
    try:
        docs = search_docs()
        logger.info(f"Documents listed: count={len(docs)}")
        return docs
    except Exception as e:
        logger.error(f"Listing documents failed: {e}")
        return []

def get_doc_entries(doc_id):
    """Get document chunks with metadata"""
    logger.info(f"Fetching document entries for doc_id={doc_id}")
    
    try:
        c = get_collection()
        results = c.get(
            where={'doc_id': doc_id},
            include=['ids', 'documents', 'metadatas']
        )
        
        entries = []
        for _id, doc, meta in zip(
            results.get('ids', []),
            results.get('documents', []),
            results.get('metadatas', [])
        ):
            entries.append({
                'id': _id,
                'text': doc,
                'meta': meta
            })
        
        entries = sorted(entries, key=lambda x: x['meta'].get('chunk_id', 0))
        logger.debug(f"Document entries retrieved for doc_id={doc_id}, num_chunks={len(entries)}")
        return entries
    except Exception as e:
        logger.error(f"Document entries fetch failed for doc_id={doc_id}: {e}")
        return []

def query_top_k(q_emb, top_k: int = TOP_K):
    """Query top k similar documents"""
    logger.info(f"Querying top {top_k} similar documents")
    c = get_collection()
    try:
        res = c.query(
            query_embeddings=[q_emb.tolist()], 
            n_results=top_k, 
            include=['documents', 'metadatas', 'distances']
        )
        docs = res.get('documents', [[]])[0]
        metas = res.get('metadatas', [[]])[0]
        dists = res.get('distances', [[]])[0]
        
        results = []
        for t, m, d in zip(docs, metas, dists):
            distance = float(d)
            similarity = None
            if 0.0 <= distance <= 1.0:
                similarity = 1.0 - distance
            meets_distance = distance <= RETRIEVAL_DISTANCE_THRESHOLD
            meets_similarity = similarity is not None and similarity >= SIMILARITY_THRESHOLD
            if meets_distance or meets_similarity:
                results.append({"text": t, "meta": m, "distance": distance})
        if not results:
            logger.debug(
                "query_top_k_filtered | requested=%s | filtered_out=%s",
                len(docs),
                len(docs),
            )
        else:
            logger.debug("Found %d matching documents", len(results))
        return results
    except Exception as e:
        logger.error(f"Similarity query failed: {e}")
        return []

def delete_doc(doc_id: str):
    """Delete a specific document and its chunks"""
    logger.info(f"Deleting document doc_id={doc_id}")
    try:
        deleted = 0
        for name in (TEXT_COLLECTION, TABLE_COLLECTION, IMAGE_COLLECTION):
            c = get_vectorstore(name)._collection
            results = c.get(where={'doc_id': doc_id})
            ids = results.get('ids', [])
            if ids:
                c.delete(ids=ids)
                deleted += len(ids)
        if deleted:
            logger.info(f"Document deleted doc_id={doc_id}, chunks_deleted={deleted}")
            return True
        logger.warning(f"Document not found doc_id={doc_id}")
        return False
    except Exception as e:
        logger.error(f"Document deletion failed doc_id={doc_id}: {e}")
        raise

def delete_all_docs():
    """Delete all documents from collection"""
    logger.warning("Deleting all documents from collection")
    try:
        vectorstore = get_vectorstore(TEXT_COLLECTION)
        client = vectorstore._client
        for name in (TEXT_COLLECTION, TABLE_COLLECTION, IMAGE_COLLECTION):
            try:
                client.delete_collection(name)
            except Exception:
                logger.warning("collection_delete_skipped | collection=%s", name)
        _VECTORSTORES.clear()
        ensure_collections()
        logger.info("All documents deleted")
        return True
    except Exception as e:
        logger.error(f"Delete all failed: {e}")
        raise

def retrieve_doc_chunks(query_emb, top_k: int = FINAL_K) -> list:
    if query_emb is None:
        raise ValueError("query_emb is required for retrieval")
    results = query_top_k(query_emb, top_k=top_k)
    return [{"text": r["text"], "meta": r["meta"]} for r in results]


ensure_collections()
