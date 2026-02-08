"""PyMuPDF4LLM-powered PDF extractor integrated with LangChain."""

import os
import re
from typing import Dict, List, Tuple

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend_langchain.core.logger import setup_logger
from backend_langchain.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger(__name__)


class PyMuPDF4LLMPDFExtractor:
    """Extract PDFs using PyMuPDF4LLM and produce LangChain-style chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        include_tables: bool = True,
        include_images: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_tables = include_tables
        self.include_images = include_images
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._table_extractor = None
        self._image_extractor = None

    @staticmethod
    def _detect_markdown_tables(text: str) -> List[str]:
        lines = text.splitlines()
        tables: List[str] = []
        i = 0
        while i < len(lines) - 1:
            header = lines[i]
            separator = lines[i + 1]
            if "|" in header and re.match(r"^\s*\|?\s*[:-]+", separator):
                start = i
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    i += 1
                tables.append("\n".join(lines[start:i]).strip())
            else:
                i += 1
        return [t for t in tables if t]

    @staticmethod
    def _detect_markdown_images(text: str) -> List[Tuple[str, str]]:
        images: List[Tuple[str, str]] = []
        for match in re.finditer(r"!\[(.*?)\]\((.*?)\)", text):
            alt_text = match.group(1).strip()
            path = match.group(2).strip()
            images.append((alt_text, path))
        return images

    def extract(self, path: str) -> Dict[str, List[Dict]]:
        """Return chunk dictionary compatible with the ingestion pipeline."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("PyMuPDF4LLMPDFExtractor: loading %s", path)
        loader = PyMuPDF4LLMLoader(file_path=path)
        base_docs: List[LCDocument] = loader.load()
        if not base_docs:
            raise ValueError(f"PyMuPDF4LLM loader returned no documents for {path}")

        chunk_docs = self.splitter.split_documents(base_docs)
        text_chunks: List[Dict] = []
        plain_texts: List[str] = []
        table_chunks: List[Dict] = []
        raw_tables: List[Dict] = []
        image_chunks: List[Dict] = []
        source_name = os.path.basename(path)

        for idx, chunk in enumerate(chunk_docs):
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("source", source_name)
            content_type = metadata.get("content_type") or metadata.get("type")
            if content_type:
                content_type = str(content_type).lower()
            page = metadata.get("page") or metadata.get("page_number")

            if self.include_tables and content_type == "table":
                table_text = chunk.page_content.strip()
                if table_text:
                    raw_tables.append(
                        {
                            "csv": table_text,
                            "metadata": {
                                "content_type": "table_raw",
                                "page": page,
                                "source": source_name,
                            },
                        }
                    )
                    table_chunks.append(
                        {
                            "text": f"Table on page {page}: {table_text[:200]}",
                            "metadata": {
                                "content_type": "table_summary",
                                "page": page,
                                "source": source_name,
                                "raw_table_csv": table_text,
                            },
                        }
                    )
                    text_chunks.append(
                        {
                            "text": f"Table on page {page}: {table_text}",
                            "metadata": {
                                "content_type": "table_summary",
                                "page": page,
                                "source": source_name,
                                "chunk_index": idx,
                            },
                        }
                    )
                continue

            if self.include_images and content_type == "image":
                image_text = chunk.page_content or f"Image on page {page}"
                image_chunk = {
                    "text": image_text,
                    "metadata": {
                        "content_type": "image_caption",
                        "page": page,
                        "source": source_name,
                        "chunk_index": idx,
                    },
                }
                image_chunks.append(image_chunk)
                text_chunks.append(image_chunk)
                continue

            plain_texts.append(chunk.page_content)
            text_chunks.append(
                {
                    "text": chunk.page_content,
                    "metadata": {
                        **metadata,
                        "content_type": metadata.get("content_type", "text"),
                        "chunk_index": idx,
                    },
                }
            )

            if self.include_tables:
                for table_text in self._detect_markdown_tables(chunk.page_content):
                    raw_tables.append(
                        {
                            "csv": table_text,
                            "metadata": {
                                "content_type": "table_raw",
                                "page": page,
                                "source": source_name,
                            },
                        }
                    )
                    table_chunks.append(
                        {
                            "text": f"Table on page {page}: {table_text[:200]}",
                            "metadata": {
                                "content_type": "table_summary",
                                "page": page,
                                "source": source_name,
                                "raw_table_csv": table_text,
                            },
                        }
                    )

            if self.include_images:
                for alt_text, image_path in self._detect_markdown_images(chunk.page_content):
                    image_chunks.append(
                        {
                            "text": alt_text or f"Image on page {page}",
                            "metadata": {
                                "content_type": "image",
                                "page": page,
                                "source": source_name,
                                "image_path": image_path,
                            },
                        }
                    )

        logger.info(
            "PyMuPDF4LLMPDFExtractor: produced %s text chunks for %s",
            len(text_chunks),
            source_name,
        )

        return {
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "image_chunks": image_chunks,
            "raw_tables": raw_tables,
            "texts": plain_texts,
        }
