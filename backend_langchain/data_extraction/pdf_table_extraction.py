"""Dedicated PDF table extraction utilities built on Camelot."""

import os
from typing import Dict, List, Sequence
import camelot
import pandas as pd

from backend_langchain.core.logger import setup_logger
from backend_langchain.vector_and_db.table_file_store import (
    store_table_as_csv,
    store_table_as_json,
)

logger = setup_logger(__name__)


class PDFTableExtractor:
    """Extract tabular content from PDFs and persist artifacts."""

    def __init__(
        self,
        camelot_flavors: Sequence[str] | None = None,
        preview_rows: int = 3,
    ) -> None:
        self.camelot_flavors = tuple(camelot_flavors or ("stream", "lattice"))
        self.preview_rows = preview_rows

    def extract(self, path: str) -> Dict[str, List[Dict]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("PDFTableExtractor: reading tables from %s", path)
        tables = self._read_with_camelot(path)
        source_name = os.path.basename(path)

        table_chunks: List[Dict] = []
        raw_tables: List[Dict] = []

        for idx, table in enumerate(tables):
            page_num = self._safe_page(table.page) or idx
            df = table.df.applymap(lambda cell: str(cell).replace("\n", " ").strip())
            table_rows = df.values.tolist()

            store_table_as_csv(source_name, page_num, table_rows)
            store_table_as_json(source_name, page_num, table_rows)

            csv_str = df.to_csv(index=False, header=False)
            raw_tables.append(
                {
                    "csv": csv_str,
                    "metadata": {
                        "content_type": "table_raw",
                        "page": page_num,
                        "source": source_name,
                    },
                }
            )

            summary = self._build_preview(table_rows, page_num)
            table_chunks.append(
                {
                    "text": summary,
                    "metadata": {
                        "content_type": "table_summary",
                        "page": page_num,
                        "source": source_name,
                        "raw_table_csv": csv_str,
                    },
                }
            )

        logger.info(
            "PDFTableExtractor: captured %s tables from %s",
            len(table_chunks),
            source_name,
        )

        return {
            "table_chunks": table_chunks,
            "raw_tables": raw_tables,
        }

    def _read_with_camelot(self, path: str):
        tables = []
        for flavor in self.camelot_flavors:
            try:
                result = camelot.read_pdf(
                    path,
                    pages="all",
                    flavor=flavor,
                    suppress_stdout=True,
                )
            except Exception as exc:
                logger.warning("Camelot (%s) failed for %s: %s", flavor, path, exc)
                continue
            if result:
                tables.extend(result)
        return tables

    @staticmethod
    def _safe_page(page_value):
        try:
            return int(page_value)
        except (TypeError, ValueError):
            return None

    def _build_preview(self, rows: List[List[str]], page_num: int) -> str:
        preview_lines = []
        for row in rows[: self.preview_rows]:
            line = " | ".join(cell.strip() for cell in row if cell)
            if line:
                preview_lines.append(line)
        body = preview_lines[0] if preview_lines else "Table extracted"
        return f"Table on page {page_num}: {body[:200]}"
