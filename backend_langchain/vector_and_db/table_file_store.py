import os
import json
import csv

from backend_langchain.core.config import TABLE_JSON_DIR, TABLE_CSV_DIR
from backend_langchain.core.logger import setup_logger

logger = setup_logger(__name__)


def _build_filename(doc_id, page, ext):
    safe_doc_id = doc_id.replace(os.sep, "_")
    return f"{safe_doc_id}_page{page}.{ext}"


def _extract_page(name: str):
    if "_page" not in name:
        return None
    try:
        page_part = name.rsplit("_page", 1)[1]
        page_str = page_part.split(".")[0]
        return int(page_str)
    except (IndexError, ValueError):
        return None

def store_table_as_json(doc_id, page, table_data):
    """Store table data as a JSON file."""
    file_path = TABLE_JSON_DIR / _build_filename(doc_id, page, "json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(table_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Stored table as JSON: {file_path}")
    return str(file_path)

def store_table_as_csv(doc_id, page, table_rows):
    """Store table data as a CSV file."""
    file_path = TABLE_CSV_DIR / _build_filename(doc_id, page, "csv")
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in table_rows:
            writer.writerow(row)
    logger.info(f"Stored table as CSV: {file_path}")
    return str(file_path)

def load_table_json(doc_id, page):
    file_path = TABLE_JSON_DIR / _build_filename(doc_id, page, "json")
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_table_csv(doc_id, page):
    file_path = TABLE_CSV_DIR / _build_filename(doc_id, page, "csv")
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def list_table_pages(doc_id):
    """Return sorted page numbers for which we have table artifacts."""
    pages = set()
    prefix = f"{doc_id.replace(os.sep, '_')}_page"
    for folder, ext in ((TABLE_JSON_DIR, "json"), (TABLE_CSV_DIR, "csv")):
        for path in folder.glob(f"{prefix}*.{ext}"):
            page = _extract_page(path.name)
            if page is not None:
                pages.add(page)
    return sorted(pages)


def load_table_data(doc_id, page, prefer_format="json"):
    """Load table data honoring a preferred format and falling back as needed."""
    ordered_formats = [prefer_format.lower()] if prefer_format else []
    ordered_formats += [fmt for fmt in ["json", "csv"] if fmt not in ordered_formats]

    for fmt in ordered_formats:
        if fmt == "json":
            payload = load_table_json(doc_id, page)
        elif fmt == "csv":
            payload = load_table_csv(doc_id, page)
        else:
            continue
        if payload is not None:
            logger.debug(
                "loaded_table_data | doc_id=%s | page=%s | format=%s",
                doc_id,
                page,
                fmt,
            )
            return {"format": fmt, "data": payload}

    logger.warning(f"table_data_missing | doc_id={doc_id} | page={page}")
    return None
