"""PDF image extraction helpers relying on PyMuPDF (fitz)."""

import os
import tempfile
from typing import Dict, List, Optional
import fitz
from backend_langchain.core.logger import setup_logger

logger = setup_logger(__name__)


class PDFImageExtractor:
    """Extracts significant images from PDF pages."""

    def __init__(self, min_dimension: int = 300, max_images: Optional[int] = None) -> None:
        self.min_dimension = min_dimension
        self.max_images = max_images

    def extract(self, path: str) -> Dict[str, List[Dict]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("PDFImageExtractor: scanning images in %s", path)
        image_chunks: List[Dict] = []
        stored_images: List[str] = []
        extracted = 0

        with fitz.open(path) as pdf_doc:
            for page_index, page in enumerate(pdf_doc, start=1):
                for image_info in page.get_images(full=True):
                    if self.max_images is not None and extracted >= self.max_images:
                        break
                    xref = image_info[0]
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.width < self.min_dimension or pix.height < self.min_dimension:
                        pix = None
                        continue
                    image_path = self._persist_pixmap(pix)
                    pix = None
                    stored_images.append(image_path)
                    extracted += 1

                    image_chunks.append(
                        {
                            "text": f"Image on page {page_index}",
                            "metadata": {
                                "content_type": "image",
                                "page": page_index,
                                "image_path": image_path,
                                "width": image_info[2],
                                "height": image_info[3],
                            },
                        }
                    )

        logger.info(
            "PDFImageExtractor: stored %s qualifying images from %s",
            len(image_chunks),
            path,
        )

        return {
            "image_chunks": image_chunks,
            "stored_images": stored_images,
        }

    def _persist_pixmap(self, pix: fitz.Pixmap) -> str:
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pix_to_save = pix
            if pix.n >= 5:
                pix_to_save = fitz.Pixmap(fitz.csRGB, pix)
            pix_to_save.save(tmp_path)
        finally:
            if pix_to_save is not pix:
                pix_to_save = None
        return tmp_path
