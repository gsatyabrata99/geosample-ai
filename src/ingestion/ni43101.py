"""
NI 43-101 technical report ingestion.

Extracts text from PDF drill reports using PyMuPDF with Tesseract OCR
fallback for scanned pages. Saves cleaned text for downstream NLP.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum characters to consider a page as having extractable text
MIN_TEXT_CHARS = 100

# Geological terms to flag as high-value pages
GEO_KEYWORDS = [
    "copper", "grade", "drill", "intercept", "assay", "mineralisation",
    "mineralization", "ore", "deposit", "lithology", "alteration",
    "porphyry", "borehole", "collar", "azimuth", "dip",
]


def extract_text_from_pdf(pdf_path: str | Path, use_ocr: bool = True) -> list[dict]:
    """
    Extract text from a PDF, page by page.
    Falls back to Tesseract OCR for scanned pages.

    Args:
        pdf_path: Path to the PDF file.
        use_ocr: Whether to OCR pages with insufficient digital text.

    Returns:
        List of dicts with keys: page_num, text, has_geo_content, method.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Extracting text from {pdf_path.name} ...")
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        method = "digital"

        # Fall back to OCR if page has little/no digital text
        if use_ocr and len(text) < MIN_TEXT_CHARS:
            text = _ocr_page(page)
            method = "ocr"

        if not text:
            continue

        has_geo = any(kw in text.lower() for kw in GEO_KEYWORDS)
        pages.append({
            "page_num": page_num,
            "text": _clean_text(text),
            "has_geo_content": has_geo,
            "method": method,
        })

    doc.close()
    logger.info(
        f"  → {len(pages)} pages extracted "
        f"({sum(p['has_geo_content'] for p in pages)} with geological content)"
    )
    return pages


def _ocr_page(page) -> str:
    """Render page to image and run Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.warning("pytesseract/Pillow not installed — skipping OCR for this page")
        return ""

    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang="eng")


def _clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, remove control chars."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def process_report_directory(
    pdf_dir: str | Path,
    output_dir: str | Path,
    use_ocr: bool = True,
) -> list[Path]:
    """
    Process all PDFs in a directory, saving extracted text to .txt files.

    Args:
        pdf_dir: Directory containing NI 43-101 PDF files.
        output_dir: Where to save extracted .txt files.
        use_ocr: Whether to use OCR fallback.

    Returns:
        List of paths to saved text files.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    logger.info(f"Processing {len(pdfs)} PDF reports ...")
    saved = []

    for pdf_path in pdfs:
        try:
            pages = extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
            # Only keep pages with geological content
            geo_pages = [p for p in pages if p["has_geo_content"]]
            full_text = "\n\n".join(p["text"] for p in geo_pages)

            out_path = output_dir / f"{pdf_path.stem}.txt"
            out_path.write_text(full_text, encoding="utf-8")
            saved.append(out_path)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Saved {len(saved)} text files to {output_dir}")
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_report_directory(
        pdf_dir="data/raw/ni43101_reports",
        output_dir="data/processed/text",
    )
