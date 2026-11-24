"""
Shared document text extraction for Mao teacher chat and student ask_doc.
Supports pdf, docx, doc, txt, pptx (case-insensitive extensions).
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Tuple

from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 15000

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".pptx",
    ".csv",
}


def normalize_extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower()


def is_supported_filename(filename: Optional[str]) -> bool:
    return normalize_extension(filename) in SUPPORTED_EXTENSIONS


def _extract_pdf_text(path: str) -> str:
    """Try PyMuPDF → pypdf → langchain PyMuPDFLoader."""
    # 1) pymupdf (fitz)
    try:
        import fitz  # type: ignore

        parts = []
        with fitz.open(path) as doc:
            for page in doc:
                parts.append(page.get_text("text") or "")
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PDF extract via pymupdf failed: %s", e)

    # 2) pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PDF extract via pypdf failed: %s", e)

    # 3) langchain loader (last resort)
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        docs = PyMuPDFLoader(path).load()
        text = "\n".join((d.page_content or "") for d in docs).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PDF extract via PyMuPDFLoader failed: %s", e)

    raise ValueError(
        "No pude extraer texto del PDF. Prueba otro archivo o un DOCX/TXT."
    )


def _loader_for_path(path: str, ext: str):
    if ext in (".docx", ".doc"):
        return UnstructuredWordDocumentLoader(path)
    if ext == ".txt":
        return TextLoader(path, encoding="utf-8")
    if ext == ".pptx":
        return UnstructuredPowerPointLoader(path)
    if ext == ".csv":
        return CSVLoader(path)
    return UnstructuredFileLoader(path)


def extract_text_from_path(
    path: str,
    filename: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    ext = normalize_extension(filename or path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Formato no soportado. Usa PDF, DOC, DOCX, TXT o PPTX."
        )
    if ext == ".pdf":
        text = _extract_pdf_text(path)
    else:
        loader = _loader_for_path(path, ext)
        docs = loader.load()
        text = "\n".join((d.page_content or "") for d in docs).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


async def extract_text_from_upload(
    file_bytes: bytes,
    filename: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Tuple[str, str]:
    """
    Write upload to a temp file, extract text, clean up.
    Returns (text, original_filename).
    """
    ext = normalize_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Formato no soportado. Usa PDF, DOC, DOCX, TXT o PPTX."
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = extract_text_from_path(tmp_path, filename=filename, max_chars=max_chars)
        return text, filename
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning("Could not remove temp doc %s: %s", tmp_path, e)
