"""Generate a downloadable PDF for Mao support materials."""
from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Any, Dict, Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"


def _safe_filename(name: str, fallback: str = "Material_de_apoyo") -> str:
    raw = (name or "").strip()
    raw = re.sub(r"\.(pdf|docx?|pptx?|txt)$", "", raw, flags=re.IGNORECASE)
    base = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ\-]+", "", raw, flags=re.UNICODE)
    base = re.sub(r"\s+", "_", base.strip())[:60] or fallback
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base


def _strip_md_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()


def markdown_to_pdf_bytes(
    markdown_text: str,
    *,
    title: str = "Material de apoyo",
    subtitle: Optional[str] = None,
) -> bytes:
    """Render simple markdown (headings, lists, paragraphs) to a PDF."""
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_left_margin(16)
    pdf.set_right_margin(16)
    pdf.set_x(16)

    pdf.add_font("DejaVu", "", FONT_REGULAR)
    pdf.add_font("DejaVu", "B", FONT_BOLD)
    pdf.add_font("DejaVu", "I", FONT_ITALIC)

    def write_block(style: str, size: float, text: str, h: float = 6) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("DejaVu", style, size)
        pdf.multi_cell(
            w=pdf.epw,
            h=h,
            text=text,
            new_x="LMARGIN",
            new_y="NEXT",
        )

    write_block("B", 16, title, h=9)
    if subtitle:
        pdf.set_text_color(90, 90, 90)
        write_block("", 10, subtitle, h=6)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_draw_color(30, 80, 140)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.ln(6)

    lines = (markdown_text or "").replace("\r\n", "\n").split("\n")
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(3)
            continue

        stripped = line.lstrip()
        if stripped.startswith("### "):
            write_block("B", 12, _strip_md_inline(stripped[4:]), h=7)
            pdf.ln(1)
            continue
        if stripped.startswith("## "):
            write_block("B", 13, _strip_md_inline(stripped[3:]), h=8)
            pdf.ln(1)
            continue
        if stripped.startswith("# "):
            write_block("B", 14, _strip_md_inline(stripped[2:]), h=8)
            pdf.ln(1)
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            write_block("", 11, f"• {_strip_md_inline(stripped[2:])}", h=6)
            continue
        if re.match(r"^\d+[\.\)]\s+", stripped):
            write_block("", 11, _strip_md_inline(stripped), h=6)
            continue
        if stripped.startswith("---"):
            pdf.ln(2)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
            pdf.ln(4)
            continue

        write_block("", 11, _strip_md_inline(stripped), h=6)

    pdf.set_y(-16)
    pdf.set_font("DejaVu", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Generado por Mao · EduApp", align="C")

    return bytes(pdf.output())


def build_support_material_attachment(
    markdown_text: str,
    *,
    title: str = "Material de apoyo",
    source_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns {filename, mime_type, content_base64} or None on failure.
    """
    if not (markdown_text or "").strip():
        return None
    try:
        filename = _safe_filename(
            f"Material_apoyo_{source_name or title}",
            fallback="Material_de_apoyo.pdf",
        )
        subtitle = f"Basado en: {source_name}" if source_name else None
        raw = markdown_to_pdf_bytes(
            markdown_text, title=title, subtitle=subtitle
        )
        return {
            "filename": filename,
            "mime_type": "application/pdf",
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "size_bytes": len(raw),
        }
    except Exception as e:
        logger.error("Failed to build support PDF: %s", e, exc_info=True)
        return None
