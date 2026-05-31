"""Hasta analiz raporunu PDF olarak dışa aktarır."""

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_UNICODE_FONT = "ReportUnicodeFont"
_FONT_REGISTERED = False


def _register_unicode_font() -> str:
    """Türkçe karakter destekli TTF font kaydı (Windows Arial/Segoe)."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return _UNICODE_FONT

    font_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(_UNICODE_FONT, str(font_path)))
            _FONT_REGISTERED = True
            return _UNICODE_FONT

    return "Helvetica"


def _qc_label(qc_score: float) -> str:
    if qc_score >= 100:
        return "Net (iyi kalite)"
    return "Düşük netlik — bulanık olabilir"


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Satır sonlarını PDF paragrafına uyarlar."""
    safe = text.replace("\n", "<br/>")
    return Paragraph(safe, style)


def build_patient_report_pdf(
    filename: str,
    analyzed_at: str,
    qc_score: float,
    wbc_count: int,
    rbc_count: int,
    wbc_avg_area: float,
    rbc_avg_area: float,
    ai_report: str,
) -> bytes:
    """Metrikler ve AI raporunu içeren PDF baytları üretir."""
    font_name = _register_unicode_font()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=15,
    )

    total = wbc_count + rbc_count
    metrics_data = [
        ["Metrik", "Değer"],
        ["WBC (Akyuvar) Sayısı", str(wbc_count)],
        ["RBC (Alyuvar) Sayısı", str(rbc_count)],
        ["Toplam Hücre", str(total)],
        ["Ortalama WBC Alanı", f"{wbc_avg_area:.2f} px²"],
        ["Ortalama RBC Alanı", f"{rbc_avg_area:.2f} px²"],
    ]
    metrics_table = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ]
        )
    )

    story = [
        _p("Kan Hücresi Sayım Raporu", title_style),
        _p(f"<b>İşlem Tarihi:</b> {analyzed_at}", body_style),
        _p(f"<b>Dosya Adı:</b> {filename}", body_style),
        Spacer(1, 0.4 * cm),
        _p("Kalite Kontrol (QC)", heading_style),
        _p(
            f"<b>Netlik Skoru:</b> {qc_score:.2f} — {_qc_label(qc_score)}",
            body_style,
        ),
        Spacer(1, 0.3 * cm),
        _p("Sayım ve Morfolojik Metrikler", heading_style),
        metrics_table,
        Spacer(1, 0.4 * cm),
        _p("AI Ön Değerlendirme Raporu", heading_style),
        _p(ai_report, body_style),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
