"""Gemini LLM ile tıbbi ön değerlendirme raporu üretimi."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_PLACEHOLDER_KEYS = {"", "buraya_anahtar_gelecek", "your_api_key_here"}


def get_api_key() -> str | None:
    """Geçerli Gemini API anahtarını döndürür; yoksa None."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key.lower() in _PLACEHOLDER_KEYS:
        return None
    return key


def _build_prompt(
    wbc_count: int,
    rbc_count: int,
    wbc_area: float,
    rbc_area: float,
    qc_score: float,
) -> str:
    return (
        "Sen uzman bir hematologsun. Mikroskopik kan yayması analizinde şu veriler "
        "elde edildi:\n"
        f"- Akyuvar (WBC) sayısı: {wbc_count}\n"
        f"- Alyuvar (RBC) sayısı: {rbc_count}\n"
        f"- Ortalama WBC piksel alanı: {wbc_area:.2f} px²\n"
        f"- Ortalama RBC piksel alanı: {rbc_area:.2f} px²\n"
        f"- Görüntü netlik skoru (QC, Laplacian varyans): {qc_score:.2f} "
        "(düşük skor bulanık görüntüye işaret edebilir)\n\n"
        "Bu değerlere bakarak, hücre morfolojisi ve sayımlar hakkında profesyonel, "
        "kısa bir tıbbi ön değerlendirme raporu yaz. "
        "(Not: Kesin teşhis koyma, sadece gözlemleri yorumla)."
    )


def generate_medical_report(
    wbc_count: int,
    rbc_count: int,
    wbc_area: float,
    rbc_area: float,
    qc_score: float,
) -> tuple[str | None, str | None]:
    """
    LLM ile tıbbi ön değerlendirme metni üretir.

    Returns:
        (rapor_metni, hata_mesaji) — başarıda hata_mesaji None olur.
    """
    api_key = get_api_key()
    if api_key is None:
        return None, (
            "API anahtarı bulunamadı, lütfen .env dosyanızı güncelleyin "
            "(GEMINI_API_KEY)."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = _build_prompt(wbc_count, rbc_count, wbc_area, rbc_area, qc_score)
        response = model.generate_content(prompt)

        if not response or not getattr(response, "text", None):
            return None, "Modelden boş yanıt alındı. Lütfen tekrar deneyin."

        return response.text.strip(), None

    except ImportError:
        return None, (
            "google-generativeai yüklü değil. "
            "Terminalde: pip install google-generativeai"
        )
    except Exception as exc:
        return None, f"Rapor oluşturulamadı: {exc}"
