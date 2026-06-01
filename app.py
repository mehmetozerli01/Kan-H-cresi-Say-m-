"""
Kan Hücresi Sayımı — Streamlit Web Dashboard
Mevcut preprocessing ve segmentation modüllerini kullanır.
"""

import glob
import hashlib
import html
import io
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import psutil

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# src/ modüllerini import edebilmek için yol ekle
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CROPPED_DIR,
    REF_RBC_MAX,
    REF_RBC_MIN,
    REF_WBC_MAX,
    REF_WBC_MIN,
    WATERSHED_THRESH_COEFF,
    WBC_MIN_AREA,
)
from logging_setup import get_logger, setup_logging
from ai_reporter import generate_medical_report, get_api_key
from pdf_exporter import build_patient_report_pdf
from preprocessing import apply_clahe_and_blur, get_sharpness_score
from segmentation import count_wbc, count_rbc_watershed

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
WBC_GALLERY_COLUMNS = 4
PLOTLY_VIEWER_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
CELL_COLOR_MAP = {"WBC": "#2e7d32", "RBC": "#c62828"}
DEVELOPER_NAME = "Mehmet Özerli"


def init_session_state() -> None:
    """Oturum değişkenlerini başlatır."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "telemetry" not in st.session_state:
        st.session_state.telemetry = None
    if "upload_fingerprint" not in st.session_state:
        st.session_state.upload_fingerprint = ""


def append_analysis_history(entry: dict) -> None:
    """Tamamlanan analizi geçmiş listesine ekler."""
    init_session_state()
    st.session_state.history.append(entry)


def get_cropped_wbc_dir() -> Path:
    """config.CROPPED_DIR ile uyumlu mutlak klasör yolu."""
    return PROJECT_ROOT / CROPPED_DIR.strip("/\\")


def list_wbc_patch_paths(source_stem: str) -> list[str]:
    """Yüklenen görüntüye ait kırpılmış WBC dosya yollarını döndürür."""
    crop_dir = get_cropped_wbc_dir()
    pattern = str(crop_dir / f"{source_stem}_WBC_*.jpg")
    return sorted(glob.glob(pattern))


def list_all_wbc_crop_files() -> list[Path]:
    """extracted_wbcs klasöründeki tüm patch dosyalarını listeler."""
    crop_dir = get_cropped_wbc_dir()
    if not crop_dir.exists():
        return []
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        files.extend(crop_dir.glob(ext))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def build_wbc_dataset_zip() -> bytes | None:
    """Tüm WBC patch'lerini ZIP arşivi olarak döndürür."""
    crop_files = list_all_wbc_crop_files()
    if not crop_files:
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in crop_files:
            archive.write(file_path, arcname=file_path.name)
    buffer.seek(0)
    return buffer.getvalue()


def create_zoomable_image_fig(img_rgb: np.ndarray, title: str = ""):
    """Yakınlaştırılabilir / kaydırılabilir Plotly görüntü figürü."""
    fig = px.imshow(img_rgb, aspect="equal")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e3a5f")),
        margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
    )
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(
        visible=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def render_wbc_gallery(source_stem: str) -> None:
    """Tespit edilen WBC patch'lerini ızgara halinde gösterir."""
    patch_paths = list_wbc_patch_paths(source_stem)
    st.subheader("Tespit Edilen Akyuvarlar (WBC)")

    if not patch_paths:
        st.caption(
            f"`{get_cropped_wbc_dir()}` içinde bu görüntü için WBC patch bulunamadı."
        )
        return

    st.caption(f"{len(patch_paths)} adet WBC patch gösteriliyor.")

    for row_start in range(0, len(patch_paths), WBC_GALLERY_COLUMNS):
        cols = st.columns(WBC_GALLERY_COLUMNS)
        row_paths = patch_paths[row_start : row_start + WBC_GALLERY_COLUMNS]
        for col, patch_path in zip(cols, row_paths):
            with col:
                patch_bgr = cv2.imread(patch_path)
                if patch_bgr is None:
                    st.warning(Path(patch_path).name)
                    continue
                h, w = patch_bgr.shape[:2]
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                st.image(patch_rgb, use_container_width=True)
                st.caption(f"{w}×{h} px")


def analyze_blood_image(
    img_bgr: np.ndarray,
    source_stem: str | None = None,
    wbc_min_area: int | None = None,
    watershed_thresh_coeff: float | None = None,
) -> tuple[int, int, float, float, np.ndarray, np.ndarray, list[dict]]:
    """Yüklenen BGR görüntü üzerinde WBC/RBC sayımı ve annotasyon."""
    blurred = apply_clahe_and_blur(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    final_output = img_rgb.copy()

    wbc_count, wbc_avg_area, final_output, wbc_cells = count_wbc(
        blurred,
        final_output,
        img_bgr=img_bgr,
        source_stem=source_stem,
        wbc_min_area=wbc_min_area,
    )
    rbc_count, rbc_avg_area, final_output, rbc_cells = count_rbc_watershed(
        img_bgr,
        blurred,
        final_output,
        watershed_thresh_coeff=watershed_thresh_coeff,
    )
    cell_records = wbc_cells + rbc_cells
    return (
        wbc_count,
        rbc_count,
        wbc_avg_area,
        rbc_avg_area,
        img_rgb,
        final_output,
        cell_records,
    )


def build_cell_details_df(cell_records: list[dict]) -> pd.DataFrame:
    """Hücre bazlı detay tablosu."""
    if not cell_records:
        return pd.DataFrame(columns=["Hücre Tipi", "Alan (px)", "Durum"])
    df = pd.DataFrame(cell_records)
    return df[["Hücre Tipi", "Alan (px)", "Durum"]]


def read_uploaded_bytes(uploaded_file) -> tuple[bytes, str]:
    """Yüklenen dosyanın ham baytlarını ve adını döndürür."""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return file_bytes, uploaded_file.name


def load_uploaded_image(uploaded_file) -> np.ndarray | None:
    """Streamlit yüklemesini OpenCV BGR dizisine çevirir."""
    file_bytes, _ = read_uploaded_bytes(uploaded_file)
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def build_upload_fingerprint(uploaded_files: list) -> str:
    """Yükleme seti için önbellek anahtarı parmak izi."""
    parts: list[str] = []
    for uploaded in uploaded_files:
        file_bytes, name = read_uploaded_bytes(uploaded)
        digest = hashlib.md5(file_bytes, usedforsecurity=False).hexdigest()
        parts.append(f"{name}:{len(file_bytes)}:{digest}")
    return "|".join(sorted(parts))


def sync_upload_cache(uploaded_files: list) -> None:
    """Yeni görüntü yüklendiğinde analiz önbelleğini temizler."""
    fingerprint = build_upload_fingerprint(uploaded_files) if uploaded_files else ""
    if st.session_state.get("upload_fingerprint") != fingerprint:
        run_cached_image_analysis.clear()
        st.session_state.upload_fingerprint = fingerprint


def collect_system_telemetry() -> tuple[float, float]:
    """Anlık CPU ve RAM kullanım yüzdeleri."""
    cpu_percent = psutil.cpu_percent(interval=0.05)
    ram_percent = psutil.virtual_memory().percent
    return cpu_percent, ram_percent


def store_telemetry(processing_sec: float) -> None:
    """Son analiz telemetrisini oturuma yazar."""
    cpu_percent, ram_percent = collect_system_telemetry()
    st.session_state.telemetry = {
        "processing_sec": processing_sec,
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
    }


@st.cache_data(show_spinner=False)
def run_cached_image_analysis(
    file_bytes: bytes,
    filename: str,
    wbc_param: int | None,
    ws_param: float | None,
) -> dict | None:
    """OpenCV/segmentasyon analizi — Streamlit önbellekli."""
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    source_stem = Path(filename).stem
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    qc_score = get_sharpness_score(img_gray)

    wbc, rbc, wbc_avg, rbc_avg, original_rgb, final_rgb, cell_records = (
        analyze_blood_image(
            img_bgr,
            source_stem=source_stem,
            wbc_min_area=wbc_param,
            watershed_thresh_coeff=ws_param,
        )
    )

    return {
        "wbc": wbc,
        "rbc": rbc,
        "wbc_avg_area": wbc_avg,
        "rbc_avg_area": rbc_avg,
        "qc_score": qc_score,
        "original": original_rgb,
        "final": final_rgb,
        "cell_records": cell_records,
        "filename": filename,
        "source_stem": source_stem,
    }


def normalize_uploaded_files(uploaded) -> list:
    """file_uploader çıktısını dosya listesine çevirir."""
    if uploaded is None:
        return []
    if isinstance(uploaded, list):
        return uploaded
    return [uploaded]


def process_single_file(
    uploaded_file,
    wbc_param: int | None,
    ws_param: float | None,
) -> tuple[dict | None, float]:
    """Tek dosyayı analiz eder; (sonuç, süre_sn) veya (None, süre) döner."""
    file_bytes, filename = read_uploaded_bytes(uploaded_file)
    t0 = time.perf_counter()
    cached = run_cached_image_analysis(file_bytes, filename, wbc_param, ws_param)
    elapsed = time.perf_counter() - t0

    if cached is None:
        return None, elapsed

    result = dict(cached)
    result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result, elapsed


def render_clinical_alerts(wbc: int, rbc: int) -> None:
    """Referans aralığı dışındaki sayımlar için bilgilendirme uyarıları."""
    if wbc < REF_WBC_MIN:
        st.warning(
            "⚠️ **Düşük WBC Sayımı:** Referans aralığının altında "
            f"({REF_WBC_MIN}–{REF_WBC_MAX}). İmmün yanıt değerlendirmesi önerilir "
            "(Leukopeni şüphesi — yalnızca bilgilendirme)."
        )
    elif wbc > REF_WBC_MAX:
        st.warning(
            "⚠️ **Yüksek WBC Sayımı:** Referans aralığının üzerinde "
            f"({REF_WBC_MIN}–{REF_WBC_MAX}). Enfeksiyon veya inflamasyon belirtisi "
            "olabilir (Lökositoz şüphesi — yalnızca bilgilendirme)."
        )

    if rbc < REF_RBC_MIN:
        st.warning(
            "⚠️ **Düşük RBC Sayımı:** Referans aralığının altında "
            f"({REF_RBC_MIN}–{REF_RBC_MAX}). Anemi riski klinik açıdan "
            "değerlendirilmelidir (yalnızca bilgilendirme)."
        )
    elif rbc > REF_RBC_MAX:
        st.info(
            "ℹ️ **Yüksek RBC Sayımı:** Referans aralığının üzerinde "
            f"({REF_RBC_MIN}–{REF_RBC_MAX}). Polisitemi veya yoğunluk artışı "
            "açısından değerlendirme düşünülebilir (yalnızca bilgilendirme)."
        )


def build_size_distribution_histogram(cell_df: pd.DataFrame):
    """Hücre alan dağılımı histogramı (Price-Jones / RDW benzeri görünüm)."""
    if cell_df.empty:
        return None

    fig = px.histogram(
        cell_df,
        x="Alan (px)",
        color="Hücre Tipi",
        barmode="overlay",
        nbins=30,
        opacity=0.72,
        color_discrete_map=CELL_COLOR_MAP,
    )
    fig.update_layout(
        title=dict(
            text="Hücre Boyut Dağılım Analizi",
            font=dict(size=16, color="#1a365d"),
        ),
        xaxis_title="Alan (px)",
        yaxis_title="Hücre Sayısı",
        legend_title="Hücre Tipi",
        bargap=0.05,
        plot_bgcolor="rgba(248,250,252,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=380,
    )
    return fig


def build_batch_comparison_chart(batch_results: list[dict]):
    """Çoklu dosya WBC/RBC karşılaştırma grafiği."""
    rows: list[dict] = []
    for item in batch_results:
        label = Path(item["filename"]).stem
        rows.append({"Dosya": label, "Hücre Tipi": "WBC", "Sayı": item["wbc"]})
        rows.append({"Dosya": label, "Hücre Tipi": "RBC", "Sayı": item["rbc"]})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Dosya",
        y="Sayı",
        color="Hücre Tipi",
        barmode="group",
        text="Sayı",
        color_discrete_map={
            "WBC": CELL_COLOR_MAP["WBC"],
            "RBC": CELL_COLOR_MAP["RBC"],
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=dict(
            text="Toplu Analiz — WBC / RBC Karşılaştırması",
            font=dict(size=16, color="#1a365d"),
        ),
        xaxis_title="Görüntü",
        yaxis_title="Hücre Sayısı",
        legend_title="Hücre Tipi",
        plot_bgcolor="rgba(248,250,252,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=420,
        xaxis_tickangle=-35,
    )
    return fig


def build_distribution_chart(wbc: int, rbc: int):
    """WBC/RBC sayısal dağılım grafiği."""
    df = pd.DataFrame(
        {
            "Hücre Tipi": ["WBC (Akyuvar)", "RBC (Alyuvar)"],
            "Sayı": [wbc, rbc],
        }
    )
    fig = px.bar(
        df,
        x="Hücre Tipi",
        y="Sayı",
        color="Hücre Tipi",
        color_discrete_map={
            "WBC (Akyuvar)": CELL_COLOR_MAP["WBC"],
            "RBC (Alyuvar)": CELL_COLOR_MAP["RBC"],
        },
        text="Sayı",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        margin=dict(t=40, b=40, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Hücre Sayısı",
        xaxis_title="",
        font=dict(family="Segoe UI, sans-serif", size=13),
        height=360,
    )
    return fig


def build_report_table(results: dict) -> pd.DataFrame:
    """Detaylı analiz raporu tablosu."""
    wbc = results["wbc"]
    rbc = results["rbc"]
    total = wbc + rbc
    wbc_pct = round((wbc / total) * 100, 2) if total > 0 else 0.0
    rbc_pct = round((rbc / total) * 100, 2) if total > 0 else 0.0

    rows = [
        {"Alan": "Dosya Adı", "Değer": results["filename"]},
        {"Alan": "Analiz Tarihi/Saati", "Değer": results["analyzed_at"]},
        {"Alan": "WBC (Akyuvar) Sayısı", "Değer": str(wbc)},
        {"Alan": "RBC (Alyuvar) Sayısı", "Değer": str(rbc)},
        {
            "Alan": "Ortalama WBC Alanı (px)",
            "Değer": f"{results.get('wbc_avg_area', 0):.2f}",
        },
        {
            "Alan": "Ortalama RBC Alanı (px)",
            "Değer": f"{results.get('rbc_avg_area', 0):.2f}",
        },
        {
            "Alan": "Netlik Skoru (QC)",
            "Değer": f"{results.get('qc_score', 0):.2f}",
        },
        {"Alan": "Toplam Hücre Sayısı", "Değer": str(total)},
        {"Alan": "WBC Oranı (%)", "Değer": f"{wbc_pct}"},
        {"Alan": "RBC Oranı (%)", "Değer": f"{rbc_pct}"},
    ]
    if results.get("dev_mode"):
        rows.extend(
            [
                {
                    "Alan": "Dev: WBC Min Alan",
                    "Değer": str(results.get("wbc_min_area_param", WBC_MIN_AREA)),
                },
                {
                    "Alan": "Dev: Watershed Eşiği",
                    "Değer": str(
                        results.get("watershed_coeff_param", WATERSHED_THRESH_COEFF)
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_csv_download(results: dict) -> bytes:
    """İndirilebilir CSV içeriği."""
    export_df = pd.DataFrame(
        [
            {
                "Dosya Adı": results["filename"],
                "Analiz Tarihi/Saati": results["analyzed_at"],
                "WBC Sayısı": results["wbc"],
                "RBC Sayısı": results["rbc"],
                "Toplam Hücre Sayısı": results["wbc"] + results["rbc"],
                "Ortalama WBC Alanı (px)": results.get("wbc_avg_area", 0),
                "Ortalama RBC Alanı (px)": results.get("rbc_avg_area", 0),
                "Netlik Skoru": results.get("qc_score", 0),
            }
        ]
    )
    return export_df.to_csv(index=False).encode("utf-8-sig")


def render_history_panel() -> None:
    """Sidebar altında oturum geçmişi."""
    init_session_state()
    st.divider()
    with st.expander("Geçmiş Analizler", expanded=False):
        history = st.session_state.history
        if not history:
            st.caption("Bu oturumda henüz analiz yapılmadı.")
            return
        for idx, item in enumerate(reversed(history), start=1):
            st.markdown(
                f"**{idx}. {item['Dosya']}**  \n"
                f"WBC: {item['WBC']} | RBC: {item['RBC']} | "
                f"Netlik: {item['Netlik']:.2f}  \n"
                f"<span style='color:#6b7c93;font-size:0.85em'>{item['Tarih']}</span>",
                unsafe_allow_html=True,
            )
            if idx < len(history):
                st.markdown("<hr style='margin:0.4em 0;opacity:0.3'>", unsafe_allow_html=True)


def render_sidebar_controls():
    """Sidebar: yükleme, dev mode, ZIP ve geçmiş."""
    st.header("📁 Görüntü Yükle")
    st.caption("Bir veya birden fazla JPG/JPEG dosyası sürükleyip bırakın.")
    uploaded_file = st.file_uploader(
        "Dosya seçin",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.divider()
    dev_mode = st.checkbox("Gelişmiş Ayarlar (Dev Mode)")
    wbc_min_area: int | None = None
    watershed_coeff: float | None = None

    if dev_mode:
        wbc_min_area = st.slider(
            "WBC Minimum Alan",
            min_value=200,
            max_value=2000,
            value=int(WBC_MIN_AREA),
            step=50,
            help=f"Varsayılan (config): {WBC_MIN_AREA}",
        )
        watershed_coeff = st.slider(
            "Watershed Eşiği",
            min_value=0.01,
            max_value=0.5,
            value=float(WATERSHED_THRESH_COEFF),
            step=0.01,
            help=f"Varsayılan (config): {WATERSHED_THRESH_COEFF}",
        )

    analyze_clicked = st.button(
        "Analizi Başlat", type="primary", use_container_width=True
    )

    st.divider()
    st.markdown("##### 🤖 Yapay Zeka Veri Seti")
    zip_data = build_wbc_dataset_zip()
    crop_count = len(list_all_wbc_crop_files())

    if zip_data is None:
        st.warning("Henüz çıkarılmış hücre yok.")
        st.caption(
            f"Analiz sonrası patch'ler `{get_cropped_wbc_dir()}` klasörüne kaydedilir."
        )
    else:
        st.caption(f"Klasörde {crop_count} WBC patch mevcut.")
        st.download_button(
            label="📦 WBC Veri Setini İndir (ZIP)",
            data=zip_data,
            file_name="wbc_extracted_dataset.zip",
            mime="application/zip",
            use_container_width=True,
        )

    render_history_panel()
    render_system_telemetry_panel()

    return uploaded_file, analyze_clicked, dev_mode, wbc_min_area, watershed_coeff


def render_system_telemetry_panel() -> None:
    """Sidebar altında son analiz donanım telemetrisi."""
    telemetry = st.session_state.get("telemetry")
    if not telemetry:
        return

    st.divider()
    st.markdown("##### 📡 Sistem Telemetrisi")
    st.info(
        f"**Görüntü İşleme Süresi:** {telemetry['processing_sec']:.2f} sn  \n"
        f"**CPU Kullanımı:** %{telemetry['cpu_percent']:.1f}  \n"
        f"**RAM Kullanımı:** %{telemetry['ram_percent']:.1f}"
    )


def apply_custom_styles() -> None:
    """HealthTech kurumsal tema CSS."""
    st.markdown(
        """
        <style>
        /* Ana arka plan */
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #1e2d3d 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e8eef4 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #dce6f0 !important;
        }
        /* Başlıklar */
        .main-header {
            font-size: 2.25rem;
            font-weight: 700;
            color: #1a365d;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #5a6b7d;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        .section-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 18px rgba(26, 54, 93, 0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        /* Metrik kartları */
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff 0%, #f0f5fb 100%);
            border: 1px solid #d0deef;
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            box-shadow: 0 6px 20px rgba(30, 58, 95, 0.10);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(30, 58, 95, 0.14);
        }
        div[data-testid="stMetric"] label {
            font-size: 0.92rem !important;
            color: #3d5a80 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: #1a365d !important;
        }
        /* Birincil butonlar */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            box-shadow: 0 6px 22px rgba(59, 130, 246, 0.55) !important;
            transform: translateY(-1px);
        }
        /* İkincil butonlar */
        .stDownloadButton > button {
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
            transition: box-shadow 0.2s ease !important;
        }
        .stDownloadButton > button:hover {
            box-shadow: 0 4px 12px rgba(30, 58, 95, 0.12) !important;
        }
        /* AI rapor kutusu */
        .ai-report-box {
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            line-height: 1.6;
            color: #1e3a2f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_row(results: dict) -> None:
    """Metrik kartları ve klinik uyarılar."""
    col_wbc, col_rbc, col_total, col_wbc_area, col_rbc_area = st.columns(5)
    total_cells = results["wbc"] + results["rbc"]
    col_wbc.metric("WBC Sayısı", results["wbc"])
    col_rbc.metric("RBC Sayısı", results["rbc"])
    col_total.metric("Toplam Hücre", total_cells)
    col_wbc_area.metric(
        "Ort. WBC Alanı",
        f"{results.get('wbc_avg_area', 0):.1f}",
        help="Geçerli WBC konturlarının ortalama piksel alanı",
    )
    col_rbc_area.metric(
        "Ort. RBC Alanı",
        f"{results.get('rbc_avg_area', 0):.1f}",
        help="Geçerli RBC konturlarının ortalama piksel alanı",
    )
    st.caption(
        f"Morfolojik alan birimi: piksel² (px) | Referans: "
        f"WBC {REF_WBC_MIN}–{REF_WBC_MAX}, RBC {REF_RBC_MIN}–{REF_RBC_MAX}"
    )
    render_clinical_alerts(results["wbc"], results["rbc"])
    render_system_telemetry_inline()


def render_system_telemetry_inline() -> None:
    """Metrik satırı altında kısa telemetri özeti."""
    telemetry = st.session_state.get("telemetry")
    if not telemetry:
        return
    st.caption(
        f"📡 Sistem Telemetrisi — İşlem: {telemetry['processing_sec']:.2f} sn | "
        f"CPU: %{telemetry['cpu_percent']:.1f} | RAM: %{telemetry['ram_percent']:.1f}"
    )


def render_single_results(results: dict) -> None:
    """Tek görüntü için tam detaylı dashboard."""
    if results.get("dev_mode"):
        st.caption(
            "⚙️ Son analiz Gelişmiş Ayarlar ile çalıştırıldı "
            f"(WBC min: {results.get('wbc_min_area_param')}, "
            f"Watershed: {results.get('watershed_coeff_param')})."
        )

    render_metrics_row(results)

    st.caption("🔍 Görüntülerde tekerlek ile zoom, sürükleyerek pan yapabilirsiniz.")
    col_orig, col_final = st.columns(2)
    with col_orig:
        st.subheader("Orijinal Görüntü")
        st.plotly_chart(
            create_zoomable_image_fig(results["original"]),
            use_container_width=True,
            config=PLOTLY_VIEWER_CONFIG,
        )
    with col_final:
        st.subheader("Analiz Sonucu (Final)")
        st.plotly_chart(
            create_zoomable_image_fig(
                results["final"],
                title=f"{results['wbc']} WBC | {results['rbc']} RBC",
            ),
            use_container_width=True,
            config=PLOTLY_VIEWER_CONFIG,
        )

    st.divider()
    st.subheader("Hücre Detay Tablosu")
    cell_df = build_cell_details_df(results.get("cell_records", []))
    st.dataframe(
        cell_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hücre Tipi": st.column_config.TextColumn("Hücre Tipi", width="small"),
            "Alan (px)": st.column_config.NumberColumn("Alan (px)", format="%.2f"),
            "Durum": st.column_config.TextColumn("Durum", width="small"),
        },
    )
    st.caption(
        f"Toplam {len(cell_df)} geçerli hücre listelendi. "
        "Sütun başlıklarına tıklayarak sıralayabilirsiniz."
    )

    source_stem = results.get("source_stem", Path(results["filename"]).stem)
    render_wbc_gallery(source_stem)

    st.divider()
    st.subheader("Hücre Dağılımı ve Boyut Analizi")
    count_col, hist_col = st.columns(2)
    cell_df = build_cell_details_df(results.get("cell_records", []))
    with count_col:
        st.plotly_chart(
            build_distribution_chart(results["wbc"], results["rbc"]),
            use_container_width=True,
        )
    with hist_col:
        size_fig = build_size_distribution_histogram(cell_df)
        if size_fig:
            st.plotly_chart(size_fig, use_container_width=True)
        else:
            st.caption("Histogram için yeterli hücre verisi yok.")

    st.markdown("##### Dışa Aktar")
    csv_name = Path(results["filename"]).stem + "_analiz.csv"
    st.download_button(
        label="📥 CSV İndir",
        data=build_csv_download(results),
        file_name=csv_name,
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Detaylı Analiz Raporunu Gör"):
        st.dataframe(
            build_report_table(results),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Yapay Zeka Tıbbi Ön Değerlendirme")

    if st.button(
        "Yapay Zeka Raporu Oluştur 🤖",
        type="primary",
        use_container_width=True,
        key="ai_report_btn",
    ):
        if get_api_key() is None:
            st.warning(
                "API anahtarı bulunamadı, lütfen .env dosyanızı güncelleyin "
                "(GEMINI_API_KEY)."
            )
        else:
            with st.spinner("Hematolog asistanı rapor yazıyor..."):
                report_text, error_msg = generate_medical_report(
                    wbc_count=results["wbc"],
                    rbc_count=results["rbc"],
                    wbc_area=float(results.get("wbc_avg_area", 0)),
                    rbc_area=float(results.get("rbc_avg_area", 0)),
                    qc_score=float(results.get("qc_score", 0)),
                )
            if error_msg:
                st.error(error_msg)
            else:
                st.session_state["ai_report"] = report_text

    ai_report = st.session_state.get("ai_report")
    if ai_report:
        st.markdown("#### AI Ön Değerlendirme Raporu")
        safe_report = html.escape(ai_report).replace("\n", "<br>")
        st.markdown(
            f'<div class="ai-report-box">{safe_report}</div>',
            unsafe_allow_html=True,
        )
        pdf_bytes = build_patient_report_pdf(
            filename=results["filename"],
            analyzed_at=results["analyzed_at"],
            qc_score=float(results.get("qc_score", 0)),
            wbc_count=results["wbc"],
            rbc_count=results["rbc"],
            wbc_avg_area=float(results.get("wbc_avg_area", 0)),
            rbc_avg_area=float(results.get("rbc_avg_area", 0)),
            ai_report=ai_report,
        )
        pdf_name = Path(results["filename"]).stem + "_tam_rapor.pdf"
        st.download_button(
            label="Tam Raporu PDF Olarak İndir 📄",
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            use_container_width=True,
        )


def render_architecture_tab() -> None:
    """Jüri sunumu için sistem mimarisi ve proje dokümantasyonu."""
    st.markdown(
        """
        <div class="section-card">
        <h2 style="color:#1a365d;margin-top:0;">Kan Hücresi Sayımı — Sistem Mimarisi</h2>
        <p style="color:#5a6b7d;line-height:1.7;">
        Bu modül, kan yayması mikroskop görüntülerinden otonom WBC/RBC sayımı yapan
        uçtan uca bir HealthTech analiz platformunun teknik özetidir.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Proje Amacı")
    st.markdown(
        """
        BCCD (Blood Cell Count and Detection) veri seti formatındaki kan yayması
        görüntülerinden **akyuvar (WBC)** ve **alyuvar (RBC)** hücrelerinin otonom
        tespiti ve sayımı; klinik karar desteği için sayısal metrikler, morfolojik
        dağılım grafikleri ve yapay zeka destekli ön değerlendirme raporu üretimi.
        """
    )

    st.markdown("### 🔬 Görüntü İşleme Pipeline")
    st.markdown(
        """
        | Adım | Teknoloji | Açıklama |
        |------|-----------|----------|
        | 1 | **CLAHE** | Kontrast sınırlı adaptif histogram eşitleme |
        | 2 | **Median Blur** | Gürültü azaltma, kenar koruma |
        | 3 | **HSV Maskeleme** | Morfoloji tabanlı WBC (mor çekirdek) segmentasyonu |
        | 4 | **Adaptive Threshold** | RBC için yerel eşikleme |
        | 5 | **Distance Transform** | Alyuvar çekirdek merkezlerinin ayrıştırılması |
        | 6 | **Watershed** | Bitişik RBC kütlelerinin bölünmesi |
        """
    )

    st.markdown("### 🤖 Yapay Zeka Entegrasyonu")
    st.markdown(
        """
        - **Model:** Google Gemini (`gemini-flash-latest`) LLM API
        - **Modül:** `src/ai_reporter.py` — prompt engineering ile hematoloji odaklı
          ön değerlendirme metni
        - **Çıktı:** Streamlit arayüzünde rapor + **ReportLab** ile PDF hasta özeti
        - **Not:** Sonuçlar yalnızca bilgilendirme amaçlıdır; kesin tanı yerine geçmez.
        """
    )

    st.markdown("### 🏗️ Yazılım Mimarisi")
    st.markdown(
        """
        ```
        app.py (Streamlit UI)
            ├── src/preprocessing.py   → CLAHE, netlik QC
            ├── src/segmentation.py    → WBC/RBC sayımı
            ├── src/config.py          → Merkezi parametreler
            ├── src/main.py            → CLI toplu işleme + Excel
            ├── src/ai_reporter.py     → Gemini raporlama
            └── src/pdf_exporter.py    → PDF dışa aktarım
        ```
        """
    )

    st.markdown("### ⚡ Performans ve Önbellek")
    st.markdown(
        """
        - `@st.cache_data` ile OpenCV analizi önbelleğe alınır; PDF/AI butonlarında
          gereksiz yeniden hesaplama engellenir.
        - Yeni görüntü yüklendiğinde parmak izi değişir ve önbellek otomatik temizlenir.
        - `psutil` ile CPU/RAM telemetrisi; `time.perf_counter` ile işlem süresi ölçülür.
        """
    )

    st.markdown("### 👤 Geliştirici")
    st.markdown(f"**{DEVELOPER_NAME}**")

    st.markdown("### 📦 Teknoloji Yığını")
    st.markdown(
        "Python · OpenCV · NumPy · Pandas · Streamlit · Plotly · "
        "Google Generative AI · ReportLab · psutil"
    )


def render_live_analysis_tab(
    uploaded_file,
    analyze_clicked: bool,
    dev_mode: bool,
    wbc_min_area: int | None,
    watershed_coeff: float | None,
) -> None:
    """Canlı analiz sekmesi: yükleme sonuçları ve grafikler."""
    uploaded_files = normalize_uploaded_files(uploaded_file)

    if analyze_clicked:
        if not uploaded_files:
            st.warning("Lütfen önce en az bir görüntü yükleyin.")
        else:
            wbc_param = wbc_min_area if dev_mode else None
            ws_param = watershed_coeff if dev_mode else None

            if dev_mode:
                st.info(
                    f"Dev Mode aktif — WBC min alan: {wbc_param}, "
                    f"Watershed: {ws_param}"
                )

            if len(uploaded_files) == 1:
                file = uploaded_files[0]
                try:
                    with st.spinner("Analiz yapılıyor..."):
                        result, elapsed = process_single_file(
                            file, wbc_param, ws_param
                        )
                except Exception as exc:
                    logger.error("Analiz hatası (%s): %s", file.name, exc)
                    st.error(f"Analiz sırasında hata oluştu: {exc}")
                else:
                    if result is None:
                        st.error(
                            f"{file.name} okunamadı. Geçerli bir JPG/JPEG dosyası yükleyin."
                        )
                    else:
                        store_telemetry(elapsed)
                        result["dev_mode"] = dev_mode
                        result["wbc_min_area_param"] = wbc_param or WBC_MIN_AREA
                        result["watershed_coeff_param"] = (
                            ws_param or WATERSHED_THRESH_COEFF
                        )
                        append_analysis_history(
                            {
                                "Dosya": result["filename"],
                                "WBC": result["wbc"],
                                "RBC": result["rbc"],
                                "Netlik": result["qc_score"],
                                "Tarih": result["analyzed_at"],
                            }
                        )
                        st.session_state.pop("ai_report", None)
                        st.session_state.pop("batch_results", None)
                        st.session_state["results"] = result
                        build_wbc_dataset_zip.clear()
            else:
                batch_results: list[dict] = []
                total_elapsed = 0.0
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                for idx, file in enumerate(uploaded_files):
                    status_text.text(
                        f"İşleniyor ({idx + 1}/{len(uploaded_files)}): {file.name}"
                    )
                    try:
                        result, elapsed = process_single_file(
                            file, wbc_param, ws_param
                        )
                        total_elapsed += elapsed
                    except Exception as exc:
                        logger.error("Analiz hatası (%s): %s", file.name, exc)
                        st.warning(f"{file.name} atlandı: {exc}")
                        result = None

                    if result is not None:
                        batch_results.append(result)
                        append_analysis_history(
                            {
                                "Dosya": result["filename"],
                                "WBC": result["wbc"],
                                "RBC": result["rbc"],
                                "Netlik": result["qc_score"],
                                "Tarih": result["analyzed_at"],
                            }
                        )

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                progress_bar.empty()
                status_text.empty()

                if not batch_results:
                    st.error("Hiçbir görüntü işlenemedi.")
                else:
                    store_telemetry(total_elapsed)
                    st.session_state.pop("ai_report", None)
                    st.session_state.pop("results", None)
                    st.session_state["batch_results"] = batch_results
                    build_wbc_dataset_zip.clear()
                    st.success(
                        f"{len(batch_results)} / {len(uploaded_files)} görüntü "
                        "başarıyla analiz edildi."
                    )

    batch_results = st.session_state.get("batch_results")
    results = st.session_state.get("results")

    if batch_results:
        render_batch_summary(batch_results)
        render_system_telemetry_inline()
    elif results:
        render_single_results(results)
    else:
        st.info(
            "Sol menüden bir veya birden fazla kan hücresi görüntüsü yükleyip "
            "**Analizi Başlat** butonuna basarak analizi başlatın."
        )


def render_batch_summary(batch_results: list[dict]) -> None:
    """Çoklu dosya toplu özet görünümü."""
    st.subheader(f"Toplu Analiz Özeti ({len(batch_results)} görüntü)")
    st.plotly_chart(
        build_batch_comparison_chart(batch_results),
        use_container_width=True,
    )

    summary_df = pd.DataFrame(
        [
            {
                "Dosya Adı": r["filename"],
                "WBC": r["wbc"],
                "RBC": r["rbc"],
                "Toplam": r["wbc"] + r["rbc"],
                "Netlik Skoru": round(r["qc_score"], 2),
                "Ort. WBC Alanı (px)": round(r["wbc_avg_area"], 2),
                "Ort. RBC Alanı (px)": round(r["rbc_avg_area"], 2),
            }
            for r in batch_results
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.caption(
        "Çoklu dosya modunda detaylı zoom ve WBC galerisi için tek bir görüntü "
        "yükleyip analiz edin."
    )


def main() -> None:
    setup_logging()
    init_session_state()
    st.set_page_config(
        page_title="Kan Hücresi Sayımı",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    st.markdown(
        '<p class="main-header">🔬 Kan Hücresi Sayım Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">BCCD mikroskop görüntülerinden akyuvar (WBC) ve alyuvar (RBC) '
        "tespitini CLAHE, HSV segmentasyonu ve Watershed algoritmalarıyla gerçekleştirir. "
        "Görüntülerde fare tekerleği ile yakınlaştırabilir, sürükleyerek kaydırabilirsiniz.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        (
            uploaded_file,
            analyze_clicked,
            dev_mode,
            wbc_min_area,
            watershed_coeff,
        ) = render_sidebar_controls()

    sync_upload_cache(normalize_uploaded_files(uploaded_file))

    tab_live, tab_arch = st.tabs(["🔬 Canlı Analiz", "📚 Sistem Mimarisi"])

    with tab_live:
        render_live_analysis_tab(
            uploaded_file,
            analyze_clicked,
            dev_mode,
            wbc_min_area,
            watershed_coeff,
        )

    with tab_arch:
        render_architecture_tab()


if __name__ == "__main__":
    main()
