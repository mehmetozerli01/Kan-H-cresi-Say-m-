"""
Kan Hücresi Sayımı — Streamlit Web Dashboard
Mevcut preprocessing ve segmentation modüllerini kullanır.
"""

import glob
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# src/ modüllerini import edebilmek için yol ekle
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CROPPED_DIR, WATERSHED_THRESH_COEFF, WBC_MIN_AREA
from logging_setup import get_logger, setup_logging
from preprocessing import apply_clahe_and_blur
from segmentation import count_wbc, count_rbc_watershed

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
WBC_GALLERY_COLUMNS = 4


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
) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
    """Yüklenen BGR görüntü üzerinde WBC/RBC sayımı ve annotasyon."""
    blurred = apply_clahe_and_blur(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    final_output = img_rgb.copy()

    wbc_count, wbc_avg_area, final_output = count_wbc(
        blurred,
        final_output,
        img_bgr=img_bgr,
        source_stem=source_stem,
        wbc_min_area=wbc_min_area,
    )
    rbc_count, rbc_avg_area, final_output = count_rbc_watershed(
        img_bgr,
        blurred,
        final_output,
        watershed_thresh_coeff=watershed_thresh_coeff,
    )
    return wbc_count, rbc_count, wbc_avg_area, rbc_avg_area, img_rgb, final_output


def load_uploaded_image(uploaded_file) -> np.ndarray | None:
    """Streamlit yüklemesini OpenCV BGR dizisine çevirir."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr


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
            "WBC (Akyuvar)": "#27ae60",
            "RBC (Alyuvar)": "#c0392b",
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
                    "Değer": str(results.get("watershed_coeff_param", WATERSHED_THRESH_COEFF)),
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
            }
        ]
    )
    return export_df.to_csv(index=False).encode("utf-8-sig")


def render_sidebar_controls() -> tuple[bool, int | None, float | None]:
    """Sidebar: yükleme, dev mode slider'ları ve WBC ZIP indirme."""
    st.header("📁 Görüntü Yükle")
    st.caption("JPG veya JPEG formatında kan yayması görüntüsü seçin.")
    uploaded_file = st.file_uploader(
        "Dosya seçin",
        type=["jpg", "jpeg"],
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

    return uploaded_file, analyze_clicked, dev_mode, wbc_min_area, watershed_coeff


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #5a6b7d;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8fafc 0%, #eef4fb 100%);
            border: 1px solid #d6e4f0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            box-shadow: 0 2px 8px rgba(30, 58, 95, 0.06);
        }
        div[data-testid="stMetric"] label {
            font-size: 0.95rem !important;
            color: #3d5a80 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #1e3a5f !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    setup_logging()
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
        "tespitini CLAHE, HSV segmentasyonu ve Watershed algoritmalarıyla gerçekleştirir.</p>",
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

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Lütfen önce bir görüntü yükleyin.")
        else:
            img_bgr = load_uploaded_image(uploaded_file)
            if img_bgr is None:
                logger.error("Görüntü okunamadı: %s", uploaded_file.name)
                st.error("Görüntü okunamadı. Geçerli bir JPG/JPEG dosyası yükleyin.")
            else:
                wbc_param = wbc_min_area if dev_mode else None
                ws_param = watershed_coeff if dev_mode else None
                try:
                    with st.spinner("Analiz yapılıyor..."):
                        source_stem = Path(uploaded_file.name).stem
                        wbc, rbc, wbc_avg, rbc_avg, original_rgb, final_rgb = (
                            analyze_blood_image(
                                img_bgr,
                                source_stem=source_stem,
                                wbc_min_area=wbc_param,
                                watershed_thresh_coeff=ws_param,
                            )
                        )
                except Exception as exc:
                    logger.error(
                        "Analiz hatası (%s): %s", uploaded_file.name, exc
                    )
                    st.error(f"Analiz sırasında hata oluştu: {exc}")
                else:
                    if dev_mode:
                        st.info(
                            f"Dev Mode aktif — WBC min alan: {wbc_param}, "
                            f"Watershed: {ws_param}"
                        )
                    logger.info(
                        "Analiz tamamlandı: %s -> WBC: %s, RBC: %s",
                        uploaded_file.name,
                        wbc,
                        rbc,
                    )
                    st.session_state["results"] = {
                        "wbc": wbc,
                        "rbc": rbc,
                        "wbc_avg_area": wbc_avg,
                        "rbc_avg_area": rbc_avg,
                        "original": original_rgb,
                        "final": final_rgb,
                        "filename": uploaded_file.name,
                        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source_stem": source_stem,
                        "dev_mode": dev_mode,
                        "wbc_min_area_param": wbc_param or WBC_MIN_AREA,
                        "watershed_coeff_param": ws_param or WATERSHED_THRESH_COEFF,
                    }
                    build_wbc_dataset_zip.clear()

    results = st.session_state.get("results")

    if results:
        if results.get("dev_mode"):
            st.caption(
                "⚙️ Son analiz Gelişmiş Ayarlar ile çalıştırıldı "
                f"(WBC min: {results.get('wbc_min_area_param')}, "
                f"Watershed: {results.get('watershed_coeff_param')})."
            )

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
        st.caption("Morfolojik alan birimi: piksel² (px)")

        col_orig, col_final = st.columns(2)
        with col_orig:
            st.subheader("Orijinal Görüntü")
            st.image(results["original"], use_container_width=True)
        with col_final:
            st.subheader("Analiz Sonucu (Final)")
            st.image(
                results["final"],
                use_container_width=True,
                caption=f"{results['wbc']} WBC | {results['rbc']} RBC",
            )

        source_stem = results.get(
            "source_stem", Path(results["filename"]).stem
        )
        render_wbc_gallery(source_stem)

        st.divider()
        st.subheader("Hücre Dağılımı")
        chart_col, action_col = st.columns([2, 1])
        with chart_col:
            st.plotly_chart(
                build_distribution_chart(results["wbc"], results["rbc"]),
                use_container_width=True,
            )
        with action_col:
            st.markdown("##### Dışa Aktar")
            st.caption("Analiz sonucunu CSV olarak indirin.")
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

    else:
        st.info(
            "Sol menüden bir kan hücresi görüntüsü yükleyip **Analizi Başlat** "
            "butonuna basarak analizi başlatın."
        )


if __name__ == "__main__":
    main()
