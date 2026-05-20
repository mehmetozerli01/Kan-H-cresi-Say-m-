"""
Kan Hücresi Sayımı — Streamlit Web Dashboard
Mevcut preprocessing ve segmentation modüllerini kullanır.
"""

import sys
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

from logging_setup import get_logger, setup_logging
from preprocessing import apply_clahe_and_blur
from segmentation import count_wbc, count_rbc_watershed

logger = get_logger(__name__)


def analyze_blood_image(img_bgr: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Yüklenen BGR görüntü üzerinde WBC/RBC sayımı ve annotasyon."""
    blurred = apply_clahe_and_blur(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    final_output = img_rgb.copy()

    wbc_count, final_output = count_wbc(blurred, final_output)
    rbc_count, final_output = count_rbc_watershed(img_bgr, blurred, final_output)
    return wbc_count, rbc_count, img_rgb, final_output


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

    return pd.DataFrame(
        [
            {"Alan": "Dosya Adı", "Değer": results["filename"]},
            {"Alan": "Analiz Tarihi/Saati", "Değer": results["analyzed_at"]},
            {"Alan": "WBC (Akyuvar) Sayısı", "Değer": str(wbc)},
            {"Alan": "RBC (Alyuvar) Sayısı", "Değer": str(rbc)},
            {"Alan": "Toplam Hücre Sayısı", "Değer": str(total)},
            {"Alan": "WBC Oranı (%)", "Değer": f"{wbc_pct}"},
            {"Alan": "RBC Oranı (%)", "Değer": f"{rbc_pct}"},
        ]
    )


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
            }
        ]
    )
    return export_df.to_csv(index=False).encode("utf-8-sig")


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

    st.markdown('<p class="main-header">🔬 Kan Hücresi Sayım Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">BCCD mikroskop görüntülerinden akyuvar (WBC) ve alyuvar (RBC) '
        "tespitini CLAHE, HSV segmentasyonu ve Watershed algoritmalarıyla gerçekleştirir.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("📁 Görüntü Yükle")
        st.caption("JPG veya JPEG formatında kan yayması görüntüsü seçin.")
        uploaded_file = st.file_uploader(
            "Dosya seçin",
            type=["jpg", "jpeg"],
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("Analizi Başlat", type="primary", use_container_width=True)

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Lütfen önce bir görüntü yükleyin.")
        else:
            img_bgr = load_uploaded_image(uploaded_file)
            if img_bgr is None:
                logger.error(
                    "Görüntü okunamadı: %s", uploaded_file.name
                )
                st.error("Görüntü okunamadı. Geçerli bir JPG/JPEG dosyası yükleyin.")
            else:
                try:
                    with st.spinner("Analiz yapılıyor..."):
                        wbc, rbc, original_rgb, final_rgb = analyze_blood_image(
                            img_bgr
                        )
                except Exception as exc:
                    logger.error(
                        "Analiz hatası (%s): %s", uploaded_file.name, exc
                    )
                    st.error(f"Analiz sırasında hata oluştu: {exc}")
                else:
                    logger.info(
                        "Analiz tamamlandı: %s -> WBC: %s, RBC: %s",
                        uploaded_file.name,
                        wbc,
                        rbc,
                    )
                    st.session_state["results"] = {
                        "wbc": wbc,
                        "rbc": rbc,
                        "original": original_rgb,
                        "final": final_rgb,
                        "filename": uploaded_file.name,
                        "analyzed_at": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }

    results = st.session_state.get("results")

    if results:
        col_wbc, col_rbc, col_total = st.columns(3)
        total_cells = results["wbc"] + results["rbc"]
        col_wbc.metric("WBC Sayısı", results["wbc"])
        col_rbc.metric("RBC Sayısı", results["rbc"])
        col_total.metric("Toplam Hücre", total_cells)

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
