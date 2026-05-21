import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_REPORT,
    DEFAULT_PROCESSED_DIR,
)
from logging_setup import get_logger, setup_logging
from preprocessing import apply_clahe_and_blur, get_sharpness_score, is_image_blurry
from segmentation import count_wbc, count_rbc_watershed

logger = get_logger(__name__)


def default_worker_count() -> int:
    """Varsayılan paralel işçi: CPU çekirdeği - 1 (en az 1)."""
    return max(1, (os.cpu_count() or 2) - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BCCD mikroskop görüntülerinden kan hücresi (WBC/RBC) sayımı."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help=f"İşlenecek görüntü klasörü (varsayılan: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_REPORT,
        help=f"Excel rapor yolu (varsayılan: {DEFAULT_OUTPUT_REPORT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="İşlenecek maksimum resim sayısı (varsayılan: tümü)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="İşlenmiş (annotasyonlu) görüntüleri output/processed_images/ altına kaydet",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Paralel işçi sayısı (varsayılan: {default_worker_count()})",
    )
    return parser.parse_args()


def _save_processed_image(
    final_output: np.ndarray, filename: str, images_dir: str
) -> None:
    """Annotasyonlu final görüntüyü diske kaydeder."""
    os.makedirs(images_dir, exist_ok=True)
    stem, _ = os.path.splitext(filename)
    out_path = os.path.join(images_dir, f"{stem}_final.jpg")
    out_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)


def _batch_worker(
    args: tuple[str, bool, str],
) -> dict[str, str | int | float] | None:
    """
    Tek görüntüyü işler (ProcessPoolExecutor için üst düzey fonksiyon).
    Loglama ana thread'de yapılır; burada sadece sonuç döner.
    """
    image_path, save_images, processed_images_dir = args

    try:
        if not os.path.exists(image_path):
            return None

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness_score = get_sharpness_score(img_gray)

        blurred = apply_clahe_and_blur(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        final_output = img_rgb.copy()

        wbc_count, wbc_avg_area, final_output = count_wbc(
            blurred, final_output, img_bgr=img_bgr, source_stem=stem
        )
        rbc_count, rbc_avg_area, final_output = count_rbc_watershed(
            img_bgr, blurred, final_output
        )

        if save_images:
            _save_processed_image(final_output, filename, processed_images_dir)

        total_cells = wbc_count + rbc_count
        return {
            "Dosya Adı": filename,
            "WBC Sayısı": wbc_count,
            "RBC Sayısı": rbc_count,
            "Toplam Hücre Sayısı": total_cells,
            "Ortalama WBC Alanı (px)": wbc_avg_area,
            "Ortalama RBC Alanı (px)": rbc_avg_area,
            "Netlik Skoru": sharpness_score,
            "_blurry": is_image_blurry(img_gray),
        }
    except Exception:
        return None


class BloodCellAnalyzer:
    """BCCD mikroskop görüntülerinde WBC ve RBC sayımı (orchestrator)."""

    def _process_image(
        self, image_path: str
    ) -> tuple[int, int, float, float, float, np.ndarray, np.ndarray] | None:
        """Tek görüntü için sayım, morfolojik metrikler ve annotasyonlu çıktı."""
        if not os.path.exists(image_path):
            return None

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness_score = get_sharpness_score(img_gray)
        if is_image_blurry(img_gray):
            logger.warning(
                "[UYARI] %s bulanık! (Skor: %s)", filename, sharpness_score
            )

        blurred = apply_clahe_and_blur(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        final_output = img_rgb.copy()

        wbc_count, wbc_avg_area, final_output = count_wbc(
            blurred, final_output, img_bgr=img_bgr, source_stem=stem
        )
        rbc_count, rbc_avg_area, final_output = count_rbc_watershed(
            img_bgr, blurred, final_output
        )
        return (
            wbc_count,
            rbc_count,
            wbc_avg_area,
            rbc_avg_area,
            sharpness_score,
            img_rgb,
            final_output,
        )

    def _count_cells(
        self, image_path: str
    ) -> tuple[int, int, float, float] | None:
        """Tek görüntü için WBC/RBC sayımı ve ortalama alan (görselleştirme yok)."""
        result = self._process_image(image_path)
        if result is None:
            return None
        wbc_count, rbc_count, wbc_avg_area, rbc_avg_area, _, _, _ = result
        return wbc_count, rbc_count, wbc_avg_area, rbc_avg_area

    def analyze_image(self, image_path: str) -> None:
        """Tek bir görüntüyü işler, sonuçları terminale yazar ve görselleştirir."""
        try:
            result = self._process_image(image_path)
        except Exception as exc:
            logger.error("Görüntü işlenemedi: %s — %s", image_path, exc)
            return

        if result is None:
            logger.error("HATA: %s bulunamadı veya okunamadı!", image_path)
            return

        wbc_count, rbc_count, wbc_avg_area, rbc_avg_area, sharpness, img_rgb, final_output = (
            result
        )
        logger.info("Netlik skoru: %s", sharpness)

        logger.info("=" * 40)
        logger.info("AKYUVAR (WBC) SAYISI: %s", wbc_count)
        logger.info("ORTALAMA WBC ALANI: %s px", wbc_avg_area)
        logger.info("ALYUVAR (RBC) SAYISI: %s", rbc_count)
        logger.info("ORTALAMA RBC ALANI: %s px", rbc_avg_area)
        logger.info("=" * 40)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Orijinal")
        axes[0].axis("off")

        axes[1].imshow(final_output)
        axes[1].set_title(f"Final: {wbc_count} WBC | {rbc_count} RBC")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _list_image_paths(folder_path: str) -> list[str]:
        """Klasördeki .jpg/.jpeg dosyalarını alfabetik/sayısal sırada döndürür."""
        patterns = [
            os.path.join(folder_path, "*.jpg"),
            os.path.join(folder_path, "*.jpeg"),
            os.path.join(folder_path, "*.JPG"),
            os.path.join(folder_path, "*.JPEG"),
        ]
        image_paths: list[str] = []
        for pattern in patterns:
            image_paths.extend(glob.glob(pattern))
        return sorted(set(image_paths))

    @staticmethod
    def _build_summary_df(df: pd.DataFrame) -> pd.DataFrame:
        """WBC, RBC ve toplam hücre için ortalama, max ve min özeti."""
        return pd.DataFrame(
            {
                "İstatistik": [
                    "Ortalama (Mean)",
                    "En Yüksek (Max)",
                    "En Düşük (Min)",
                ],
                "WBC": [
                    df["WBC Sayısı"].mean(),
                    df["WBC Sayısı"].max(),
                    df["WBC Sayısı"].min(),
                ],
                "RBC": [
                    df["RBC Sayısı"].mean(),
                    df["RBC Sayısı"].max(),
                    df["RBC Sayısı"].min(),
                ],
                "Toplam Hücre": [
                    df["Toplam Hücre Sayısı"].mean(),
                    df["Toplam Hücre Sayısı"].max(),
                    df["Toplam Hücre Sayısı"].min(),
                ],
            }
        )

    @staticmethod
    def _save_report(
        df: pd.DataFrame,
        summary_df: pd.DataFrame,
        output_path: str = DEFAULT_OUTPUT_REPORT,
    ) -> str:
        """DataFrame'i Excel'e kaydeder; openpyxl yoksa CSV'ye düşer."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        xlsx_path = (
            output_path
            if output_path.endswith(".xlsx")
            else output_path.replace(".csv", ".xlsx")
        )

        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Sayim Sonuclari", index=False)
                summary_df.to_excel(writer, sheet_name="Ozet Istatistik", index=False)
            return xlsx_path
        except (ImportError, ValueError, ModuleNotFoundError) as exc:
            logger.warning("Excel kaydı başarısız, CSV'ye düşülüyor: %s", exc)
            csv_path = xlsx_path.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            summary_path = csv_path.replace(".csv", "_ozet.csv")
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            return csv_path

    def analyze_batch(
        self,
        folder_path: str,
        limit: int | None = None,
        output_path: str = DEFAULT_OUTPUT_REPORT,
        save_images: bool = False,
        processed_images_dir: str = DEFAULT_PROCESSED_DIR,
        workers: int | None = None,
    ) -> pd.DataFrame | None:
        """Klasördeki görüntüleri paralel işler, sonuçları Excel/CSV raporuna yazar."""
        if not os.path.isdir(folder_path):
            logger.error("HATA: Klasör bulunamadı: %s", folder_path)
            return None

        image_paths = self._list_image_paths(folder_path)
        if not image_paths:
            logger.error(
                "HATA: %s içinde .jpg/.jpeg dosyası bulunamadı!", folder_path
            )
            return None

        if limit is not None:
            image_paths = image_paths[:limit]

        if workers is None:
            workers = default_worker_count()

        total = len(image_paths)
        logger.info("Toplu işlem başlıyor: %s görüntü", total)
        logger.info("Girdi klasörü: %s", folder_path)
        logger.info("Rapor çıktısı: %s", output_path)
        logger.info("Paralel işçi sayısı: %s", workers)
        if save_images:
            logger.info("Görüntü kaydı: %s", processed_images_dir)

        worker_args = [
            (path, save_images, processed_images_dir) for path in image_paths
        ]
        results: list[dict[str, str | int | float]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(_batch_worker, arg): arg[0] for arg in worker_args
            }
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                filename = os.path.basename(image_path)
                try:
                    row = future.result()
                except Exception as exc:
                    logger.error(
                        "[HATA] %s işlenemedi, atlanıyor... Sebep: %s",
                        filename,
                        exc,
                    )
                    continue

                if row is None:
                    logger.error(
                        "[HATA] %s işlenemedi, atlanıyor... (dosya okunamadı)",
                        filename,
                    )
                    continue

                if row.pop("_blurry", False):
                    logger.warning(
                        "[UYARI] %s bulanık! (Skor: %s)",
                        filename,
                        row["Netlik Skoru"],
                    )

                logger.info(
                    "[İŞLENİYOR] %s -> WBC: %s, RBC: %s | Ort. Alan: WBC=%s px, "
                    "RBC=%s px | Netlik: %s",
                    filename,
                    row["WBC Sayısı"],
                    row["RBC Sayısı"],
                    row["Ortalama WBC Alanı (px)"],
                    row["Ortalama RBC Alanı (px)"],
                    row["Netlik Skoru"],
                )
                results.append(
                    {k: v for k, v in row.items() if not k.startswith("_")}
                )

        if not results:
            logger.error(
                "HATA: İşlenebilir görüntü bulunamadı, rapor oluşturulmadı."
            )
            return None

        results.sort(key=lambda r: r["Dosya Adı"])
        df = pd.DataFrame(results)
        summary_df = self._build_summary_df(df)
        saved_path = self._save_report(df, summary_df, output_path)

        logger.info("Toplu işlem tamamlandı: %s görüntü işlendi.", len(results))
        logger.info("Rapor kaydedildi: %s", saved_path)
        if save_images:
            logger.info("İşlenmiş görüntüler kaydedildi: %s", processed_images_dir)
        return df


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    workers = args.workers if args.workers is not None else default_worker_count()
    analyzer = BloodCellAnalyzer()
    analyzer.analyze_batch(
        folder_path=args.input,
        limit=args.limit,
        output_path=args.output,
        save_images=args.save_images,
        workers=workers,
    )
