import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

from config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_REPORT,
    DEFAULT_PROCESSED_DIR,
)
from logging_setup import get_logger, setup_logging
from preprocessing import apply_clahe_and_blur
from segmentation import count_wbc, count_rbc_watershed

logger = get_logger(__name__)


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
    return parser.parse_args()


class BloodCellAnalyzer:
    """BCCD mikroskop görüntülerinde WBC ve RBC sayımı (orchestrator)."""

    def _process_image(
        self, image_path: str
    ) -> tuple[int, int, np.ndarray, np.ndarray] | None:
        """Tek görüntü için sayım ve annotasyonlu çıktı."""
        if not os.path.exists(image_path):
            return None

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        blurred = apply_clahe_and_blur(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        final_output = img_rgb.copy()

        wbc_count, final_output = count_wbc(blurred, final_output)
        rbc_count, final_output = count_rbc_watershed(
            img_bgr, blurred, final_output
        )
        return wbc_count, rbc_count, img_rgb, final_output

    def _count_cells(self, image_path: str) -> tuple[int, int] | None:
        """Tek görüntü için WBC/RBC sayımı (görselleştirme yok)."""
        result = self._process_image(image_path)
        if result is None:
            return None
        wbc_count, rbc_count, _, _ = result
        return wbc_count, rbc_count

    @staticmethod
    def _save_processed_image(
        final_output: np.ndarray, filename: str, images_dir: str
    ) -> None:
        """Annotasyonlu final görüntüyü diske kaydeder."""
        os.makedirs(images_dir, exist_ok=True)
        stem, _ = os.path.splitext(filename)
        out_path = os.path.join(images_dir, f"{stem}_final.jpg")
        out_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr)

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

        wbc_count, rbc_count, img_rgb, final_output = result

        logger.info("=" * 40)
        logger.info("AKYUVAR (WBC) SAYISI: %s", wbc_count)
        logger.info("ALYUVAR (RBC) SAYISI: %s", rbc_count)
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
    ) -> pd.DataFrame | None:
        """Klasördeki görüntüleri işler, sonuçları Excel/CSV raporuna yazar."""
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

        total = len(image_paths)
        logger.info("Toplu işlem başlıyor: %s görüntü", total)
        logger.info("Girdi klasörü: %s", folder_path)
        logger.info("Rapor çıktısı: %s", output_path)
        if save_images:
            logger.info("Görüntü kaydı: %s", processed_images_dir)

        results: list[dict[str, str | int]] = []

        for image_path in image_paths:
            filename = os.path.basename(image_path)

            try:
                if save_images:
                    result = self._process_image(image_path)
                    if result is None:
                        logger.error(
                            "[HATA] %s işlenemedi, atlanıyor... (dosya okunamadı)",
                            filename,
                        )
                        continue
                    wbc_count, rbc_count, _, final_output = result
                    self._save_processed_image(
                        final_output, filename, processed_images_dir
                    )
                else:
                    counts = self._count_cells(image_path)
                    if counts is None:
                        logger.error(
                            "[HATA] %s işlenemedi, atlanıyor... (dosya okunamadı)",
                            filename,
                        )
                        continue
                    wbc_count, rbc_count = counts

            except Exception as exc:
                logger.error(
                    "[HATA] %s işlenemedi, atlanıyor... Sebep: %s",
                    filename,
                    exc,
                )
                continue

            total_cells = wbc_count + rbc_count
            logger.info(
                "[İŞLENİYOR] %s -> WBC: %s, RBC: %s",
                filename,
                wbc_count,
                rbc_count,
            )
            results.append(
                {
                    "Dosya Adı": filename,
                    "WBC Sayısı": wbc_count,
                    "RBC Sayısı": rbc_count,
                    "Toplam Hücre Sayısı": total_cells,
                }
            )

        if not results:
            logger.error(
                "HATA: İşlenebilir görüntü bulunamadı, rapor oluşturulmadı."
            )
            return None

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
    analyzer = BloodCellAnalyzer()
    analyzer.analyze_batch(
        folder_path=args.input,
        limit=args.limit,
        output_path=args.output,
        save_images=args.save_images,
    )
