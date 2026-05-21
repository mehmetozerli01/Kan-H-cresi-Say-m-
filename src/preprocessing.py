import cv2
import numpy as np

from config import (
    BLUR_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_BLUR_KSIZE,
)

_CLAHE = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
)


def get_sharpness_score(img_gray: np.ndarray) -> float:
    """Laplacian varyans ile netlik skoru (yüksek = daha net)."""
    return round(float(cv2.Laplacian(img_gray, cv2.CV_64F).var()), 2)


def is_image_blurry(img_gray: np.ndarray) -> bool:
    """Skor BLUR_THRESHOLD altındaysa görüntü bulanık kabul edilir."""
    return get_sharpness_score(img_gray) < BLUR_THRESHOLD


def apply_clahe_and_blur(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE ve Median Blur ile görüntü iyileştirme."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = _CLAHE.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.medianBlur(enhanced_bgr, MEDIAN_BLUR_KSIZE)
