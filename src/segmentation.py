import os

import cv2
import numpy as np

from config import (
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    ADAPTIVE_THRESH_MAX_VAL,
    BORDER_MARGIN,
    CROPPED_DIR,
    DIST_TRANSFORM_MASK_SIZE,
    SAVE_CROPPED_WBC,
    RBC_CIRCLE_COLOR,
    RBC_CIRCLE_RADIUS,
    RBC_DILATE_ITERATIONS,
    RBC_MIN_AREA,
    RBC_MORPH_KERNEL_SIZE,
    RBC_MORPH_OPEN_ITERATIONS,
    WATERSHED_THRESH_COEFF,
    WATERSHED_THRESH_MAX_VAL,
    WBC_FONT_SCALE,
    WBC_FONT_THICKNESS,
    WBC_LOWER_PURPLE,
    WBC_MIN_AREA,
    WBC_MORPH_KERNEL_SIZE,
    WBC_RECT_COLOR,
    WBC_RECT_THICKNESS,
    WBC_UPPER_PURPLE,
)


def _average_area(areas: list[float]) -> float:
    """Geçerli hücre alanlarının ortalaması; hücre yoksa 0."""
    if not areas:
        return 0.0
    return round(sum(areas) / len(areas), 2)


def _cell_record(cell_type: str, area: float) -> dict[str, str | float]:
    """Tek hücre için detay satırı."""
    return {
        "Hücre Tipi": cell_type,
        "Alan (px)": round(float(area), 2),
        "Durum": "Tam",
    }


def _is_border_cell(
    x: int,
    y: int,
    w: int,
    h: int,
    cx: float,
    cy: float,
    img_h: int,
    img_w: int,
    margin: int = BORDER_MARGIN,
) -> bool:
    """Kontur veya merkezi görüntü kenarına çok yakınsa True döner."""
    if x <= margin or y <= margin:
        return True
    if x + w >= img_w - margin or y + h >= img_h - margin:
        return True
    if cx <= margin or cy <= margin:
        return True
    if cx >= img_w - margin or cy >= img_h - margin:
        return True
    return False


def _save_wbc_crop(
    img_bgr: np.ndarray, x: int, y: int, w: int, h: int, source_stem: str, index: int
) -> None:
    """Geçerli WBC bölgesini orijinal görüntüden kırpıp diske kaydeder (ML veri seti)."""
    os.makedirs(CROPPED_DIR, exist_ok=True)
    crop = img_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return
    out_path = os.path.join(CROPPED_DIR, f"{source_stem}_WBC_{index}.jpg")
    cv2.imwrite(out_path, crop)


def count_wbc(
    blurred: np.ndarray,
    output: np.ndarray,
    img_bgr: np.ndarray | None = None,
    source_stem: str | None = None,
    wbc_min_area: int | None = None,
) -> tuple[int, float, np.ndarray, list[dict[str, str | float]]]:
    """HSV renk uzayı ve alan filtreleme ile akyuvar tespiti."""
    min_area = WBC_MIN_AREA if wbc_min_area is None else wbc_min_area
    img_h, img_w = output.shape[:2]
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_purple = np.array(WBC_LOWER_PURPLE)
    upper_purple = np.array(WBC_UPPER_PURPLE)
    mask_wbc = cv2.inRange(hsv, lower_purple, upper_purple)
    cleaned_wbc = cv2.morphologyEx(
        mask_wbc,
        cv2.MORPH_OPEN,
        np.ones(WBC_MORPH_KERNEL_SIZE, np.uint8),
    )
    wbc_contours, _ = cv2.findContours(
        cleaned_wbc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    wbc_count = 0
    wbc_areas: list[float] = []
    cell_records: list[dict[str, str | float]] = []
    wbc_crop_index = 0
    save_crops = SAVE_CROPPED_WBC and img_bgr is not None and source_stem

    for cnt in wbc_contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        if _is_border_cell(x, y, w, h, cx, cy, img_h, img_w):
            continue

        wbc_count += 1
        wbc_areas.append(area)
        cell_records.append(_cell_record("WBC", area))
        wbc_crop_index += 1
        if save_crops:
            _save_wbc_crop(img_bgr, x, y, w, h, source_stem, wbc_crop_index)
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            WBC_RECT_COLOR,
            WBC_RECT_THICKNESS,
        )
        cv2.putText(
            output,
            "WBC",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            WBC_FONT_SCALE,
            WBC_RECT_COLOR,
            WBC_FONT_THICKNESS,
        )

    return wbc_count, _average_area(wbc_areas), output, cell_records


def count_rbc_watershed(
    img_bgr: np.ndarray,
    blurred: np.ndarray,
    output: np.ndarray,
    watershed_thresh_coeff: float | None = None,
) -> tuple[int, float, np.ndarray, list[dict[str, str | float]]]:
    """Adaptive Threshold, Distance Transform ve Watershed ile alyuvar sayımı."""
    thresh_coeff = (
        WATERSHED_THRESH_COEFF
        if watershed_thresh_coeff is None
        else watershed_thresh_coeff
    )
    img_h, img_w = output.shape[:2]
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        ADAPTIVE_THRESH_MAX_VAL,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_C,
    )

    kernel = np.ones(RBC_MORPH_KERNEL_SIZE, np.uint8)
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=RBC_MORPH_OPEN_ITERATIONS
    )
    sure_bg = cv2.dilate(opening, kernel, iterations=RBC_DILATE_ITERATIONS)

    dist_transform = cv2.distanceTransform(
        opening, cv2.DIST_L2, DIST_TRANSFORM_MASK_SIZE
    )
    ret, sure_fg = cv2.threshold(
        dist_transform,
        thresh_coeff * dist_transform.max(),
        WATERSHED_THRESH_MAX_VAL,
        0,
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)

    rbc_count = 0
    rbc_areas: list[float] = []
    cell_records: list[dict[str, str | float]] = []

    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) == 0:
            continue

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area <= RBC_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)

        if _is_border_cell(x, y, w, h, cx, cy, img_h, img_w):
            continue

        rbc_count += 1
        rbc_areas.append(area)
        cell_records.append(_cell_record("RBC", area))
        cv2.circle(
            output,
            (int(cx), int(cy)),
            RBC_CIRCLE_RADIUS,
            RBC_CIRCLE_COLOR,
            -1,
        )

    return rbc_count, _average_area(rbc_areas), output, cell_records
