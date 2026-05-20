import cv2
import numpy as np

from config import (
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    ADAPTIVE_THRESH_MAX_VAL,
    DIST_TRANSFORM_MASK_SIZE,
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


def count_wbc(
    blurred: np.ndarray, output: np.ndarray
) -> tuple[int, np.ndarray]:
    """HSV renk uzayı ve alan filtreleme ile akyuvar tespiti."""
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
    for cnt in wbc_contours:
        if cv2.contourArea(cnt) > WBC_MIN_AREA:
            wbc_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
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
    return wbc_count, output


def count_rbc_watershed(
    img_bgr: np.ndarray, blurred: np.ndarray, output: np.ndarray
) -> tuple[int, np.ndarray]:
    """Adaptive Threshold, Distance Transform ve Watershed ile alyuvar sayımı."""
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
        WATERSHED_THRESH_COEFF * dist_transform.max(),
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
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > RBC_MIN_AREA:
                rbc_count += 1
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(
                    output,
                    (int(cx), int(cy)),
                    RBC_CIRCLE_RADIUS,
                    RBC_CIRCLE_COLOR,
                    -1,
                )

    return rbc_count, output
