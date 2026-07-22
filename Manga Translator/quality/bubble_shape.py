import logging
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from core.models import BoundingBox

logger = logging.getLogger("quality.bubble_shape")


class BubbleShape(str, Enum):
    RECTANGLE = "RECTANGLE"
    OVAL = "OVAL"
    SPIKY = "SPIKY"
    CLOUD = "CLOUD"
    UNKNOWN = "UNKNOWN"


def detect_bubble_contour(page_image: np.ndarray, bbox: BoundingBox, padding: int = 15) -> np.ndarray:
    """
    Crops a small region around the text bbox and isolates the bubble's
    fill area via thresholding. Use the INPAINTED page image —
    inpainting removes text inside the bubble, leaving the border intact.
    """
    if cv2 is None:
        raise ImportError("opencv-python-headless is not installed.")

    h, w = page_image.shape[:2]
    x0 = max(0, bbox.x_min - padding)
    y0 = max(0, bbox.y_min - padding)
    x1 = min(w, bbox.x_max + padding)
    y1 = min(h, bbox.y_max + padding)
    
    crop = page_image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def classify_bubble_shape(mask: np.ndarray) -> BubbleShape:
    """
    Classifies bubble shapes into categories: RECTANGLE, OVAL, SPIKY, CLOUD, or UNKNOWN
    using contour mathematics on the thresholded crop mask.
    """
    if cv2 is None or mask.size <= 1:
        return BubbleShape.UNKNOWN

    try:
        # Find contours of the foreground (typically white/255 areas in thresholded mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BubbleShape.UNKNOWN

        # Select the largest contour, which is likely the bubble boundary
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 100:  # Too small to classify reliably
            return BubbleShape.UNKNOWN

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            return BubbleShape.UNKNOWN

        # Convex hull metrics
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Circularity: 4 * pi * area / perimeter^2
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Rectangularity: area / min_bounding_rect_area
        _, (rw, rh), _ = cv2.minAreaRect(c)
        rect_area = rw * rh
        rectangularity = area / rect_area if rect_area > 0 else 0.0

        # Count convexity defects to detect spiky/cloud shapes
        hull_indices = cv2.convexHull(c, returnPoints=False)
        defects_count = 0
        if len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(c, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        # d is distance in 1/256 of a pixel
                        if d / 256.0 > 4.0:
                            defects_count += 1
            except Exception:
                pass

        logger.debug(
            "[BubbleShape] solidity=%.3f, circularity=%.3f, rectangularity=%.3f, defects=%d",
            solidity, circularity, rectangularity, defects_count
        )

        # Heuristic rules
        # Spiky bubbles have sharp corners and low solidity, many convexity defects
        if defects_count >= 8 or (solidity < 0.85 and defects_count >= 5):
            return BubbleShape.SPIKY

        # Cloud bubbles have rounded bumps and medium/high solidity with some convexity defects
        if defects_count >= 3:
            return BubbleShape.CLOUD

        # Rectangle bubbles have high rectangularity
        if rectangularity > 0.82:
            return BubbleShape.RECTANGLE

        # Oval bubbles have high circularity
        if circularity > 0.72:
            return BubbleShape.OVAL

        # Secondary logic based on best fit
        if rectangularity > circularity:
            return BubbleShape.RECTANGLE
        elif circularity > 0.55:
            return BubbleShape.OVAL

        return BubbleShape.UNKNOWN

    except Exception as exc:
        logger.warning("Failed to classify bubble shape: %s", exc)
        return BubbleShape.UNKNOWN
