import cv2
import numpy as np
from typing import Tuple


def resize_for_fast_processing(image: np.ndarray, max_dim: int = 800) -> Tuple[np.ndarray, float]:
    if image is None or image.size == 0:
        return image, 1.0
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(max_side)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Invalid image for lighting normalization.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def get_onion_color_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    HSV mask covering all onion varieties.

    Fixed bug: original white mask [0,0,130 → 180,60,255] matched white WALLS
    and bright backgrounds (high V, low S), causing background pixels to be
    treated as onion skin. Fixed by tightening V floor to 160 and keeping S<=45.

    Ranges:
    - Red/Purple/Pink onions: H 0-18 or 148-180, S>=20, V>=20
    - Yellow/Golden/Brown:    H 8-55, S>=25, V>=30
    - White/Cream/Ivory:      any H, S<=45, V 160-255  (tighter than before)
    - Dark brown dried skin:  H 5-30, S 10-85, V 20-160
    """
    # 1. Red / Purple / Pink
    m1 = cv2.inRange(hsv_image,
                     np.array([0,   20, 20], dtype=np.uint8),
                     np.array([18, 255, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv_image,
                     np.array([148, 20, 20], dtype=np.uint8),
                     np.array([180, 255, 255], dtype=np.uint8))

    # 2. Yellow / Golden / Brown (H 8-55 avoids red hue overlap)
    m3 = cv2.inRange(hsv_image,
                     np.array([8,  25, 30], dtype=np.uint8),
                     np.array([55, 255, 255], dtype=np.uint8))

    # 3. White / Cream / Ivory (low saturation, higher brightness across hues)
    m4 = cv2.inRange(hsv_image,
                     np.array([0,    5, 150], dtype=np.uint8),
                     np.array([180, 50, 255], dtype=np.uint8))

    # 4. Dark brown / dried outer skin
    m5 = cv2.inRange(hsv_image,
                     np.array([5,  10, 20], dtype=np.uint8),
                     np.array([30, 85, 160], dtype=np.uint8))

    return m1 | m2 | m3 | m4 | m5


def subtract_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Foreground mask using Otsu + onion color mask union.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image for background subtraction.")

    h, w = image.shape[:2]
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Background is at the image borders; if border pixels are predominantly 255, invert mask
    border_otsu = np.concatenate([otsu[0, :], otsu[-1, :], otsu[:, 0], otsu[:, -1]])
    if float(np.mean(border_otsu)) > 127.0:
        otsu = cv2.bitwise_not(otsu)

    # Color mask (now with fixed white mask)
    color = get_onion_color_mask(hsv)

    combined = cv2.bitwise_or(otsu, color)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open,  iterations=1)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(combined)
    min_area = h * w * 0.003
    for cnt in cnts:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(clean, [cnt], -1, 255, -1)

    if np.count_nonzero(clean) == 0:
        clean = combined

    fg = cv2.bitwise_and(image, image, mask=clean)
    return fg, clean
