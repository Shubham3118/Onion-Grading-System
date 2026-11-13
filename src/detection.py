import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception as e:
    YOLO = None
    HAS_YOLO = False
    print(f"Notice: YOLO not available ({e}). Using OpenCV shape-based detection.")

from src.preprocessing import get_onion_color_mask


class OnionDetectorStub:
    """
    Shape-first onion detection pipeline.
    Detects any round/oval blob in the image regardless of color.
    Only uses color as a final soft check, never as a hard reject.
    """

    def __init__(self, model_weights: str = "yolov8n-seg.pt"):
        self.model: Optional[Any] = None
        if HAS_YOLO:
            try:
                self.model = YOLO(model_weights)
            except Exception as e:
                print(f"YOLOv8 load failed ({e}). Using contour mode.")
                self.model = None

    # ─── shape validators ─────────────────────────────────────────────────────

    def _is_round_enough(self, contour: np.ndarray) -> bool:
        area = float(cv2.contourArea(contour))
        if area < 200:
            return False
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter == 0:
            return False
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / float(h) if h > 0 else 0.0
        # Very relaxed: circularity >= 0.15, aspect 0.25–3.0
        return circularity >= 0.15 and 0.25 <= aspect <= 3.0

    def _touches_border(self, contour: np.ndarray, img_shape: Tuple) -> bool:
        """Reject only blobs that fill the entire frame edge-to-edge (99%+).
        Previously 94% was too aggressive — it rejected real close-up onion images."""
        h, w = img_shape[:2]
        x, y, cw, ch = cv2.boundingRect(contour)
        return (cw >= w * 0.99) or (ch >= h * 0.99)

    # ─── main detect entry ────────────────────────────────────────────────────

    def detect_and_measure(self, image: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:

        # YOLO path
        if self.model is not None:
            try:
                results = self.model(image, verbose=False)
                items = []
                item_id = 1
                for r in results:
                    if r.boxes is not None and r.masks is not None:
                        for box, mdata in zip(r.boxes, r.masks.data):
                            if int(box.cls[0].cpu().numpy()) == 0:
                                continue
                            mnp = (mdata.cpu().numpy() * 255).astype(np.uint8)
                            cnts, _ = cv2.findContours(mnp, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                            if cnts:
                                cnt = max(cnts, key=cv2.contourArea)
                                area = float(cv2.contourArea(cnt))
                                (_, _), r2 = cv2.minEnclosingCircle(cnt)
                                x, y, w, h = cv2.boundingRect(cnt)
                                item_m = np.zeros(image.shape[:2], dtype=np.uint8)
                                cv2.drawContours(item_m, [cnt], -1, 255, -1)
                                items.append({"item_id": item_id,
                                              "pixel_area": area,
                                              "max_diameter": float(r2 * 2),
                                              "bbox": (int(x), int(y), int(w), int(h)),
                                              "contour": cnt,
                                              "item_mask": item_m})
                                item_id += 1
                if items:
                    return items
            except Exception as e:
                print(f"YOLO inference failed: {e}")

        return self._opencv_detect(image, mask)

    # ─── OpenCV shape-first pipeline ──────────────────────────────────────────

    def _opencv_detect(self, image: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Tries multiple segmentation strategies in order.
        Returns as soon as any strategy finds valid blobs.
        """
        h, w = image.shape[:2]
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        strategies = [
            ("provided_mask",  lambda: mask),
            ("otsu",           lambda: self._otsu_mask(blurred, gray)),
            ("adaptive",       lambda: self._adaptive_mask(blurred)),
            ("canny_fill",     lambda: self._canny_mask(blurred, h, w)),
            ("color_mask",     lambda: self._color_mask(image)),
            ("whole_image",    lambda: self._whole_image_mask(h, w)),  # last resort
        ]

        for name, fn in strategies:
            try:
                m = fn()
                if m is None or np.count_nonzero(m) == 0:
                    continue
                items = self._extract_items(image, m)
                if items:
                    print(f"Detection: found {len(items)} item(s) via [{name}]")
                    return items
            except Exception as e:
                print(f"Strategy [{name}] failed: {e}")
                continue

        print("Detection: no items found in any strategy — using full-frame fallback.")
        # Absolute last resort: treat the entire image as one onion
        # (handles close-up shots where the onion fills the whole frame)
        full_mask = self._whole_image_mask(h, w)
        cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            area = float(h * w) * 0.80   # approximate onion area
            radius = min(h, w) / 2.0 * 0.90
            item_m = full_mask.copy()
            return [{
                "item_id":      1,
                "pixel_area":   area,
                "max_diameter": float(radius * 2.0),
                "bbox":         (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90)),
                "contour":      cnt,
                "item_mask":    item_m
            }]
        return []

    # ─── segmentation helpers ─────────────────────────────────────────────────

    def _otsu_mask(self, blurred: np.ndarray, gray: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border_mean = float(np.mean(np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])))
        if border_mean > 150:          # light background → invert
            otsu = cv2.bitwise_not(otsu)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=3)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k, iterations=1)
        return otsu

    def _adaptive_mask(self, blurred: np.ndarray) -> np.ndarray:
        adapt = cv2.adaptiveThreshold(blurred, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 4)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k, iterations=3)
        return adapt

    def _canny_mask(self, blurred: np.ndarray, h: int, w: int) -> np.ndarray:
        edges = cv2.Canny(blurred, 10, 50)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edges = cv2.dilate(edges, k, iterations=4)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((h, w), dtype=np.uint8)
        for cnt in cnts:
            if cv2.contourArea(cnt) > (h * w * 0.003):
                cv2.drawContours(filled, [cnt], -1, 255, -1)
        return filled

    def _color_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m = get_onion_color_mask(hsv)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=4)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
        return m

    def _whole_image_mask(self, h: int, w: int) -> np.ndarray:
        """Last resort: treat entire image center region as potential onion area."""
        m = np.zeros((h, w), dtype=np.uint8)
        pad_h, pad_w = int(h * 0.05), int(w * 0.05)
        m[pad_h:h - pad_h, pad_w:w - pad_w] = 255
        return m

    # ─── item extraction from mask ────────────────────────────────────────────

    def _extract_items(self, image: np.ndarray,
                       mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        From a binary mask, find round blobs and return them as onion items.
        Shape-first: validates roundness only, no color gating.
        """
        img_h, img_w = image.shape[:2]

        # Distance transform to separate touching onions
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        # Very low threshold so we don't lose small onions
        _, fg = cv2.threshold(dist, 0.10 * dist.max(), 255, 0)
        fg = np.uint8(fg)

        # Fill and close gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        items = []
        item_id = 1
        img_area = float(img_h * img_w)

        for cnt in cnts:
            area = float(cv2.contourArea(cnt))

            # Skip tiny noise (< 0.3% of image) and full-frame blobs (> 98%)
            if area < img_area * 0.003:
                continue
            if self._touches_border(cnt, image.shape):
                continue
            if not self._is_round_enough(cnt):
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            # Use bbox_ prefix to avoid shadowing img_h, img_w
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)

            # Build per-item mask at FULL image dimensions (not bbox dimensions)
            # BUG FIX: previously used (h,w) from boundingRect — wrong size,
            # caused cv2.bitwise_and shape mismatch with the full-image mask.
            item_m = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(item_m, [cnt], -1, 255, -1)
            k_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            item_m = cv2.dilate(item_m, k_restore, iterations=2)
            item_m = cv2.bitwise_and(item_m, mask)
            if np.count_nonzero(item_m) < 100:
                # Fallback: plain filled contour without AND with mask
                item_m = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(item_m, [cnt], -1, 255, -1)

            # Scale up to compensate for distance transform erosion
            items.append({
                "item_id":      item_id,
                "pixel_area":   area * 1.6,
                "max_diameter": float(radius * 2.0) * 1.2,
                "bbox":         (int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)),
                "contour":      cnt,
                "item_mask":    item_m
            })
            item_id += 1

        return items
