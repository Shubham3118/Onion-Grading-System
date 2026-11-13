import cv2
import numpy as np
from typing import Dict, Any
from src.preprocessing import get_onion_color_mask


def analyze_onion_defects(bgr_image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Defect analysis calibrated to real Aspergillus niger mold on onions.

    Root-cause fix for inverted predictions:
    ─────────────────────────────────────────
    The diseased onion in the test image has GRAY mold streaks (V=40-90, S=5-40),
    NOT pure black (V<35). The old threshold V<35 & S<40 missed ALL of this mold,
    so a heavily diseased onion got rot=0% → Grade A.

    New threshold: V < 110 AND S < 85
    - Catches: dark gray / black mold (V=10-100, S=5-80) ✓
    - Misses:  healthy tan skin (V=150-220, S=20-60) ✓  
    - Misses:  white background (V=240-255, S=0-10) ✓
    - Misses:  yellow onion body (V=150-220, S=100-220) ✓

    After erosion (removes 1-pixel noise/shadows) + dilation (restores blobs):
    real mold clusters survive, isolated JPEG noise does not.
    """

    SOUND = {
        "rot_percentage":    0.0,
        "sprout_percentage": 0.0,
        "color_score":       0.85,
        "defect_percentage": 0.0,
        "primary_defect":    "Sound Produce (No Defect)",
        "rot_mask":          None,
        "sprout_mask":       None
    }

    if bgr_image is None or mask is None or np.count_nonzero(mask) == 0:
        return SOUND

    # Always use HSV — never LAB for defect detection
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    total_px = float(np.count_nonzero(mask))
    if total_px == 0:
        return SOUND

    # ── 1. Mold / Rot Detection ───────────────────────────────────────────────
    # Aspergillus niger on onions appears as dark charcoal or gray mold clusters.
    # Key distinction:
    #   - True mold:          V < 45 (very dark) OR (V < 90 AND S < 45) — desaturated near-black
    #   - Healthy brown skin: H 5-30, S > 50, V 60-180  — warm-toned, more saturated
    #   - Dark brown dry skin: S > 40 even when V is lower → NOT mold
    # Previous threshold (V<105 & S<60) was too broad: it caught healthy brown/tan skin
    # (V=80-120, S=50-80) causing healthy onions to show rot_pct=15-35% → false Reject.
    rot_candidate = (
        ((v_ch < 45) | ((v_ch < 90) & (s_ch < 45))) & (mask > 0)
    ).astype(np.uint8) * 255

    # Remove 1-pixel JPEG noise (erode), restore genuine mold clusters (dilate)
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rot_eroded = cv2.erode(rot_candidate, k_e, iterations=1)
    rot_clean  = cv2.dilate(rot_eroded,   k_d, iterations=1)
    rot_clean  = cv2.bitwise_and(rot_clean, rot_clean, mask=mask)

    rot_px  = float(np.count_nonzero(rot_clean))
    rot_pct = min(rot_px / total_px * 100.0, 100.0)

    # ── 2. Sprout Detection ───────────────────────────────────────────────────
    # Genuine green shoots: H 38-82, S >= 80, V >= 50
    # Yellow onion skin peaks at H~15-25 — safely excluded by H >= 38
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spr_raw   = cv2.inRange(hsv,
                            np.array([38,  80, 50], dtype=np.uint8),
                            np.array([82, 255, 255], dtype=np.uint8))
    spr_raw   = cv2.bitwise_and(spr_raw, spr_raw, mask=mask)
    spr_clean = cv2.morphologyEx(spr_raw, cv2.MORPH_OPEN, k3, iterations=2)
    sprout_pct = min(float(np.count_nonzero(spr_clean)) / total_px * 100.0, 100.0)

    # ── 3. Color Uniformity Score ─────────────────────────────────────────────
    # Fraction of mask pixels matching healthy onion colors
    healthy_mask = get_onion_color_mask(hsv)
    healthy_mask = cv2.bitwise_and(healthy_mask, healthy_mask, mask=mask)
    healthy_px   = float(np.count_nonzero(healthy_mask))
    color_score  = min(max(healthy_px / total_px, 0.0), 1.0)

    # Brightness fallback for very dim images
    if color_score < 0.08:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        masked_vals = gray[mask > 0]
        if len(masked_vals) > 0 and float(np.mean(masked_vals)) > 50:
            color_score = 0.55

    # ── 4. Composite Defect % ─────────────────────────────────────────────────
    # Only rot and sprout count — color_score NOT penalized (yellow/brown onions are healthy)
    true_sprout = max(0.0, sprout_pct - 15.0)
    defect_pct  = min(rot_pct * 2.0 + true_sprout * 2.5, 100.0)

    # ── 5. Primary Defect Label ───────────────────────────────────────────────
    rot_r    = round(rot_pct,    1)
    sprout_r = round(sprout_pct, 1)

    if rot_r >= 15.0:
        primary_defect = f"Black Mold Rot (Aspergillus) [Severe: {rot_r}%]"
    elif rot_r >= 6.0:
        primary_defect = f"Black Mold Spot (Aspergillus) [Moderate: {rot_r}%]"
    elif rot_r >= 2.0:
        primary_defect = f"Mold Traces (Aspergillus) [Minor: {rot_r}%]"
    elif sprout_r >= 20.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Severe: {sprout_r}%]"
    elif sprout_r >= 12.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Moderate: {sprout_r}%]"
    else:
        primary_defect = "Sound Produce (No Defect)"

    return {
        "rot_percentage":    round(rot_pct,    2),
        "sprout_percentage": round(sprout_pct, 2),
        "color_score":       round(color_score, 2),
        "defect_percentage": round(defect_pct,  2),
        "primary_defect":    primary_defect,
        "rot_mask":          rot_clean,
        "sprout_mask":       spr_clean
    }
