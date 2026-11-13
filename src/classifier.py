import numpy as np
from typing import Dict, Any
from src.ai_model import OnionQualityAIModel


class OnionAGMARKClassifier:
    """
    Simplified 2-grade classifier: Grade A (good) or Reject (bad/diseased).
    - Grade A: healthy onion, no significant mold/defects
    - Reject:  mold detected, heavy defects, or sprouting
    """

    def __init__(self, pixels_per_mm: float = 2.5):
        self.pixels_per_mm = max(pixels_per_mm, 0.3)
        self.ai_model = OnionQualityAIModel()

    def classify_onion(
        self,
        pixel_area: float,
        max_diameter_px: float,
        defect_analysis: Dict[str, Any],
        today_mandi_modal_price: float = 40.0
    ) -> Dict[str, Any]:

        diameter_mm = float(max_diameter_px / self.pixels_per_mm)
        area_mm2    = float(pixel_area / (self.pixels_per_mm ** 2))
        area_cm2    = float(area_mm2 / 100.0)

        rot_pct    = float(defect_analysis.get("rot_percentage",    0.0))
        sprout_pct = float(defect_analysis.get("sprout_percentage", 0.0))
        color_score= float(defect_analysis.get("color_score",       0.85))
        defect_pct = float(defect_analysis.get("defect_percentage", 0.0))
        primary_def= defect_analysis.get("primary_defect", "Sound Produce (No Defect)")

        # ── Reject condition ─────────────────────────────────────────────────
        # Any visible mold, heavy sprouting, or high defect score → Reject
        is_reject = (
            rot_pct    >= 8.0  or
            sprout_pct >= 15.0 or
            defect_pct >= 20.0
        )

        ai_grade, ai_conf = self.ai_model.predict_quality_grade(
            diam_mm=diameter_mm, area_cm2=area_cm2,
            rot_pct=rot_pct, sprout_pct=sprout_pct,
            color_score=color_score, defect_pct=defect_pct
        )

        if is_reject or ai_grade == "Reject":
            grade        = "Reject"
            grade_desc   = "Unfit for retail sale — Diseased / Mold / Defect detected"
            quality_stat = f"Rejected (Rot:{rot_pct:.1f}% Sprout:{sprout_pct:.1f}% Defect:{defect_pct:.1f}%)"
            mandi_price  = 0.0
            confidence   = round(ai_conf if is_reject else 0.99, 2)
        elif diameter_mm >= 55.0 or ai_grade == "Grade A":
            grade        = "Grade A"
            grade_desc   = "AGMARK Grade A (Large ≥55mm, High Quality Export Grade)"
            quality_stat = f"Premium Quality (AI Confidence: {int(ai_conf*100)}%)"
            mandi_price  = round(today_mandi_modal_price * 1.0, 1)
            confidence   = round(ai_conf, 2)
        elif diameter_mm >= 40.0 or ai_grade == "Grade B":
            grade        = "Grade B"
            grade_desc   = "AGMARK Grade B (Medium 40–55mm, Standard Commercial Grade)"
            quality_stat = f"Standard Domestic Quality (AI Confidence: {int(ai_conf*100)}%)"
            mandi_price  = round(today_mandi_modal_price * 0.65, 1)
            confidence   = round(ai_conf, 2)
        else:
            grade        = "Small Grade"
            grade_desc   = "AGMARK Small Grade (30–40mm, Sambar / Pickle Size)"
            quality_stat = f"Small Commercial Grade (AI Confidence: {int(ai_conf*100)}%)"
            mandi_price  = round(today_mandi_modal_price * 0.45, 1)
            confidence   = round(ai_conf, 2)

        return {
            "grade":          grade,
            "quality_status": quality_stat,
            "agmark_standard": grade_desc,
            "ai_confidence":  round(confidence if not is_reject else 0.99, 2),
            "metrics": {
                "diameter_mm":     round(diameter_mm, 1),
                "area_cm2":        round(area_cm2, 2),
                "pixel_area":      round(pixel_area, 0),
                "max_diameter_px": round(max_diameter_px, 1)
            },
            "defects": {
                "color_score":       round(color_score, 2),
                "rot_percentage":    round(rot_pct, 2),
                "sprout_percentage": round(sprout_pct, 2),
                "defect_percentage": round(defect_pct, 2),
                "primary_defect":    primary_def
            },
            "market_estimate": {
                "estimated_price_inr_per_kg": mandi_price,
                "currency": "INR (₹)"
            }
        }
