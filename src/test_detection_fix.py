"""Quick test to verify the detection size bug fix and mold threshold calibration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import cv2

passed = 0
failed = 0


def check(cond, msg_pass, msg_fail):
    global passed, failed
    if cond:
        print(f"  PASS: {msg_pass}")
        passed += 1
    else:
        print(f"  FAIL: {msg_fail}")
        failed += 1


# ── Test 1: _extract_items no longer crashes with size mismatch ────────────
print("\n[1] detection._extract_items size-mismatch bug fix")
from src.detection import OnionDetectorStub
det = OnionDetectorStub()

img = np.ones((400, 600, 3), dtype=np.uint8) * 200
cv2.circle(img, (300, 200), 120, (60, 30, 180), -1)
mask = np.zeros((400, 600), dtype=np.uint8)
cv2.circle(mask, (300, 200), 120, 255, -1)

try:
    items = det._extract_items(img, mask)
    check(True, f"_extract_items returned {len(items)} item(s) without crash", "")
    if items:
        m = items[0]["item_mask"]
        check(m.shape == (400, 600),
              f"item_mask shape correct: {m.shape}",
              f"item_mask shape WRONG: {m.shape} (expected (400, 600))")
    else:
        print("  WARN: 0 items found — may be circularity filter too strict")
except Exception as e:
    check(False, "", f"_extract_items crashed: {e}")


# ── Test 2: healthy brown onion → LOW rot (mold threshold calibration) ─────
print("\n[2] defect_analysis — healthy brown/tan onion skin should NOT be flagged as mold")
from src.defect_analysis import analyze_onion_defects

def solid_bgr_image(h_val, s_val, v_val, size=200):
    """Create solid-color BGR image from HSV values."""
    hsv_px = np.array([[[h_val, s_val, v_val]]], dtype=np.uint8)
    bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr_px
    return img

mask2 = np.ones((200, 200), dtype=np.uint8) * 255

# Healthy red-brown: H=10, S=130, V=140
r = analyze_onion_defects(solid_bgr_image(10, 130, 140), mask2)
check(r["rot_percentage"] < 5.0,
      f"Red-brown healthy: rot={r['rot_percentage']:.1f}% (OK < 5%)",
      f"Red-brown healthy FALSELY flagged: rot={r['rot_percentage']:.1f}%")

# Healthy yellow onion: H=22, S=180, V=200
r = analyze_onion_defects(solid_bgr_image(22, 180, 200), mask2)
check(r["rot_percentage"] < 5.0,
      f"Yellow healthy: rot={r['rot_percentage']:.1f}% (OK < 5%)",
      f"Yellow healthy FALSELY flagged: rot={r['rot_percentage']:.1f}%")

# Healthy dark brown dry skin: H=12, S=70, V=90
r = analyze_onion_defects(solid_bgr_image(12, 70, 90), mask2)
check(r["rot_percentage"] < 15.0,
      f"Dark brown dry skin: rot={r['rot_percentage']:.1f}% (OK < 15%)",
      f"Dark brown dry skin FALSELY flagged: rot={r['rot_percentage']:.1f}%")

# White onion: H=0, S=10, V=240
r = analyze_onion_defects(solid_bgr_image(0, 10, 240), mask2)
check(r["rot_percentage"] < 5.0,
      f"White onion skin: rot={r['rot_percentage']:.1f}% (OK < 5%)",
      f"White onion FALSELY flagged: rot={r['rot_percentage']:.1f}%")


# ── Test 3: gray Aspergillus mold → HIGH rot ──────────────────────────────
print("\n[3] defect_analysis — gray/black mold MUST be detected")

# Gray mold: H=10, S=30, V=70
r = analyze_onion_defects(solid_bgr_image(10, 30, 70), mask2)
check(r["rot_percentage"] > 20.0,
      f"Gray mold (V=70, S=30): rot={r['rot_percentage']:.1f}% (OK > 20%)",
      f"Gray mold NOT detected: rot={r['rot_percentage']:.1f}%")

# Very dark mold: H=0, S=10, V=35
r = analyze_onion_defects(solid_bgr_image(0, 10, 35), mask2)
check(r["rot_percentage"] > 30.0,
      f"Black mold (V=35): rot={r['rot_percentage']:.1f}% (OK > 30%)",
      f"Black mold NOT detected: rot={r['rot_percentage']:.1f}%")


# ── Test 4: full pipeline — healthy vs diseased → correct grade ────────────
print("\n[4] End-to-end: healthy onion → Grade A/B, diseased → Reject")
from src.classifier import OnionAGMARKClassifier
clf = OnionAGMARKClassifier(pixels_per_mm=2.5)

# Healthy yellow onion image
healthy_img = solid_bgr_image(22, 180, 200, size=300)
healthy_mask = np.ones((300, 300), dtype=np.uint8) * 255
defects_h = analyze_onion_defects(healthy_img, healthy_mask)
# pixel_area ~70686 at radius 150, max_diam_px ~375 (150mm)
result_h = clf.classify_onion(70686, 375, defects_h, today_mandi_modal_price=40.0)
check(result_h["grade"] != "Reject",
      f"Healthy yellow onion → {result_h['grade']} (not Reject ✓)",
      f"Healthy yellow onion → {result_h['grade']} (WRONG — should not be Reject!)")

# Diseased gray mold onion
mold_img = solid_bgr_image(10, 30, 70, size=300)
defects_d = analyze_onion_defects(mold_img, healthy_mask)
result_d = clf.classify_onion(70686, 375, defects_d, today_mandi_modal_price=40.0)
check(result_d["grade"] == "Reject",
      f"Gray mold onion → {result_d['grade']} (Reject ✓)",
      f"Gray mold onion → {result_d['grade']} (WRONG — should be Reject!)")


print(f"\n{'='*50}")
print(f"Results: {passed} PASSED, {failed} FAILED")
if failed == 0:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗ — review output above")
