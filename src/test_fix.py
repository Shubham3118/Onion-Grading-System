"""
Full verification test for prediction inversion fix.
Tests against pixel values extracted from the actual diseased onion image.
"""
import sys, numpy as np, cv2
sys.path.insert(0, '.')
from src.defect_analysis import analyze_onion_defects
from src.classifier import OnionAGMARKClassifier

clf = OnionAGMARKClassifier(pixels_per_mm=2.5)
PASS = 0; FAIL = 0

def check(name, condition, msg=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ PASS: {name}")
        PASS += 1
    else:
        print(f"  ✗ FAIL: {name} {msg}")
        FAIL += 1

def make_onion(body_bgr, mold_bgr=None, mold_radius=0, size=300):
    img = np.full((size, size, 3), 255, dtype=np.uint8)   # white background
    cv2.circle(img, (size//2, size//2), size//2 - 10, body_bgr, -1)
    if mold_bgr and mold_radius > 0:
        # Large mold patches covering ~40% of onion surface
        cv2.circle(img, (size//2 - 50, size//2 - 40), mold_radius,     mold_bgr, -1)
        cv2.circle(img, (size//2 + 40, size//2 + 20), mold_radius - 10, mold_bgr, -1)
        cv2.circle(img, (size//2 - 20, size//2 + 50), mold_radius - 15, mold_bgr, -1)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size//2, size//2), size//2 - 10, 255, -1)
    return img, mask

print("=" * 60)
print("TEST SUITE — Diseased vs Healthy Onion Classification")
print("=" * 60)

# ── Test 1: Diseased onion with heavy gray mold (matches real image) ─
print("\nTEST 1: Diseased onion — heavy gray Aspergillus mold")
print("  (Pale tan skin BGR~(180,170,150) + dark gray mold BGR~(55,55,50))")
# Tan/beige skin in HSV: H~15, S~25, V~170 → healthy
# Gray mold patches in HSV: H~15, S~15, V~55 → should be caught
img1, mask1 = make_onion(
    body_bgr=(150, 170, 180),    # BGR pale tan skin
    mold_bgr=(50,  55,  55),     # BGR dark gray mold (V~55 in HSV)
    mold_radius=65
)
d1 = analyze_onion_defects(img1, mask1)
print(f"  rot={d1['rot_percentage']:.1f}% sprout={d1['sprout_percentage']:.1f}% defect={d1['defect_percentage']:.1f}%")
print(f"  primary_defect: {d1['primary_defect']}")
r1 = clf.classify_onion(60000, 240, d1, today_mandi_modal_price=40.0)
print(f"  GRADE: {r1['grade']}")
check("Gray mold detected (rot > 5%)", d1['rot_percentage'] > 5.0,
      f"got {d1['rot_percentage']:.1f}%")
check("Diseased onion graded Reject or low", r1['grade'] in ('Reject', 'Small Grade'),
      f"got {r1['grade']}")

# ── Test 2: Very heavy black mold (darkest case) ──────────────────────
print("\nTEST 2: Heavily diseased — near-black mold patches")
img2, mask2 = make_onion(
    body_bgr=(150, 170, 180),
    mold_bgr=(10,  12,  12),     # near-black mold
    mold_radius=70
)
d2 = analyze_onion_defects(img2, mask2)
print(f"  rot={d2['rot_percentage']:.1f}% defect={d2['defect_percentage']:.1f}%")
r2 = clf.classify_onion(60000, 240, d2, today_mandi_modal_price=40.0)
print(f"  GRADE: {r2['grade']}")
check("Black mold detected (rot > 10%)", d2['rot_percentage'] > 10.0,
      f"got {d2['rot_percentage']:.1f}%")
check("Near-black mold → Reject", r2['grade'] == 'Reject', f"got {r2['grade']}")

# ── Test 3: Clean healthy yellow onion ────────────────────────────────
print("\nTEST 3: Healthy yellow onion — no mold")
img3, mask3 = make_onion(body_bgr=(30, 160, 200))  # BGR yellow, no mold
d3 = analyze_onion_defects(img3, mask3)
print(f"  rot={d3['rot_percentage']:.1f}% defect={d3['defect_percentage']:.1f}%")
print(f"  primary_defect: {d3['primary_defect']}")
r3 = clf.classify_onion(60000, 240, d3, today_mandi_modal_price=40.0)
print(f"  GRADE: {r3['grade']}")
check("Healthy onion rot ~0%", d3['rot_percentage'] < 2.0,
      f"got {d3['rot_percentage']:.1f}%")
check("Healthy onion NOT Rejected", r3['grade'] != 'Reject', f"got {r3['grade']}")

# ── Test 4: Clean white/cream onion ───────────────────────────────────
print("\nTEST 4: Healthy white/cream onion — no mold")
img4, mask4 = make_onion(body_bgr=(220, 220, 215))  # BGR cream
d4 = analyze_onion_defects(img4, mask4)
print(f"  rot={d4['rot_percentage']:.1f}% defect={d4['defect_percentage']:.1f}%")
r4 = clf.classify_onion(60000, 240, d4, today_mandi_modal_price=40.0)
print(f"  GRADE: {r4['grade']}")
check("White onion NOT Rejected", r4['grade'] != 'Reject', f"got {r4['grade']}")

# ── Test 5: Shadow on healthy onion should NOT trigger rot ────────────
print("\nTEST 5: Single shadow pixel on healthy onion — should not be rot")
img5, mask5 = make_onion(body_bgr=(30, 160, 200))
cv2.circle(img5, (155, 150), 3, (20, 20, 25), -1)  # tiny 3px shadow
d5 = analyze_onion_defects(img5, mask5)
print(f"  rot={d5['rot_percentage']:.2f}% (should be ~0%)")
check("Shadow noise not counted as rot", d5['rot_percentage'] < 1.5,
      f"got {d5['rot_percentage']:.2f}%")

# ── Test 6: Confirm white background not counted as mold ─────────────
print("\nTEST 6: White background pixels — should not trigger rot")
img6, mask6 = make_onion(body_bgr=(30, 160, 200))
# background is white (255,255,255) in BGR → HSV (0,0,255) → V=255 >> 110
d6 = analyze_onion_defects(img6, mask6)
check("White background not counted as rot", d6['rot_percentage'] < 1.0,
      f"got {d6['rot_percentage']:.2f}%")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED — prediction inversion is fully fixed!")
else:
    print("Some tests failed — check output above")
print("=" * 60)
