"""
Pixel-level analysis of the diseased onion image to find correct HSV thresholds.
Run: py src/analyze_image.py <path_to_image>
"""
import sys, cv2, numpy as np

if len(sys.argv) < 2:
    print("Usage: py src/analyze_image.py <image_path>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print("Could not load image:", sys.argv[1])
    sys.exit(1)

h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image size: {w}x{h}")

# Remove white background (border mean > 220 → white bg)
border = np.concatenate([gray[0,:],gray[-1,:],gray[:,0],gray[:,-1]])
print(f"Border mean brightness: {np.mean(border):.1f}")

# Simple threshold to get onion mask
_, fg_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=3)

onion_pixels = hsv[fg_mask > 0]
print(f"\nOnion region pixels: {len(onion_pixels)}")
print(f"Onion H range: {onion_pixels[:,0].min()}-{onion_pixels[:,0].max()} mean={onion_pixels[:,0].mean():.1f}")
print(f"Onion S range: {onion_pixels[:,1].min()}-{onion_pixels[:,1].max()} mean={onion_pixels[:,1].mean():.1f}")
print(f"Onion V range: {onion_pixels[:,2].min()}-{onion_pixels[:,2].max()} mean={onion_pixels[:,2].mean():.1f}")

# Find dark gray / mold pixels (V < 100 on the onion body)
mold_cand = ((hsv[:,:,2] < 100) & (hsv[:,:,1] < 60) & (fg_mask > 0))
mold_px = hsv[mold_cand]
print(f"\nMold candidate pixels (V<100, S<60, on onion): {len(mold_px)}")
if len(mold_px) > 0:
    print(f"  H: {mold_px[:,0].min()}-{mold_px[:,0].max()} mean={mold_px[:,0].mean():.1f}")
    print(f"  S: {mold_px[:,1].min()}-{mold_px[:,1].max()} mean={mold_px[:,1].mean():.1f}")
    print(f"  V: {mold_px[:,2].min()}-{mold_px[:,2].max()} mean={mold_px[:,2].mean():.1f}")
    pct = len(mold_px) / len(onion_pixels) * 100
    print(f"  = {pct:.1f}% of onion area")

# Wider mold search: V < 120 AND S < 80
mold2 = ((hsv[:,:,2] < 120) & (hsv[:,:,1] < 80) & (fg_mask > 0))
mold2_px = hsv[mold2]
print(f"\nWider mold (V<120, S<80): {len(mold2_px)} = {len(mold2_px)/max(len(onion_pixels),1)*100:.1f}%")

# Check what the black mold streaks look like vs healthy tan skin
# Sample a small center patch of onion (expected healthy tan)
cy, cx = h//2, w//2
patch = hsv[cy-30:cy+30, cx-30:cx+30][fg_mask[cy-30:cy+30, cx-30:cx+30] > 0]
if len(patch) > 0:
    print(f"\nCenter patch (should be tan skin):")
    print(f"  H={patch[:,0].mean():.1f} S={patch[:,1].mean():.1f} V={patch[:,2].mean():.1f}")

# Top-left dark region (expected mold)
tl_hsv = hsv[h//4:h//2, w//8:w//3]
tl_mask = fg_mask[h//4:h//2, w//8:w//3]
tl_px = tl_hsv[tl_mask > 0]
if len(tl_px) > 0:
    print(f"\nTop-left dark region (should be mold):")
    print(f"  H={tl_px[:,0].mean():.1f} S={tl_px[:,1].mean():.1f} V={tl_px[:,2].mean():.1f}")
    dark_in_tl = tl_px[tl_px[:,2] < 100]
    print(f"  Dark pixels (V<100): {len(dark_in_tl)} / {len(tl_px)} = {len(dark_in_tl)/max(len(tl_px),1)*100:.1f}%")
