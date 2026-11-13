"""
AGMARK Onion Grading System — Database & Model Health Check
Run: py src/db_check.py
"""
import os, csv, joblib
from collections import Counter
import numpy as np

BASE = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE, "active_learning_dataset.csv")
MODEL_PATH = os.path.join(BASE, "onion_quality_model.joblib")

LABELS = {0: "Grade A", 1: "Grade B", 2: "Small Grade", 3: "Reject"}
SEP = "=" * 62

def check_csv():
    print(f"\n{'─'*62}")
    print("  📂  ACTIVE LEARNING DATASET  (active_learning_dataset.csv)")
    print(f"{'─'*62}")

    if not os.path.exists(CSV_PATH):
        print("  ✗  FILE NOT FOUND — no samples logged yet.")
        return [], []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows   = list(reader)

    print(f"  Header : {header}")
    print(f"  Total rows (samples): {len(rows)}")

    malformed = [i+2 for i, r in enumerate(rows) if len(r) != 9]
    if malformed:
        print(f"  ✗  Malformed rows at line(s): {malformed}")
    else:
        print(f"  ✓  All rows have correct 9 columns")

    # Type validation
    bad_vals = []
    for i, row in enumerate(rows):
        try:
            [float(v) for v in row[:8]]
            int(row[8])
        except Exception as e:
            bad_vals.append((i+2, str(e)))
    if bad_vals:
        print(f"  ✗  Rows with invalid numeric values:")
        for line, err in bad_vals[:5]:
            print(f"       Line {line}: {err}")
    else:
        print(f"  ✓  All values are valid numerics")

    # Label distribution
    labels = []
    for row in rows:
        try: labels.append(int(row[8]))
        except: pass

    dist = Counter(labels)
    print(f"\n  Label Distribution:")
    all_ok = True
    for lbl_idx, lbl_name in LABELS.items():
        count = dist.get(lbl_idx, 0)
        status = "✓" if count > 0 else "⚠ MISSING"
        print(f"    [{lbl_idx}] {lbl_name:12s} → {count:4d} samples  {status}")
        if count == 0:
            all_ok = False

    if not all_ok:
        print("\n  ⚠  WARNING: Some grade classes have ZERO samples in the")
        print("     active learning dataset. This causes class imbalance.")
        print("     The model retrain may underperform on missing classes.")
        print("     FIX: Process onion images of all 4 grade types to build")
        print("     balanced training data.")

    # Duplicate check
    raw = [tuple(r) for r in rows]
    dupes = len(raw) - len(set(raw))
    if dupes > 0:
        print(f"\n  ⚠  {dupes} duplicate rows found in dataset.")
    else:
        print(f"\n  ✓  No duplicate rows")

    return rows, labels


def check_model():
    print(f"\n{'─'*62}")
    print("  🤖  AI MODEL FILE  (onion_quality_model.joblib)")
    print(f"{'─'*62}")

    if not os.path.exists(MODEL_PATH):
        print("  ✗  MODEL FILE NOT FOUND — will be created on first run.")
        return

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"  File size: {size_kb:.1f} KB")

    try:
        saved = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"  ✗  CORRUPT MODEL FILE: {e}")
        print("  FIX: Delete onion_quality_model.joblib — it will retrain on next startup.")
        return

    # Keys check
    required_keys = {"model", "scaler", "user_samples_count"}
    missing_keys = required_keys - set(saved.keys())
    if missing_keys:
        print(f"  ✗  Missing keys in model file: {missing_keys}")
    else:
        print(f"  ✓  Model file has all required keys")

    model  = saved.get("model")
    scaler = saved.get("scaler")
    uc     = saved.get("user_samples_count", 0)

    print(f"  User samples logged: {uc}")
    print(f"  n_estimators:        {getattr(model, 'n_estimators', '?')}")
    print(f"  Classes in model:    {getattr(model, 'classes_', '?')}")

    # Predict sanity check
    try:
        X_test = np.array([[65.0, 33.0, 0.5, 0.0, 0.9, 1.5, 0.85, 1.0],  # Grade A
                           [45.0, 16.0, 3.0, 1.0, 0.7, 5.0, 0.75, 1.1],  # Grade B
                           [35.0,  9.0, 2.0, 1.0, 0.6, 4.0, 0.70, 1.0],  # Small Grade
                           [25.0,  5.0,15.0,10.0, 0.2,40.0, 0.40, 0.8]]) # Reject
        X_sc = scaler.transform(X_test)
        preds = model.predict(X_sc)
        class_names = ["Grade A", "Grade B", "Small Grade", "Reject"]
        print(f"\n  Sanity Prediction Test:")
        expected = [0, 1, 2, 3]
        all_pass = True
        for i, (pred, exp) in enumerate(zip(preds, expected)):
            ok = pred == exp
            if not ok: all_pass = False
            status = "✓" if ok else f"✗ GOT {class_names[int(pred)]}"
            print(f"    Test {i+1} (expect {class_names[exp]:12s}): {status}")
        if all_pass:
            print(f"\n  ✓  All sanity tests passed — model is healthy")
        else:
            print(f"\n  ⚠  Some sanity tests failed — model may need retraining")
            print(f"     FIX: Delete onion_quality_model.joblib to force retrain")
    except Exception as e:
        print(f"  ✗  Prediction test failed: {e}")


def check_sih_badge():
    """Check if the 'SIH Winning Feature' badge text in HTML is correct."""
    print(f"\n{'─'*62}")
    print("  🏆  SIH BADGE TEXT CHECK  (templates/index.html)")
    print(f"{'─'*62}")
    html_path = os.path.join(BASE, "templates", "index.html")
    if not os.path.exists(html_path):
        print("  ✗  index.html not found")
        return
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    checks = [
        ("SIH WINNING FEATURE", "Hero badge text"),
        ("AGMARK AI", "Brand name in navbar"),
        ("priceChart", "Chart.js canvas ID"),
        ("grade_image", "API endpoint reference"),
        ("market_trends", "Market API reference"),
    ]
    all_ok = True
    for text, desc in checks:
        found = text in content
        if not found: all_ok = False
        print(f"  {'✓' if found else '✗'} {desc:30s} → '{text}'")
    if all_ok:
        print(f"\n  ✓  HTML template looks correct")
    else:
        print(f"\n  ✗  Some expected strings missing from HTML template")


def print_summary(rows, labels):
    print(f"\n{SEP}")
    print("  📊  SUMMARY REPORT")
    print(SEP)
    total = len(rows)
    dist  = Counter(labels)
    grade_a     = dist.get(0, 0)
    grade_b     = dist.get(1, 0)
    small       = dist.get(2, 0)
    reject      = dist.get(3, 0)
    issues = []
    if total == 0:
        issues.append("No samples logged yet — run an image assessment first")
    if small == 0:
        issues.append("Label 2 (Small Grade) has 0 samples — class missing from dataset")
    if grade_b < 5:
        issues.append(f"Label 1 (Grade B) only has {grade_b} samples — too few for reliable retraining")
    if total > 0 and max(dist.values()) / total > 0.70:
        dominant = LABELS[max(dist, key=dist.get)]
        issues.append(f"Dataset is heavily skewed towards '{dominant}' — consider more diverse images")

    if issues:
        print(f"\n  ⚠  {len(issues)} issue(s) found:\n")
        for i, iss in enumerate(issues, 1):
            print(f"    {i}. {iss}")
        print(f"\n  Recommended fixes:")
        print(f"    • Upload onion images of ALL 4 grade types (A, B, Small, Reject)")
        print(f"    • Each retraining cycle triggers every 5 new samples")
        print(f"    • Delete active_learning_dataset.csv to reset user samples")
        print(f"    • Delete onion_quality_model.joblib to force full retrain")
    else:
        print(f"\n  ✓  No critical issues found. Database and model are healthy.")

    print(f"\n  Dataset: {total} samples | A:{grade_a} B:{grade_b} Small:{small} Reject:{reject}")
    print(SEP + "\n")


if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  AGMARK AI — Database & Model Health Check")
    print(SEP)
    rows, labels = check_csv()
    check_model()
    check_sih_badge()
    print_summary(rows, labels)
