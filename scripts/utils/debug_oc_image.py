"""
debug_qc_image.py
-----------------
Run QC checks on a single image and report pass/fail for each constraint.

Usage:
    python debug_qc_image.py --img_path abc/xyz.png [--mode semantic|cam_yes|cam_no]

Modes:
    semantic  (default) — red >= SEMANTIC_RED_THRESHOLD, green >= SEMANTIC_GREEN_THRESHOLD
    cam_yes             — red >= CAM_RED_THRESHOLD  (Yes-label cam_pov check)
    cam_no              — red <  CAM_RED_THRESHOLD  (No-label  cam_pov check)
"""

import argparse
import sys
import cv2
import numpy as np

# ── Thresholds (keep in sync with build_vpt1_dataset.py) ─────────────────────
CAM_RED_THRESHOLD        = 500
SEMANTIC_RED_THRESHOLD   = 500
SEMANTIC_GREEN_THRESHOLD = 1000

# ── Pixel counters ────────────────────────────────────────────────────────────

def _red_pixel_count(img_bgr: np.ndarray) -> int:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] >= 0.95) &
        (img_rgb[:, :, 1] <= 0.05) &
        (img_rgb[:, :, 2] <= 0.05)
    )
    return int(mask.sum())


def _green_pixel_count(img_bgr: np.ndarray) -> int:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] <= 0.05) &
        (img_rgb[:, :, 1] >= 0.95) &
        (img_rgb[:, :, 2] <= 0.05)
    )
    return int(mask.sum())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="Path to image file")
    parser.add_argument("--mode", default="semantic",
                        choices=["semantic", "cam_yes", "cam_no"],
                        help="QC mode (default: semantic)")
    args = parser.parse_args()

    img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ FAIL  — could not read image: {args.img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"\nImage : {args.img_path}  ({w}x{h})")
    print(f"Mode  : {args.mode}\n")

    red   = _red_pixel_count(img)
    green = _green_pixel_count(img)

    if args.mode == "semantic":
        checks = [
            ("red   >= SEMANTIC_RED_THRESHOLD",   red,   ">=", SEMANTIC_RED_THRESHOLD),
            ("green >= SEMANTIC_GREEN_THRESHOLD",  green, ">=", SEMANTIC_GREEN_THRESHOLD),
        ]
    elif args.mode == "cam_yes":
        checks = [
            ("red   >= CAM_RED_THRESHOLD (Yes label)", red, ">=", CAM_RED_THRESHOLD),
        ]
    else:  # cam_no
        checks = [
            ("red   <  CAM_RED_THRESHOLD (No label)",  red, "<",  CAM_RED_THRESHOLD),
        ]

    all_passed = True
    for label, value, op, threshold in checks:
        passed = (value >= threshold) if op == ">=" else (value < threshold)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {label}")
        print(f"           value={value}, threshold={threshold}")
        if not passed:
            all_passed = False

    print()
    print("─" * 40)
    print(f"  {'✅ ALL CHECKS PASSED' if all_passed else '❌ ONE OR MORE CHECKS FAILED'}")
    print("─" * 40)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()