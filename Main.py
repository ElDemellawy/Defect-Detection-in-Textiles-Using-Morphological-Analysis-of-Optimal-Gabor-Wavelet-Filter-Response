"""
============================================================
Reimplementation of:
  H. Alimohamadi, A. Ahmadyfard, E. Shojaee,
  "Defect Detection in Textiles Using Morphological Analysis
   of Optimal Gabor Wavelet Filter Response,"
  ICCAE 2009, pp. 26-30.  DOI: 10.1109/ICCAE.2009.43
 
Author : Mohamed Eldemellawy
Course : Computer Vision – Assignment 3
         Prof. Marcus Barkowsky, THD – Summer 2026
============================================================
"""

import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# 0.  PAPER PARAMETERS  (Section III of the paper)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GaborBankParams:
    """
    Exact parameters reported in Section III of the paper:
      M = 4  scales
      N = 6  orientations
      U_l = 0.01  lower centre frequency
      U_h = 0.4   upper centre frequency
      mask_size = 33×33 pixels
    """
    M:         int   = 4      # number of scales
    N:         int   = 6      # number of orientations
    U_l:       float = 0.01   # lower centre frequency  (Eq. 7)
    U_h:       float = 0.4    # upper centre frequency  (Eq. 7)
    mask_size: int   = 33     # filter kernel side length (must be odd)


# ══════════════════════════════════════════════════════════════════════
# 1.  IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════

def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as grayscale float64."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════
# 2.  GABOR WAVELET FILTER BANK  (Eq. 1 – 7)
# ══════════════════════════════════════════════════════════════════════

def build_gabor_bank(params: GaborBankParams) -> dict:
    """Build the Gabor filter bank."""
    M, N       = params.M, params.N
    U_l, U_h   = params.U_l, params.U_h
    size       = params.mask_size
    half       = size // 2

    # α: geometric scale ratio between successive scales  (Eq. 7)
    alpha = (U_h / U_l) ** (1.0 / (M - 1))

    bank = {}
    # pixel coordinate grids (row = y, col = x)
    y_grid, x_grid = np.meshgrid(
        np.arange(-half, half + 1),
        np.arange(-half, half + 1),
        indexing="ij"
    )

    for m in range(M):
        # Centre frequency at this scale  (Eq. 6)
        W = (alpha ** m) * U_l

        # σ_x for this scale  (Eq. 4)
        sigma_x = (
            (alpha + 1) * np.sqrt(2.0 * np.log(2))
        ) / (
            2.0 * np.pi * (alpha ** m) * (alpha - 1) * U_l
        )

        for n in range(N):
            theta = (n * np.pi) / N   # orientation  θ = nπ/N

            # σ_y for this scale/orientation  (Eq. 5)
            inner = (U_h ** 2) / (2.0 * np.log(2)) - (1.0 / (2.0 * np.pi * sigma_x)) ** 2
            inner = max(inner, 1e-9)   # numerical guard
            sigma_y = 1.0 / (
                2.0 * np.pi * np.tan(np.pi / (2.0 * N)) * np.sqrt(inner)
            )

            # Rotate & scale coordinates  (Eq. 3)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_rot = (alpha ** (-m)) * ( x_grid * cos_t + y_grid * sin_t)
            y_rot = (alpha ** (-m)) * (-x_grid * sin_t + y_grid * cos_t)

            # Gaussian envelope
            gaussian = np.exp(
                -0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2)
            )
            # Complex sinusoidal carrier
            carrier = np.exp(1j * 2.0 * np.pi * W * x_rot)

            # Assemble kernel and scale by α^{-m} / normalisation (Eq. 2)
            kernel = (alpha ** (-m)) * gaussian * carrier
            kernel /= (2.0 * np.pi * sigma_x * sigma_y)

            bank[(m, n)] = kernel.astype(np.complex64)

    return bank


# ══════════════════════════════════════════════════════════════════════
# 3.  FEATURE EXTRACTION  (Eq. 8)
# ══════════════════════════════════════════════════════════════════════

def gabor_response(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute the Gabor response magnitude."""
    real_part = cv2.filter2D(image, cv2.CV_64F, np.real(kernel))
    imag_part = cv2.filter2D(image, cv2.CV_64F, np.imag(kernel))
    return np.abs(real_part + 1j * imag_part)


# ══════════════════════════════════════════════════════════════════════
# 4.  FISHER CRITERION + OPTIMAL FILTER SELECTION  (Eq. 9 – 12)
# ══════════════════════════════════════════════════════════════════════

def mean_abs_deviation(feat: np.ndarray) -> float:
    """Return the mean absolute deviation."""
    return float(np.mean(np.abs(feat - np.mean(feat))))


def select_optimal_filter(image: np.ndarray, bank: dict) -> tuple:
    """Pick the best filter by Fisher score."""
    best_key, best_F, best_feat = None, -np.inf, None
    all_scores = {}

    M_total = max(k for (k, _) in bank.keys()) + 1

    for (m, n), kernel in bank.items():
        # Skip the finest scale (m = M-1): at the finest scale the Gabor
        # filter locks onto individual thread crossings in the regular weave
        # pattern, producing high F values even on defect-free fabric.
        # The paper's intent is to find the scale that best discriminates
        # defects from background — coarser scales serve this better.
        if m == M_total - 1:
            continue

        feat = gabor_response(image, kernel)
        mu    = float(np.mean(feat))
        sigma = mean_abs_deviation(feat)
        if sigma < 1e-10:
            continue
        F = mu / sigma                    # Eq. (11)
        all_scores[(m, n)] = F
        if F > best_F:
            best_F, best_key, best_feat = F, (m, n), feat

    return best_key, best_F, best_feat, all_scores


# ══════════════════════════════════════════════════════════════════════
# 5.  INPUT IMAGE NORMALISATION  (Eq. 15 – page 3 of the paper)
# ══════════════════════════════════════════════════════════════════════

def normalize_input(image: np.ndarray) -> np.ndarray:
    """Normalize the input image by its mean."""
    M = float(np.mean(image))
    return np.abs(M - image)


# ══════════════════════════════════════════════════════════════════════
# 6-7.  MORPHOLOGICAL GRAYSCALE RECONSTRUCTION  (Eq. 13 – 14)
#       Hand-coded – core contribution, no skimage
# ══════════════════════════════════════════════════════════════════════

def grayscale_reconstruction(marker: np.ndarray, mask: np.ndarray,
                              max_iter: int = 1000) -> np.ndarray:
    """Perform simple grayscale reconstruction by dilation."""
    marker = np.float64(marker.copy())
    mask   = np.float64(mask)

    # 3×3 flat rectangular structuring element
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for _ in range(max_iter):
        dilated    = cv2.dilate(marker.astype(np.float32), se).astype(np.float64)
        new_marker = np.minimum(dilated, mask)
        if np.array_equal(new_marker, marker):
            break
        marker = new_marker

    return marker


# ══════════════════════════════════════════════════════════════════════
# 8.  H-DOME EXTRACTION  (Eq. 16)
# ══════════════════════════════════════════════════════════════════════

def extract_h_domes(feat: np.ndarray, h: float) -> np.ndarray:
    """Extract h-domes from the feature image."""
    I      = np.float64(feat)
    marker = np.clip(I - h, 0.0, None)          # J = I − h, clipped ≥ 0
    rec    = grayscale_reconstruction(marker, I)
    domes  = np.clip(I - rec, 0.0, None)
    return domes


# ══════════════════════════════════════════════════════════════════════
# 9.  FULL PIPELINE FOR ONE IMAGE
# ══════════════════════════════════════════════════════════════════════

def detect_defects(image: np.ndarray,
                   params: Optional[GaborBankParams] = None) -> dict:
    """Run the full defect-detection pipeline on one image."""
    if params is None:
        params = GaborBankParams()

    image = np.float64(image)

    # ── Step 1-4: find optimal filter on the RAW image ──────────────
    bank = build_gabor_bank(params)
    best_key, best_F, _, all_scores = select_optimal_filter(image, bank)
    optimal_kernel = bank[best_key]

    # ── Step 5: normalise the INPUT image ───────────────────────────
    norm_image = normalize_input(image)

    # ── Step 6: re-filter normalised image with optimal kernel ───────
    feat = gabor_response(norm_image, optimal_kernel)

    # ── Step 7-8: adaptive h-dome extraction ────────────────────────
    h     = mean_abs_deviation(feat)        # adaptive h = MAD of feature
    domes = extract_h_domes(feat, h)

    # ── Step 9: binarise dome image → defect mask ────────────────────
    #
    # KEY INSIGHT: on real fabric, regular texture creates many small
    # periodic dome responses everywhere. True defects produce ONE large
    # connected dome region. So the strategy is:
    #   (a) threshold at a high sigma level to get candidate blobs
    #   (b) filter by connected component area — keep only large blobs
    #
    # This matches the paper's intent: morphological h-domes naturally
    # suppress small texture peaks when h is large enough, and what
    # remains should be spatially coherent defect regions.

    binary_mask = np.zeros(domes.shape, dtype=np.uint8)

    if domes.max() > 1e-9:
        # (a) Threshold: keep pixels significantly above the dome mean
        dome_mean = float(np.mean(domes))
        dome_std  = float(np.std(domes))
        thresh_val = max(dome_mean + 3.5 * dome_std, np.percentile(domes, 99.0))
        raw_mask = (domes > thresh_val).astype(np.uint8) * 255

        # (b) Morphological clean-up first (remove single-pixel noise)
        se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  se3)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, se3)

        # (c) Connected component analysis — filter out small blobs
        # Real defects cover at least MIN_DEFECT_AREA pixels.
        # Texture noise blobs are tiny (< 100 pixels typically).
        # We set min area relative to image size for scale invariance.
        image_area    = domes.shape[0] * domes.shape[1]
        min_blob_area = max(80, int(image_area * 0.0003))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )
        for label_idx in range(1, num_labels):   # skip background (0)
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_blob_area:
                binary_mask[labels == label_idx] = 255

    return {
        "optimal_filter": best_key,
        "F_value":        best_F,
        "all_scores":     all_scores,
        "norm_image":     norm_image,
        "feature_image":  feat,
        "h_value":        h,
        "dome_image":     domes,
        "binary_mask":    binary_mask,
    }


# ══════════════════════════════════════════════════════════════════════
# 10.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def visualize_result(result: dict, image: np.ndarray,
                     title: str = "", save_path: Optional[Path] = None):
    """Save a simple visualization of the pipeline output."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=11)

    panels = [
        (image,                    "1. Original image"),
        (result["norm_image"],     "2. Normalised N(x,y)=|M-T|"),
        (result["feature_image"],  f"3. Optimal Gabor response\n"
                                   f"   filter (m={result['optimal_filter'][0]}, "
                                   f"n={result['optimal_filter'][1]}), "
                                   f"F={result['F_value']:.2f}"),
        (result["dome_image"],     f"4. h-Domes  (h={result['h_value']:.4f})"),
        (result["binary_mask"],    "5. Binary defect mask"),
    ]

    for ax, (img, lbl) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 11.  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════

def is_defective_image(binary_mask: np.ndarray,
                        min_defect_pixels: int = 150) -> bool:
    """Decide whether an image is defective from the mask size."""
    return int(np.sum(binary_mask > 0)) >= min_defect_pixels


def evaluate(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Compute basic pixel-level detection metrics."""
    pred = (pred_mask > 0).astype(bool)
    gt   = (gt_mask   > 0).astype(bool)

    TP = int(np.sum( pred &  gt))
    FP = int(np.sum( pred & ~gt))
    TN = int(np.sum(~pred & ~gt))
    FN = int(np.sum(~pred &  gt))

    precision       = TP / (TP + FP + 1e-9)
    recall          = TP / (TP + FN + 1e-9)
    f1              = 2 * precision * recall / (precision + recall + 1e-9)
    false_alarm_rate = FP / (FP + TN + 1e-9)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision":        round(precision,        4),
        "recall":           round(recall,           4),
        "F1":               round(f1,               4),
        "false_alarm_rate": round(false_alarm_rate, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# 12.  DATASET PROCESSING  (adapt paths to your AITEX folder layout)
# ══════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    """Find image files under a folder."""
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def process_dataset(
    defect_dir:    Path,
    defect_free_dir: Path,
    gt_mask_dir:   Optional[Path] = None,
    output_dir:    Optional[Path] = None,
    params:        Optional[GaborBankParams] = None,
    limit:         Optional[int]  = None,
) -> dict:
    """Run the pipeline on a full dataset."""
    if params is None:
        params = GaborBankParams()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    defect_paths      = find_images(defect_dir)[:limit]
    defect_free_paths = find_images(defect_free_dir)[:limit]

    results_defective  = []
    results_defect_free = []
    all_metrics        = []

    print(f"\n{'─'*60}")
    print(f"  Gabor filter bank:  M={params.M} scales × N={params.N} orientations")
    print(f"  Frequencies:        U_l={params.U_l}  U_h={params.U_h}")
    print(f"  Mask size:          {params.mask_size}×{params.mask_size}")
    print(f"  Defective images:   {len(defect_paths)}")
    print(f"  Defect-free images: {len(defect_free_paths)}")
    print(f"{'─'*60}\n")

    # ── Process defective images ──────────────────────────────────────
    for img_path in defect_paths:
        image  = load_gray(img_path)
        result = detect_defects(image, params)

        # Optional ground-truth evaluation
        metrics = None
        if gt_mask_dir:
            gt_path = Path(gt_mask_dir) / (img_path.stem + ".png")
            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                metrics = evaluate(result["binary_mask"], gt_mask)
                all_metrics.append(metrics)

        # Optional visualisation save
        if output_dir:
            vis_path = Path(output_dir) / f"{img_path.stem}_result.png"
            visualize_result(result, image,
                             title=f"{img_path.name}  –  DEFECTIVE",
                             save_path=vis_path)

        detected = is_defective_image(result["binary_mask"])
        row = {"path": img_path, "result": result, "metrics": metrics,
               "label": "defective", "detected": detected}
        results_defective.append(row)
        print(f"[DEFECT]  {img_path.name:<40}  "
              f"filter=({result['optimal_filter'][0]},{result['optimal_filter'][1]})  "
              f"F={result['F_value']:.2f}  "
              f"h={result['h_value']:.4f}  "
              f"detected={'YES' if detected else 'MISSED'}"
              + (f"  pixel_FAR={metrics['false_alarm_rate']:.4f}" if metrics else ""))

    # ── Process defect-free images (should give no detection) ─────────
    false_alarm_count = 0
    for img_path in defect_free_paths:
        image  = load_gray(img_path)
        result = detect_defects(image, params)

        # For defect-free images the binary mask should give no detection.
        # Image-level decision: is_defective_image() matches the paper's
        # per-image reporting (not per-pixel).
        is_false_alarm = is_defective_image(result["binary_mask"])
        if is_false_alarm:
            false_alarm_count += 1

        if output_dir:
            vis_path = Path(output_dir) / f"{img_path.stem}_result.png"
            visualize_result(result, image,
                             title=f"{img_path.name}  –  DEFECT-FREE",
                             save_path=vis_path)

        row = {"path": img_path, "result": result, "metrics": None,
               "label": "defect_free", "false_alarm": is_false_alarm}
        results_defect_free.append(row)
        print(f"[FREE]    {img_path.name:<40}  "
              f"filter=({result['optimal_filter'][0]},{result['optimal_filter'][1]})  "
              f"F={result['F_value']:.2f}  "
              f"false_alarm={'YES' if is_false_alarm else 'no'}")

    # ── Summary ───────────────────────────────────────────────────────
    n_free = len(defect_free_paths)
    far    = false_alarm_count / n_free if n_free else 0.0

    # Image-level detection rate on defective images
    n_defective = len(results_defective)
    n_detected  = sum(1 for r in results_defective if r.get("detected", False))
    detection_rate = n_detected / n_defective if n_defective else 0.0

    print(f"\n{'═'*60}")
    print(f"  IMAGE-LEVEL RESULTS  (matches paper's reporting style)")
    print(f"  Detection rate  (defective):  "
          f"{n_detected}/{n_defective} = {detection_rate*100:.1f}%")
    print(f"  False alarm rate (defect-free): "
          f"{false_alarm_count}/{n_free} = {far*100:.1f}%")
    print(f"  Paper reported: 100% detection, 3.2% false alarm rate")

    if all_metrics:
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall    = np.mean([m["recall"]    for m in all_metrics])
        avg_f1        = np.mean([m["F1"]        for m in all_metrics])
        print(f"  Avg precision : {avg_precision:.4f}")
        print(f"  Avg recall    : {avg_recall:.4f}")
        print(f"  Avg F1        : {avg_f1:.4f}")
    print(f"{'═'*60}\n")

    return {
        "defective":   results_defective,
        "defect_free": results_defect_free,
        "false_alarm_rate": far,
        "metrics":     all_metrics,
    }


# ══════════════════════════════════════════════════════════════════════
# 13.  SMOKE TEST  (run directly: python gabor_defect_detection.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if len(sys.argv) >= 3:
        # Usage: python gabor_defect_detection.py <defect_dir> <defect_free_dir> [gt_dir] [out_dir]
        defect_dir      = Path(sys.argv[1])
        defect_free_dir = Path(sys.argv[2])
        gt_dir          = Path(sys.argv[3]) if len(sys.argv) > 3 else None
        out_dir         = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("output")

        process_dataset(
            defect_dir      = defect_dir,
            defect_free_dir = defect_free_dir,
            gt_mask_dir     = gt_dir,
            output_dir      = out_dir,
            limit           = None,        # set e.g. 10 for quick test
        )

    else:
        # ── Synthetic smoke test (no dataset needed) ─────────────────
        print("No dataset paths provided – running synthetic smoke test.\n"
              "Usage: python gabor_defect_detection.py <defect_dir> <defect_free_dir> "
              "[gt_dir] [out_dir]\n")

        rng  = np.random.default_rng(42)
        size = 256

        # Simulate woven textile: sinusoidal base texture + noise
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        texture = (
            120
            + 25 * np.sin(xx * 2 * np.pi / 16)
            + 20 * np.cos(yy * 2 * np.pi / 12)
            + rng.normal(0, 5, (size, size))
        )

        # Add a dark "hole" defect
        defective = texture.copy()
        defective[110:130, 100:145] -= 55

        # Run pipeline
        params = GaborBankParams()          # paper's exact parameters
        result = detect_defects(defective, params)

        # Save result visualisation
        out = Path("smoke_test_output")
        out.mkdir(exist_ok=True)
        visualize_result(result, defective,
                         title="Synthetic smoke test – dark hole defect",
                         save_path=out / "smoke_test_result.png")

        # Ground-truth mask for the synthetic defect
        gt = np.zeros((size, size), dtype=np.uint8)
        gt[110:130, 100:145] = 255
        metrics = evaluate(result["binary_mask"], gt)

        print(f"Optimal filter : (m={result['optimal_filter'][0]}, "
              f"n={result['optimal_filter'][1]})")
        print(f"Fisher score F : {result['F_value']:.4f}")
        print(f"Adaptive h     : {result['h_value']:.6f}")
        print(f"Metrics        : {metrics}")
        print(f"\nResult image saved to: {out / 'smoke_test_result.png'}")