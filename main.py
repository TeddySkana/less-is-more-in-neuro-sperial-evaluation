#!/usr/bin/env python3
"""
Spiral tremor analysis pipeline based on the manuscript:
"Less is more: Comparison of automated spiral-drawing metrics with clinician-rated tremor scores after MRgFUS thalamotomy"

What this script does
---------------------
1. Reads a metadata CSV and a folder of spiral drawing images.
2. Preprocesses each spiral image: grayscale, optional crop, resize to 500x500.
3. Extracts an approximate patient trace mask, optionally subtracting an empty template image.
4. Computes three scalar metrics from Spiral A images:
   - Sobel-based orientation irregularity score.
   - Optimal-solution deviation score, if a reference/template is provided.
   - Total drawn line length in mm using skeleton 8-connectivity.
5. Optionally fits an ordinal logistic model with subject-cluster robust SEs:
      CRST ~ sobel + optimal_deviation + line_length + post + treated_hand + post:treated_hand
   This is the manuscript's Python-friendly sensitivity model. A true frequentist ordered-logit mixed
   model with subject random intercept is not available in standard statsmodels; use the optional PyMC
   function below if you need a Bayesian random-intercept version.

Expected metadata columns, by default
-------------------------------------
Required for metric computation:
    image_path             Relative path under --data_dir, or absolute path.

Required for model fitting:
    subject_id             Patient/subject identifier.
    crst_score             Clinician-rated CRST spiral score, integer 0..4.
    hand                   Drawing hand, e.g. right/left.
    treated_hand           Treated hand, e.g. right/left.
    time                   pre/post indicator. Alternatively provide drawing_date and treatment_date.

You can override column names with CLI arguments.

Example
-------
python spiral_tremor_pipeline.py \
  --data_dir /path/to/images \
  --metadata_csv /path/to/metadata.csv \
  --output_dir outputs \
  --empty_template /path/to/empty_spiral_A.png \
  --fit_model

Notes
-----
The manuscript describes the optimal-solution method at a high level. To make this script usable without
private code, this implementation supports two reproducible options:
  A) --ideal_reference: an image containing the ideal midline/reference trace.
  B) --empty_template: an empty template image. The script estimates a midline-like reference from the
     largest skeletonized template component. This is an approximation and should be validated against
     the original reference-generation code if exact replication is required.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class PipelineConfig:
    image_size: int = 500
    dpi: float = 300.0
    threshold_method: str = "otsu"  # otsu, adaptive, or percentile
    dark_percentile: float = 15.0
    min_object_size: int = 25
    template_dilation_px: int = 1
    auto_crop: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h, applied before resize
    sobel_edge_percentile: float = 75.0
    reference_points: int = 600


# -----------------------------
# Image loading and preprocessing
# -----------------------------

def read_grayscale(path: Path) -> np.ndarray:
    """Read an image as uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def crop_to_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


def auto_crop_foreground(img: np.ndarray, pad_fraction: float = 0.08) -> np.ndarray:
    """
    Roughly crop around dark foreground pixels, expand to a square crop.
    This is a practical fallback. For a clinical/research run, prefer a fixed ROI or pre-cropped images.
    """
    # Smooth and threshold dark content.
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only meaningful components and avoid tiny dust marks.
    mask = binary_inv.astype(bool)
    mask = remove_small_objects(mask, min_size=50)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad = int(max(w, h) * pad_fraction)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(w, h) + 2 * pad

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img.shape[1], x0 + side)
    y1 = min(img.shape[0], y0 + side)

    # Re-adjust if the crop clipped at image boundary.
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return img[y0:y1, x0:x1]


def preprocess_spiral_image(img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Crop and resize a scan/spiral to the manuscript's 500x500 grayscale analysis image."""
    out = img.copy()
    if cfg.roi is not None:
        out = crop_to_roi(out, cfg.roi)
    elif cfg.auto_crop:
        out = auto_crop_foreground(out)
    out = resize_square(out, cfg.image_size)
    return out


# -----------------------------
# Patient trace extraction
# -----------------------------

def threshold_dark_pixels(gray: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Return a boolean mask of dark ink/template pixels."""
    if cfg.threshold_method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            10,
        )
        mask = th > 0
    elif cfg.threshold_method == "percentile":
        cutoff = np.percentile(gray, cfg.dark_percentile)
        mask = gray <= cutoff
    elif cfg.threshold_method == "otsu":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = th > 0
    else:
        raise ValueError(f"Unknown threshold_method: {cfg.threshold_method}")

    mask = remove_small_objects(mask.astype(bool), min_size=cfg.min_object_size)
    return mask.astype(bool)


def subtract_template_mask(
    scan_gray: np.ndarray,
    scan_dark: np.ndarray,
    template_gray: Optional[np.ndarray],
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Approximate removal of printed template lines.

    If an empty template is provided, combine two cues:
    1. Difference from the empty template.
    2. Dark pixels that are not on a slightly dilated template mask.

    This is intentionally conservative because exact trace isolation is dataset/template dependent.
    """
    if template_gray is None:
        return scan_dark

    template_gray = preprocess_spiral_image(template_gray, cfg)
    template_dark = threshold_dark_pixels(template_gray, cfg)
    if cfg.template_dilation_px > 0:
        template_dark_dil = ndi.binary_dilation(template_dark, iterations=cfg.template_dilation_px)
    else:
        template_dark_dil = template_dark

    # New dark pixels outside printed template.
    outside_template = scan_dark & ~template_dark_dil

    # Difference-based pixels capture added handwriting that changes intensity relative to template.
    diff = cv2.absdiff(scan_gray, template_gray)
    diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
    # Otsu may fail when differences are subtle; enforce a lower bound on the threshold.
    otsu_val, diff_th = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val < 8:
        diff_th = (diff_blur >= 8).astype(np.uint8) * 255
    diff_mask = (diff_th > 0) & scan_dark

    mask = outside_template | diff_mask
    mask = remove_small_objects(mask.astype(bool), min_size=cfg.min_object_size)
    return mask.astype(bool)


def largest_components(mask: np.ndarray, max_components: int = 3) -> np.ndarray:
    """Keep the largest connected components. Useful to remove dust/text after thresholding."""
    lab = label(mask)
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    if not props:
        return mask.astype(bool)
    keep_labels = {p.label for p in props[:max_components]}
    return np.isin(lab, list(keep_labels))


def extract_trace_mask(
    scan_gray: np.ndarray,
    cfg: PipelineConfig,
    template_gray: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract a binary patient trace mask from a preprocessed grayscale spiral image."""
    scan_dark = threshold_dark_pixels(scan_gray, cfg)
    mask = subtract_template_mask(scan_gray, scan_dark, template_gray, cfg)
    # Keep a few large components because tremor/retracing can fragment the line.
    mask = largest_components(mask, max_components=5)
    return mask.astype(bool)


# -----------------------------
# Metric 1: line length
# -----------------------------

def skeleton_length_pixels(mask: np.ndarray) -> float:
    """
    Length of a skeleton using 8-connectivity.

    Orthogonal edges count as 1 pixel and diagonal edges as sqrt(2) pixels.
    We count each pair once by checking four forward directions.
    """
    skel = skeletonize(mask > 0).astype(bool)
    if skel.sum() == 0:
        return float("nan")

    length = 0.0
    # right, down, down-right, down-left
    directions = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
    ]
    for dy, dx, dist in directions:
        shifted = np.zeros_like(skel, dtype=bool)
        if dy >= 0 and dx >= 0:
            shifted[dy:, dx:] = skel[: skel.shape[0] - dy, : skel.shape[1] - dx]
        elif dy >= 0 and dx < 0:
            shifted[dy:, :dx] = skel[: skel.shape[0] - dy, -dx:]
        length += np.logical_and(skel, shifted).sum() * dist

    return float(length)


def line_length_mm(mask: np.ndarray, dpi: float) -> float:
    pixels = skeleton_length_pixels(mask)
    return pixels * (25.4 / dpi)


# -----------------------------
# Metric 2: Sobel orientation irregularity
# -----------------------------

def circular_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed difference between two angles in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def sobel_orientation_irregularity(gray: np.ndarray, trace_mask: np.ndarray, cfg: PipelineConfig) -> float:
    """
    Sobel-based orientation irregularity score.

    For edge pixels in the isolated trace, compute Sobel gradients, estimate local tangent orientation,
    compare with the radial azimuth from the spiral center, and summarize the circular differences by SD
    in degrees. Higher values indicate more dispersed local orientations.
    """
    if trace_mask.sum() == 0:
        return float("nan")

    # Sobel derivatives. OpenCV returns horizontal/vertical image derivatives.
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    valid_mag = mag[trace_mask]
    if valid_mag.size == 0:
        return float("nan")
    mag_cutoff = np.percentile(valid_mag, cfg.sobel_edge_percentile)
    edge_mask = trace_mask & (mag >= mag_cutoff) & (mag > 0)
    if edge_mask.sum() < 5:
        edge_mask = trace_mask & (mag > 0)
    if edge_mask.sum() < 5:
        return float("nan")

    ys, xs = np.where(edge_mask)
    cy, cx = ndi.center_of_mass(trace_mask.astype(float))
    if not np.isfinite(cx) or not np.isfinite(cy):
        cy = (gray.shape[0] - 1) / 2.0
        cx = (gray.shape[1] - 1) / 2.0

    # Gradient orientation is normal to the edge. Add pi/2 to estimate the tangent/edge orientation.
    edge_tangent_angle = np.arctan2(gy[ys, xs], gx[ys, xs]) + np.pi / 2.0
    radial_azimuth = np.arctan2(ys - cy, xs - cx)
    rel = circular_angle_difference(edge_tangent_angle, radial_azimuth)

    # Circular SD, returned in degrees for easier interpretability.
    R = np.sqrt(np.mean(np.cos(rel)) ** 2 + np.mean(np.sin(rel)) ** 2)
    if R <= 0:
        return 180.0
    circ_sd_rad = np.sqrt(-2.0 * np.log(R))
    return float(np.degrees(circ_sd_rad))


# -----------------------------
# Metric 3: optimal-solution deviation
# -----------------------------

def skeleton_points(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask > 0).astype(bool)
    ys, xs = np.where(skel)
    return np.column_stack([xs, ys]).astype(float)  # x, y


def order_points_nearest_neighbor(points: np.ndarray) -> np.ndarray:
    """
    Greedy ordering of skeleton points. This is a simple, dependency-light way to turn a skeleton into a
    polyline. For production use, replace with graph-based longest-path ordering if the traces branch.
    """
    if len(points) <= 2:
        return points

    # Start from the point farthest from centroid to favor an endpoint-like start.
    centroid = points.mean(axis=0)
    start = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    unused = np.ones(len(points), dtype=bool)
    order = [start]
    unused[start] = False

    current = points[start]
    for _ in range(len(points) - 1):
        idxs = np.where(unused)[0]
        if len(idxs) == 0:
            break
        d2 = np.sum((points[idxs] - current) ** 2, axis=1)
        nxt = idxs[int(np.argmin(d2))]
        order.append(nxt)
        unused[nxt] = False
        current = points[nxt]
    return points[order]


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample ordered points to n equally spaced points along arc length."""
    if len(points) == 0:
        return np.full((n, 2), np.nan)
    if len(points) == 1:
        return np.repeat(points, n, axis=0)

    diffs = np.diff(points, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total <= 0:
        return np.repeat(points[:1], n, axis=0)

    target = np.linspace(0.0, total, n)
    x = np.interp(target, s, points[:, 0])
    y = np.interp(target, s, points[:, 1])
    return np.column_stack([x, y])


def reference_mask_from_image(ref_gray: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    ref_gray = preprocess_spiral_image(ref_gray, cfg)
    mask = threshold_dark_pixels(ref_gray, cfg)
    mask = largest_components(mask, max_components=3)
    return mask


def aggregate_manhattan_deviation(
    trace_mask: np.ndarray,
    reference_mask: Optional[np.ndarray],
    cfg: PipelineConfig,
    dpi: float,
) -> float:
    """
    Aggregate Manhattan distance between corresponding resampled patient-trace and reference points.
    Returned in mm. NaN if no reference is available.
    """
    if reference_mask is None or trace_mask.sum() == 0 or reference_mask.sum() == 0:
        return float("nan")

    trace_pts = skeleton_points(trace_mask)
    ref_pts = skeleton_points(reference_mask)
    if len(trace_pts) < 2 or len(ref_pts) < 2:
        return float("nan")

    trace_ordered = order_points_nearest_neighbor(trace_pts)
    ref_ordered = order_points_nearest_neighbor(ref_pts)

    trace_rs = resample_polyline(trace_ordered, cfg.reference_points)
    ref_rs = resample_polyline(ref_ordered, cfg.reference_points)

    # Account for potential opposite drawing/order direction by taking the lower distance.
    d_forward = np.abs(trace_rs - ref_rs).sum(axis=1).sum()
    d_reverse = np.abs(trace_rs - ref_rs[::-1]).sum(axis=1).sum()
    distance_px = min(d_forward, d_reverse)
    return float(distance_px * (25.4 / dpi))


# -----------------------------
# Metadata handling
# -----------------------------

def resolve_image_path(data_dir: Path, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return data_dir / p


def normalize_hand(value) -> str:
    s = str(value).strip().lower()
    if s in {"r", "right", "right hand", "rh"}:
        return "right"
    if s in {"l", "left", "left hand", "lh"}:
        return "left"
    return s


def infer_is_post(row: pd.Series, time_col: str, drawing_date_col: Optional[str], treatment_date_col: Optional[str]) -> int:
    if time_col in row and pd.notna(row[time_col]):
        s = str(row[time_col]).strip().lower()
        if s in {"post", "after", "followup", "follow-up", "fu", "1", "true", "yes"}:
            return 1
        if s in {"pre", "before", "baseline", "0", "false", "no"}:
            return 0
        # Try numeric values.
        try:
            return int(float(s) > 0)
        except ValueError:
            pass

    if drawing_date_col and treatment_date_col and drawing_date_col in row and treatment_date_col in row:
        drawing = pd.to_datetime(row[drawing_date_col], errors="coerce")
        treatment = pd.to_datetime(row[treatment_date_col], errors="coerce")
        if pd.notna(drawing) and pd.notna(treatment):
            return int(drawing > treatment)

    raise ValueError("Could not infer pre/post status. Provide a usable time column or drawing/treatment dates.")


# -----------------------------
# Per-image processing
# -----------------------------

def compute_metrics_for_row(
    row: pd.Series,
    data_dir: Path,
    cfg: PipelineConfig,
    image_col: str,
    template_gray: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
) -> Dict[str, float | str]:
    img_path = resolve_image_path(data_dir, row[image_col])
    gray_raw = read_grayscale(img_path)
    gray = preprocess_spiral_image(gray_raw, cfg)
    trace_mask = extract_trace_mask(gray, cfg, template_gray=template_gray)

    metrics: Dict[str, float | str] = {
        "resolved_image_path": str(img_path),
        "trace_pixels": int(trace_mask.sum()),
        "line_length_mm": line_length_mm(trace_mask, cfg.dpi),
        "sobel_orientation_irregularity_deg": sobel_orientation_irregularity(gray, trace_mask, cfg),
        "optimal_solution_deviation_mm": aggregate_manhattan_deviation(trace_mask, reference_mask, cfg, cfg.dpi),
    }
    return metrics


def compute_all_metrics(
    metadata: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    image_col: str,
    empty_template_path: Optional[Path] = None,
    ideal_reference_path: Optional[Path] = None,
    save_masks: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    template_gray = None
    if empty_template_path is not None:
        template_gray = read_grayscale(empty_template_path)

    reference_mask = None
    if ideal_reference_path is not None:
        reference_mask = reference_mask_from_image(read_grayscale(ideal_reference_path), cfg)
    elif empty_template_path is not None:
        # Approximation: use the skeletonized empty-template foreground as a reference. For exact
        # replication, provide an ideal midline image using --ideal_reference.
        reference_mask = reference_mask_from_image(read_grayscale(empty_template_path), cfg)

    rows: List[Dict[str, float | str]] = []
    for idx, row in metadata.iterrows():
        try:
            metrics = compute_metrics_for_row(
                row=row,
                data_dir=data_dir,
                cfg=cfg,
                image_col=image_col,
                template_gray=template_gray,
                reference_mask=reference_mask,
            )
            metrics["processing_error"] = ""
        except Exception as exc:  # continue processing other scans
            metrics = {
                "resolved_image_path": str(resolve_image_path(data_dir, row[image_col])),
                "trace_pixels": np.nan,
                "line_length_mm": np.nan,
                "sobel_orientation_irregularity_deg": np.nan,
                "optimal_solution_deviation_mm": np.nan,
                "processing_error": repr(exc),
            }
        rows.append(metrics)

    metrics_df = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    metrics_df.to_csv(output_dir / "spiral_metrics.csv", index=False)
    return metrics_df


# -----------------------------
# Ordinal model: cluster-robust sensitivity analysis
# -----------------------------

def cluster_robust_covariance_ordered(result, groups: Sequence) -> np.ndarray:
    """
    Cluster-robust sandwich covariance for statsmodels OrderedModel.
    Uses per-observation scores and the model's naive covariance as the bread.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    score_obs = result.model.score_obs(result.params)
    bread = np.asarray(result.cov_params())

    unique_groups = groups.dropna().unique()
    meat = np.zeros((score_obs.shape[1], score_obs.shape[1]), dtype=float)
    for g in unique_groups:
        idx = np.where(groups.values == g)[0]
        sg = score_obs[idx, :].sum(axis=0)[:, None]
        meat += sg @ sg.T

    n = len(groups)
    k = score_obs.shape[1]
    G = len(unique_groups)
    correction = 1.0
    if G > 1 and n > k:
        correction = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return correction * bread @ meat @ bread


def fit_ordinal_cluster_model(
    df: pd.DataFrame,
    output_dir: Path,
    subject_col: str,
    crst_col: str,
    hand_col: str,
    treated_hand_col: str,
    time_col: str,
    drawing_date_col: Optional[str] = None,
    treatment_date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Fit ordered logistic regression with cluster-robust SEs by subject."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    model_df = df.copy()

    model_df["crst_score_int"] = pd.to_numeric(model_df[crst_col], errors="coerce").astype("Int64")
    model_df["is_post"] = model_df.apply(
        infer_is_post,
        axis=1,
        time_col=time_col,
        drawing_date_col=drawing_date_col,
        treatment_date_col=treatment_date_col,
    )
    model_df["is_treated_hand"] = (
        model_df[hand_col].map(normalize_hand) == model_df[treated_hand_col].map(normalize_hand)
    ).astype(int)
    model_df["post_x_treated"] = model_df["is_post"] * model_df["is_treated_hand"]

    predictors = [
        "sobel_orientation_irregularity_deg",
        "optimal_solution_deviation_mm",
        "line_length_mm",
        "is_post",
        "is_treated_hand",
        "post_x_treated",
    ]

    # Keep only complete cases.
    keep_cols = [subject_col, "crst_score_int"] + predictors
    model_df = model_df[keep_cols].dropna().copy()
    if model_df.empty:
        raise ValueError("No complete rows are available for model fitting.")

    # OrderedModel should not include an intercept/constant; thresholds are estimated separately.
    y = model_df["crst_score_int"].astype(int)
    X = model_df[predictors].astype(float)

    mod = OrderedModel(y, X, distr="logit")
    res = mod.fit(method="bfgs", disp=False, maxiter=1000)

    cov_cluster = cluster_robust_covariance_ordered(res, model_df[subject_col])
    params = res.params
    se_cluster = pd.Series(np.sqrt(np.diag(cov_cluster)), index=params.index)

    # Only report regression coefficients, not cutpoints.
    coef_names = predictors
    rows = []
    for name in coef_names:
        beta = params[name]
        se = se_cluster[name]
        z = beta / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
        rows.append(
            {
                "term": name,
                "coef_log_odds": beta,
                "cluster_robust_se": se,
                "z": z,
                "p_value": p,
                "odds_ratio": np.exp(beta),
                "or_ci_low": np.exp(ci_low),
                "or_ci_high": np.exp(ci_high),
            }
        )

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "ordinal_cluster_model_odds_ratios.csv", index=False)

    # Save fitted cutpoints and model details too.
    pd.DataFrame({"parameter": params.index, "estimate": params.values, "cluster_robust_se": se_cluster.values}).to_csv(
        output_dir / "ordinal_cluster_model_all_parameters.csv", index=False
    )
    with open(output_dir / "ordinal_cluster_model_statsmodels_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(res.summary()))

    return summary


# -----------------------------
# Optional Bayesian mixed model
# -----------------------------

def fit_bayesian_ordinal_mixed_model(
    df: pd.DataFrame,
    output_dir: Path,
    subject_col: str,
    crst_col: str,
    predictors: Sequence[str],
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
) -> None:
    """
    Optional Bayesian ordered-logit random-intercept model using PyMC.

    Install extras first:
        pip install pymc arviz

    This is closer to the manuscript's random-intercept model than statsmodels OrderedModel, but it is
    Bayesian and may require tuning for your dataset.
    """
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    data = df[[subject_col, crst_col] + list(predictors)].dropna().copy()
    y = pd.to_numeric(data[crst_col]).astype(int).values
    X_raw = data[list(predictors)].astype(float)
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0).replace(0, 1)
    X = X.values
    subj_codes, subj_uniques = pd.factorize(data[subject_col])

    coords = {
        "obs": np.arange(len(data)),
        "predictor": list(predictors),
        "subject": subj_uniques.astype(str),
        "cutpoint": np.arange(len(np.unique(y)) - 1),
    }

    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", mu=0, sigma=2, dims="predictor")
        sigma_subject = pm.HalfNormal("sigma_subject", sigma=1)
        subject_re = pm.Normal("subject_re", mu=0, sigma=sigma_subject, dims="subject")
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2, 2, len(np.unique(y)) - 1),
            sigma=2,
            transform=pm.distributions.transforms.ordered,
            dims="cutpoint",
        )
        eta = pm.Deterministic("eta", pt.dot(X, beta) + subject_re[subj_codes], dims="obs")
        pm.OrderedLogistic("crst", eta=eta, cutpoints=cutpoints, observed=y, dims="obs")
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, output_dir / "bayesian_ordinal_mixed_model.nc")
    az.summary(idata, var_names=["beta", "sigma_subject", "cutpoints"]).to_csv(
        output_dir / "bayesian_ordinal_mixed_model_summary.csv"
    )


# -----------------------------
# CLI
# -----------------------------

def parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if value is None or value == "":
        return None
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute spiral tremor metrics and fit ordinal model.")
    parser.add_argument("--data_dir", required=True, type=Path, help="Folder containing drawing images.")
    parser.add_argument("--metadata_csv", required=True, type=Path, help="CSV with image paths and metadata.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output folder.")

    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--crst_col", default="crst_score")
    parser.add_argument("--hand_col", default="hand")
    parser.add_argument("--treated_hand_col", default="treated_hand")
    parser.add_argument("--time_col", default="time")
    parser.add_argument("--drawing_date_col", default=None)
    parser.add_argument("--treatment_date_col", default=None)

    parser.add_argument("--empty_template", default=None, type=Path, help="Optional empty Spiral A template image.")
    parser.add_argument("--ideal_reference", default=None, type=Path, help="Optional ideal midline/reference image.")
    parser.add_argument("--dpi", default=300.0, type=float)
    parser.add_argument("--image_size", default=500, type=int)
    parser.add_argument("--threshold_method", choices=["otsu", "adaptive", "percentile"], default="otsu")
    parser.add_argument("--roi", default=None, type=parse_roi, help="Optional fixed crop ROI as x,y,w,h.")
    parser.add_argument("--no_auto_crop", action="store_true", help="Disable auto-crop before resizing.")
    parser.add_argument("--fit_model", action="store_true", help="Fit ordinal cluster-robust model after metrics.")
    parser.add_argument("--run_pymc_mixed", action="store_true", help="Also run optional Bayesian mixed model.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = PipelineConfig(
        image_size=args.image_size,
        dpi=args.dpi,
        threshold_method=args.threshold_method,
        auto_crop=not args.no_auto_crop,
        roi=args.roi,
    )

    metadata = pd.read_csv(args.metadata_csv)
    if args.image_col not in metadata.columns:
        raise ValueError(f"Image column '{args.image_col}' not found in metadata CSV.")

    metrics_df = compute_all_metrics(
        metadata=metadata,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        image_col=args.image_col,
        empty_template_path=args.empty_template,
        ideal_reference_path=args.ideal_reference,
    )
    print(f"Wrote metrics to {args.output_dir / 'spiral_metrics.csv'}")

    if args.fit_model:
        summary = fit_ordinal_cluster_model(
            df=metrics_df,
            output_dir=args.output_dir,
            subject_col=args.subject_col,
            crst_col=args.crst_col,
            hand_col=args.hand_col,
            treated_hand_col=args.treated_hand_col,
            time_col=args.time_col,
            drawing_date_col=args.drawing_date_col,
            treatment_date_col=args.treatment_date_col,
        )
        print(f"Wrote ordinal model summary to {args.output_dir / 'ordinal_cluster_model_odds_ratios.csv'}")
        print(summary.to_string(index=False))

    if args.run_pymc_mixed:
        model_df = metrics_df.copy()
        model_df["is_post"] = model_df.apply(
            infer_is_post,
            axis=1,
            time_col=args.time_col,
            drawing_date_col=args.drawing_date_col,
            treatment_date_col=args.treatment_date_col,
        )
        model_df["is_treated_hand"] = (
            model_df[args.hand_col].map(normalize_hand) == model_df[args.treated_hand_col].map(normalize_hand)
        ).astype(int)
        model_df["post_x_treated"] = model_df["is_post"] * model_df["is_treated_hand"]
        predictors = [
            "sobel_orientation_irregularity_deg",
            "optimal_solution_deviation_mm",
            "line_length_mm",
            "is_post",
            "is_treated_hand",
            "post_x_treated",
        ]
        fit_bayesian_ordinal_mixed_model(
            df=model_df,
            output_dir=args.output_dir,
            subject_col=args.subject_col,
            crst_col=args.crst_col,
            predictors=predictors,
        )
        print(f"Wrote PyMC mixed-model outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
