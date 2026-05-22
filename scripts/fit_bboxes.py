"""
fit_bboxes.py
=================
Optional pre-processing step: refine predicted bounding boxes by removing
whitespace padding estimated from prototype masks.

Pipeline
--------
1. For each prototype sprite (grayscale PNG):
   - Threshold → binary mask
   - Dilate once (fills small gaps)
   - Keep only the largest connected component
   - Measure tight bbox → compute padding ratios (left/right/top/bottom)
   - Save cropped sprite to OUTPUT_CROPPED_PROTOS_FOLDER

2. For each predicted bbox in character_measurements/:
   - Look up the sprite_index → padding ratios
   - Shrink the bbox by those ratios (asymmetric trim shifts cx/cy)
   - Write fitted JSON to OUTPUT_BBOX_METRICS_JSON

3. (Optional) Visualise fitted bboxes drawn on line images.

4. (Optional) Side-by-side before/after for a single line.

Usage in notebook
-----------------
    from fit_bboxes import fit_bboxes_pipeline, visualise_fitted_bboxes

    fit_bboxes_pipeline(
        proto_folder          = "input/prototypes/baseline_without_aspect_ratio",
        input_measurements    = "input/character_measurements",
        output_measurements   = "output/character_measurements_fitted",
        output_cropped_protos = "output/prototype_cropped",   # optional
        threshold             = 0.75 * 255,
        sprite_range          = range(0, 120),
        save_ratios_csv       = "output/prototype_bbox_proportions.csv",
    )

    # optional — draws fitted bboxes on line images
    visualise_fitted_bboxes(
        fitted_measurements = "output/character_measurements_fitted",
        line_images            = "dataset/images",
        output_vis_folder      = "output/line_level_with_boxes",
    )

    # optional — side-by-side before/after for one line
    plot_before_after(
        doc              = "btv1b84472995_f009",
        line             = "btv1b84472995_f009_eSc_line_00acb699",
        orig_measurements = "input/character_measurements",
        fitted_measurements = "output/character_measurements_fitted",
        line_images       = "dataset/images",
    )

All functions return useful objects so results can be inspected in-notebook.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label as ndimage_label


# =============================================================================
# STEP 1 — prototype mask → padding ratios
# =============================================================================

def _get_tight_bbox(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (y1, y2, x1, x2) pixel indices of the tight bounding box of the
    foreground in binary_mask, or None if the mask is empty.
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def compute_padding_ratios(
    proto_folder: str | Path,
    sprite_range: Iterable[int] = range(0, 120),
    threshold: float = 0.75 * 255,
    output_cropped_folder: Optional[str | Path] = None,
    save_ratios_csv: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    For each prototype sprite, compute the fraction of whitespace padding on
    each side and return a dict:

        dict_ratios[sprite_index] = (left_ratio, right_ratio, top_ratio, bottom_ratio)

    Parameters
    ----------
    proto_folder : path to folder containing grayscale sprite PNGs named {idx}.png
    sprite_range : which sprite indices to process (default 0–119)
    threshold    : pixel value below which a pixel is considered ink (default 0.75×255)
    output_cropped_folder : if given, save tight-cropped sprites here
    save_ratios_csv       : if given, save a summary CSV to this path
    verbose      : print progress

    Returns
    -------
    dict_ratios : dict mapping sprite_index -> (left, right, top, bottom) ratios
    """
    proto_folder = Path(proto_folder)
    if output_cropped_folder:
        output_cropped_folder = Path(output_cropped_folder)
        output_cropped_folder.mkdir(parents=True, exist_ok=True)

    dict_ratios: Dict[int, Tuple[float, float, float, float]] = {}
    rows = []

    for sprite_idx in sprite_range:
        path = proto_folder / f"{sprite_idx}.png"
        if not path.exists():
            continue

        img = Image.open(path).convert("L")
        arr = np.array(img)
        orig_h, orig_w = arr.shape

        # Threshold + dilation
        mask = (arr < threshold).astype(np.uint8)
        mask = binary_dilation(mask).astype(np.uint8)

        # Largest connected component
        labeled, n = ndimage_label(mask)
        if n == 0:
            if verbose:
                print(f"  sprite {sprite_idx}: no components found, skipping")
            continue

        sizes = [(lbl, int(np.sum(labeled == lbl))) for lbl in range(1, n + 1)]
        largest_label = max(sizes, key=lambda x: x[1])[0]
        largest = (labeled == largest_label).astype(np.uint8)

        # Tight bbox → padding in pixels → padding as fraction of sprite size
        bbox = _get_tight_bbox(largest)
        if bbox is None:
            continue
        y1, y2, x1, x2 = bbox

        left_ratio   = x1            / orig_w
        right_ratio  = (orig_w - 1 - x2) / orig_w
        top_ratio    = y1            / orig_h
        bottom_ratio = (orig_h - 1 - y2) / orig_h

        dict_ratios[sprite_idx] = (left_ratio, right_ratio, top_ratio, bottom_ratio)

        # Save tight-cropped sprite
        if output_cropped_folder:
            cropped = img.crop((x1, y1, x2 + 1, y2 + 1))
            cropped.save(output_cropped_folder / f"{sprite_idx}.png")

        crop_h = y2 - y1 + 1
        crop_w = x2 - x1 + 1

        # Diff map stats (pixels lost by cropping — sanity check)
        cropped_arr = np.array(img.crop((x1, y1, x2 + 1, y2 + 1)))
        restored = np.zeros_like(arr)
        restored[y1:y2+1, x1:x2+1] = cropped_arr
        diff_map    = np.abs(arr.astype(np.int32) - restored.astype(np.int32))
        diff_pixels = int(np.sum(diff_map > 0))

        rows.append({
            "sprite_index":  sprite_idx,
            "orig_w":        orig_w,    "orig_h":        orig_h,
            "crop_w":        crop_w,    "crop_h":        crop_h,
            "left_ratio":    left_ratio,  "right_ratio":   right_ratio,
            "top_ratio":     top_ratio,   "bottom_ratio":  bottom_ratio,
            "lost_pixels":   diff_pixels,
            "lost_percent":  100 * diff_pixels / diff_map.size,
        })

    df_ratios = pd.DataFrame(rows)
    if save_ratios_csv and not df_ratios.empty:
        Path(save_ratios_csv).parent.mkdir(parents=True, exist_ok=True)
        df_ratios.to_csv(save_ratios_csv, index=False)

    if verbose:
        print(f"compute_padding_ratios: {len(dict_ratios)} sprites processed.")

    return dict_ratios


# =============================================================================
# STEP 2 — apply ratios to prediction bboxes
# =============================================================================

def _fit_bbox(
    bbox: Dict[str, float],
    ratios: Tuple[float, float, float, float],
) -> Dict[str, float]:
    """
    Trim whitespace padding from a predicted bbox using prototype ratios.

    Each ratio is the fraction of the sprite image that was empty padding on
    that side. We shrink the bbox by the same fraction of its own size.
    cx/cy are recomputed because asymmetric trimming shifts the centre.

    Returns a new bbox dict with updated cx, cy, w, h.
    The original dict is not modified.
    """
    left_ratio, right_ratio, top_ratio, bottom_ratio = ratios

    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]

    x1 = cx - w / 2 + w * left_ratio
    x2 = cx + w / 2 - w * right_ratio
    y1 = cy - h / 2 + h * top_ratio
    y2 = cy + h / 2 - h * bottom_ratio

    return {
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "w":  x2 - x1,
        "h":  y2 - y1,
    }


def apply_fitting(
    dict_ratios: Dict[int, Tuple[float, float, float, float]],
    input_measurements: str | Path,
    output_measurements: str | Path,
    verbose: bool = True,
) -> Tuple[int, set]:
    """
    Walk input_measurements/, fit every bbox whose sprite_index is in
    dict_ratios, and write fitted JSONs to output_measurements/.

    Parameters
    ----------
    dict_ratios          : output of compute_padding_ratios()
    input_measurements   : path to original character_measurements/ folder
    output_measurements  : path to write fitted JSONs (created if needed)
    verbose              : print progress

    Returns
    -------
    (fitted_files, skipped_sprites)
        fitted_files  : number of JSON files written
        skipped_sprites  : set of sprite indices with no prototype (bbox unchanged)
    """
    input_measurements  = Path(input_measurements)
    output_measurements = Path(output_measurements)

    fitted_files  = 0
    skipped_sprites: set = set()

    for doc in sorted(os.listdir(input_measurements)):
        in_doc  = input_measurements  / doc
        out_doc = output_measurements / doc
        if not in_doc.is_dir():
            continue
        out_doc.mkdir(parents=True, exist_ok=True)

        for fname in sorted(os.listdir(in_doc)):
            if not fname.endswith(".json"):
                continue

            with open(in_doc / fname, "r") as f:
                data = json.load(f)

            for pred in data.get("predictions", []):
                sprite = pred.get("sprite_index")
                if sprite is None:
                    continue
                if sprite in dict_ratios:
                    pred["bbox"] = _fit_bbox(pred["bbox"], dict_ratios[sprite])
                else:
                    skipped_sprites.add(sprite)

            with open(out_doc / fname, "w") as f:
                json.dump(data, f, indent=2)

            fitted_files += 1

    if verbose:
        print(f"apply_fitting: {fitted_files} files written to {output_measurements}")
        if skipped_sprites:
            print(f"  Sprites with no prototype (bbox unchanged): {sorted(skipped_sprites)}")

    return fitted_files, skipped_sprites


# =============================================================================
# CONVENIENCE WRAPPER — runs steps 1 + 2 together
# =============================================================================

def fit_bboxes_pipeline(
    proto_folder: str | Path,
    input_measurements: str | Path,
    output_measurements: str | Path,
    output_cropped_protos: Optional[str | Path] = None,
    threshold: float = 0.75 * 255,
    sprite_range: Iterable[int] = range(0, 120),
    save_ratios_csv: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Run the full bbox-fitting pipeline in one call.

    Parameters
    ----------
    proto_folder          : prototypes/baseline_without_aspect_ratio/
    input_measurements    : original character_measurements/
    output_measurements   : destination for fitted JSONs
    output_cropped_protos : (optional) save tight-cropped sprites here
    threshold             : ink/background threshold (default 0.75 × 255)
    sprite_range          : sprite indices to process (default 0–119)
    save_ratios_csv       : (optional) save padding ratios to this CSV path
    verbose               : print progress

    Returns
    -------
    dict_ratios : the padding-ratio dict (useful for inspection)
    """
    dict_ratios = compute_padding_ratios(
        proto_folder          = proto_folder,
        sprite_range          = sprite_range,
        threshold             = threshold,
        output_cropped_folder = output_cropped_protos,
        save_ratios_csv       = save_ratios_csv,
        verbose               = verbose,
    )

    apply_fitting(
        dict_ratios         = dict_ratios,
        input_measurements  = input_measurements,
        output_measurements = output_measurements,
        verbose             = verbose,
    )

    return dict_ratios


# =============================================================================
# STEP 3 (optional) — visualise fitted bboxes on line images
# =============================================================================

def _draw_bbox(draw: ImageDraw.ImageDraw,
               cx: float, cy: float, w: float, h: float,
               color: str = "red", width: int = 2) -> None:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def visualise_fitted_bboxes(
    fitted_measurements: str | Path,
    line_images: str | Path,
    output_vis_folder: str | Path,
    color: str = "red",
    verbose: bool = True,
) -> None:
    """
    Draw fitted bboxes on their line images and save to output_vis_folder/.

    Parameters
    ----------
    fitted_measurements : output_measurements from apply_fitting()
    line_images            : folder containing per-doc subfolders of line PNGs
    output_vis_folder      : destination for visualisation PNGs
    color                  : bbox outline colour (default "red")
    """
    fitted_measurements = Path(fitted_measurements)
    line_images            = Path(line_images)
    output_vis_folder      = Path(output_vis_folder)
    output_vis_folder.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(fitted_measurements):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            json_path  = Path(root) / fname
            base       = fname[:-5]
            doc        = Path(root).name
            image_path = line_images / doc / (base + ".png")

            if not image_path.exists():
                if verbose:
                    print(f"  no image for {fname}")
                continue

            with open(json_path) as f:
                data = json.load(f)

            img  = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for pred in data.get("predictions", []):
                b = pred["bbox"]
                _draw_bbox(draw, b["cx"], b["cy"], b["w"], b["h"], color=color)

            out_path = output_vis_folder / (base + "_fitted.png")
            img.save(out_path)
            if verbose:
                print(f"  saved: {out_path}")

    if verbose:
        print("visualise_fitted_bboxes: done.")


# =============================================================================
# STEP 4 (optional) — side-by-side before/after for a single line
# =============================================================================

def plot_before_after(
    doc: str,
    line: str,
    orig_measurements: str | Path,
    fitted_measurements: str | Path,
    line_images: str | Path,
    figsize: Tuple[int, int] = (20, 6),
    color: str = "red",
) -> None:
    """
    Show a before/after comparison for a single line image.

    Parameters
    ----------
    doc               : document folder name, e.g. "btv1b84472995_f009"
    line              : line filename stem, e.g. "btv1b84472995_f009_eSc_line_00acb699"
    orig_measurements : original character_measurements/ folder
    fitted_measurements : fitted character_measurements/ folder
    line_images       : folder containing per-doc subfolders of line PNGs
    """
    import matplotlib.pyplot as plt

    orig_measurements = Path(orig_measurements)
    fitted_measurements = Path(fitted_measurements)
    line_images       = Path(line_images)
    image_path        = line_images / doc / (line + ".png")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, meas_folder, title in zip(
        axes,
        [orig_measurements, fitted_measurements],
        ["Before fitting", "After fitting"],
    ):
        json_path = meas_folder / doc / (line + ".json")
        if not json_path.exists():
            ax.set_title(f"{title} — JSON not found")
            ax.axis("off")
            continue
        with open(json_path) as f:
            data = json.load(f)
        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pred in data.get("predictions", []):
            b = pred["bbox"]
            _draw_bbox(draw, b["cx"], b["cy"], b["w"], b["h"], color=color)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.tight_layout()
    plt.show()
