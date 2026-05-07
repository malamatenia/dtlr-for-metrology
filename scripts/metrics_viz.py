"""
metrics_viz.py
======================
Three-layer analysis pipeline for manuscript studies.

USAGE IN NOTEBOOK
-----------------

# ── 0. Config (define once) ──────────────────────────────────────────────────
from metrics_viz import StudyConfig, FolioIndex
from metrics_viz import compute_letter_metrics, compute_bigram_metrics, compute_word_metrics
from metrics_viz import plot_linear, plot_cross

config = StudyConfig(
    annotation_json = "path/to/master.json",
    character_measurements    = "path/to/character_measurements/",
    fig_output_dir   = "path/to/figures/",
    highlight_ranges = [
        ('1r',   '7v'),
        ('170r', '178r'), ('178r', '185v'),
        ('401r', '409r'), ('409r', '416v'),
        ('467r', '474v'), ('474v', '480v'),
    ],
    gathering_labels = {
        ('1r',   '7v'):   "Gath. I",
        ('170r', '178r'): "Gath. XXII",
        ('178r', '185v'): "Gath. XXIII",
        ('401r', '409r'): "Gath. LII",
        ('409r', '416v'): "Gath. LIII",
        ('467r', '474v'): "Gath. LXII",
        ('474v', '480v'): "Gath. LXIII",
    },
)

# ── 1. Build corpus (once per session) ───────────────────────────────────────
# corpus, build_stats, metadata = build_and_crop_working_corpus(...)

# ── 2. Build folio index (once per session) ──────────────────────────────────
idx = FolioIndex(config)

# ── 3. Compute metrics — Option B: accumulate columns into one wide DataFrame ─
df = idx.base_df()                                     # one row per folio, gp, folio cols
df = compute_letter_metrics(df, corpus, "n", config)   # adds mean_ar_n, cv_ar_n, count_n
df = compute_letter_metrics(df, corpus, "a", config)   # adds mean_ar_a, cv_ar_a, count_a
df = compute_bigram_metrics(df, "en", config)          # adds mean_b_distance_m_en, count_en
df = compute_bigram_metrics(df, "et", config)
df = compute_word_metrics(df, config)                  # adds mean_w_distance_normalized_m, etc.

# ── 4. Visualise ─────────────────────────────────────────────────────────────
plot_linear(df, y_metric="mean_ar_n",               config=config, idx=idx)
plot_linear(df, y_metric="mean_b_distance_m_en",    config=config, idx=idx)
plot_cross( df, x_metric="mean_ar_n",
               y_metric="mean_b_distance_m_en",     config=config)
"""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GP_COLORS: Dict[str, str] = {
    "GP1": "#009E73",
    "GP2": "#E69F00",
    "GP3": "#0072B2",
    "GP4": "#CC79A7",
}

# Valid characters for average-letter-width normalisation
VALID_AVG_WIDTH_CHARS = {
    "A","B","C","D","E","F","G","H","I","K","L","M",
    "N","O","P","Q","R","S","T","U","W","X","Y",
    "a","b","c","d","e","f","g","h","i","k","l","m",
    "n","o","p","q","r","s","t","u","w","x","y","z",
    "\u0111","\u0127","\u0142",
    "\u0391","\u03A9",
    "\u1E9C",
    "\ua751","\ua753","\ua758","\ua759",
    "\ua76f","\ua775",
}

# Metric label registry  (column_name → human label for axis / colorbar)
METRIC_LABELS: Dict[str, str] = {
    # ── letter ──────────────────────────────────────────────────────────────
    "mean_ar_{char}":   "Aspect Ratio of '{char}'",
    "std_ar_{char}":    "Std of Aspect Ratio of '{char}'",
    "cv_ar_{char}":     "Coefficient of Variation of Aspect Ratio of '{char}'",
    # ── bigram ──────────────────────────────────────────────────────────────
    "mean_b_distance_m_{bigram}":          "'{bigram}' Bigram Distance",
    "mean_b_distance_avg_letter_{bigram}": "'{bigram}' Bigram Distance (norm. avg letter)",
    "mean_b_distance_px_{bigram}":         "'{bigram}' Bigram Distance (pixels)",
    "mean_b_ar_{bigram}":                  "'{bigram}' Bigram Aspect Ratio",
    "cv_b_distance_{bigram}":              "'{bigram}' Bigram Distance Coefficient of Variation",
    # ── word ────────────────────────────────────────────────────────────────
    "mean_distance":                 "Mean Word Distance (px)",
    "std_distance":                  "Std Word Distance (px)",
    "cv_distance":                   "Coefficient of Variation of Word Distance (px)",
    "mean_w_distance_normalized_m":    "Word Distance",
    "std_distance_normalized_m":     "Std Word Distance",
    "cv_distance_normalized_m":      "Coefficient of Variation of Word Distance",
    "mean_distance_normalized_avg":  "Word Distance (norm. avg letter)",
    "std_distance_normalized_avg":   "Std Word Distance (norm. avg letter)",
    "cv_distance_normalized_avg":    "Coefficient of Variation of Word Distance (norm. avg letter)",
}


def get_metric_label(col: str) -> str:
    """Return a human-readable label for a metric column name."""
    # Direct match
    if col in METRIC_LABELS:
        return METRIC_LABELS[col]
    # Template match: e.g. mean_ar_n → "Mean Aspect Ratio of 'n'"
    for template, label_tpl in METRIC_LABELS.items():
        if "{char}" in template:
            prefix = template.split("{char}")[0]
            if col.startswith(prefix):
                char = col[len(prefix):]
                return label_tpl.replace("{char}", char)
        if "{bigram}" in template:
            prefix = template.split("{bigram}")[0]
            if col.startswith(prefix):
                bigram = col[len(prefix):]
                return label_tpl.replace("{bigram}", bigram)
    return col  # fallback: return column name as-is


# ══════════════════════════════════════════════════════════════════════════════
# STUDY CONFIG  — define once per notebook / manuscript study
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudyConfig:
    """
    All manuscript-level constants in one place.
    Pass this object to every compute_ and plot_ function.
    """
    annotation_json:    str
    character_measurements:        str
    fig_output_dir:      str = "."

    # ── GP ──────────────────────────────────────────────────────────────────
    gp_colors: Dict[str, str] = field(default_factory=lambda: dict(GP_COLORS))

    # ── Folio annotation ────────────────────────────────────────────────────
    highlight_ranges: List[Tuple[str, str]] = field(default_factory=list)
    gathering_labels: Dict[Tuple[str, str], str] = field(default_factory=dict)
    exclude_doc_ids:  List[str] = field(default_factory=list)

    # ── Line selection ───────────────────────────────────────────────────────
    line_selection_mode: str = None  # all | MainZone#1 | MainZone#2 | recto_verso

    # ── Filtering ────────────────────────────────────────────────────────────
    bbox_filter_mode:      Optional[str]   = "threshold"  # None | percentage | threshold
    outlier_std_threshold: Optional[float] = None
    width_height_tol:      Tuple           = (None, None)

    # ── Prototype overlay (for plot_cross) ──────────────────────────────────
    prototypes: Optional[str] = None   # folder containing per-doc prototype subfolders

    # ── Markers — used by plot_linear when plotting multiple series ──────────
    # Map: character/bigram → matplotlib marker string
    letter_markers: Dict[str, str] = field(default_factory=lambda: {
        'a': 's', 't': 'x', 'd': '*', 'e': 'P', 'n': 'o',
        'i': '^', 'r': 'v', 'u': 'D', 'm': 'h', 's': 'p',
    })
    bigram_markers: Dict[str, str] = field(default_factory=lambda: {
        'en': 'o', 'et': 's', 'de': 'p', 'er': '^', 'es': 'v',
    })
    # Fallback cycle for any series not in the dicts above
    _default_markers: List[str] = field(
        default_factory=lambda: ['o','s','^','D','v','P','X','*','h','p'],
        repr=False
    )

    # ── Plot defaults ────────────────────────────────────────────────────────
    xlabel_fontsize:       int = 25
    ylabel_fontsize:       int = 25
    colorbar_label_size:   int = 18
    colorbar_tick_size:    int = 15
    legend_fontsize:       int = 15
    annotation_fontsize:   int = 15

    def __post_init__(self):
        os.makedirs(self.fig_output_dir, exist_ok=True)

    def get_marker(self, series_key: str, fallback_idx: int = 0) -> str:
        """Return marker for a letter or bigram series key."""
        if series_key in self.letter_markers:
            return self.letter_markers[series_key]
        if series_key in self.bigram_markers:
            return self.bigram_markers[series_key]
        return self._default_markers[fallback_idx % len(self._default_markers)]


# ══════════════════════════════════════════════════════════════════════════════
# FOLIO INDEX  — built once, reused everywhere
# ══════════════════════════════════════════════════════════════════════════════

class FolioIndex:
    """
    Builds and caches all folio-level mappings from master JSON.
    Pass to plot_linear() so it knows the global folio order.
    """

    def __init__(self, config: StudyConfig):
        self.config = config
        self.doc_mappings,  _ = _build_doc_mappings(config.annotation_json)
        self.line_mappings, _ = _build_line_mappings(config.annotation_json)
        self.doc_line_order   = _build_doc_line_order(config.annotation_json)

        # folio ↔ doc ↔ GP
        self.folio_to_doc: Dict[str, str] = {}
        self.doc_to_gp:    Dict[str, str] = {}
        for doc, folio in self.doc_mappings["folio"].items():
            if folio:
                self.folio_to_doc[folio] = doc
                self.doc_to_gp[doc] = self.doc_mappings["gp"].get(doc, "Unknown")

        self.all_folios_sorted: List[str] = sorted(
            self.folio_to_doc.keys(), key=folio_sort_key
        )
        self.folio_status: Dict[str, str] = {
            folio: ("excluded" if self.folio_to_doc[folio] in config.exclude_doc_ids
                    else "included")
            for folio in self.all_folios_sorted
        }
        # x-position for folio progression plots
        self.x_positions: Dict[str, int] = {
            folio: i for i, folio in enumerate(self.all_folios_sorted)
        }

    def base_df(self) -> pd.DataFrame:
        """
        Return an empty-metrics DataFrame: one row per folio,
        with columns [folder, folio, gp].
        All compute_*_metrics() functions add columns to this.
        """
        rows = []
        for folio, doc in self.folio_to_doc.items():
            gp = self.doc_to_gp.get(doc)
            if gp:
                rows.append({"folder": doc, "folio": folio, "gp": gp})
        return pd.DataFrame(rows).sort_values("folio", key=lambda s: s.map(folio_sort_key))

    def get_range_gp_color(self, beg_folio: str, end_folio: str) -> Tuple[str, Optional[str]]:
        """Resolve the GP color for a highlight range."""
        colors = self.config.gp_colors
        beg_doc = self.folio_to_doc.get(beg_folio)
        if beg_doc:
            beg_gp = self.doc_to_gp.get(beg_doc)
            if beg_gp and beg_gp in colors:
                return colors[beg_gp], beg_gp
        fs = self.all_folios_sorted
        if beg_folio in fs and end_folio in fs:
            i_lo = min(fs.index(beg_folio), fs.index(end_folio))
            i_hi = max(fs.index(beg_folio), fs.index(end_folio))
            gp_counts: Dict[str, int] = {}
            for f in fs[i_lo:i_hi + 1]:
                doc = self.folio_to_doc.get(f)
                gp  = self.doc_to_gp.get(doc) if doc else None
                if gp:
                    gp_counts[gp] = gp_counts.get(gp, 0) + 1
            if gp_counts:
                mgp = max(gp_counts, key=gp_counts.get)
                if mgp in colors:
                    return colors[mgp], mgp
        return "#888888", None


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS  — JSON loading, file selection
# ══════════════════════════════════════════════════════════════════════════════

def folio_sort_key(folio) -> tuple:
    """Natural sort key: '12v' → (12, 1, 0), '3ra' → (3, 0, 1)."""
    m = re.match(r'(\d+)([rv])([ab])?', str(folio).lower())
    if m:
        return (int(m.group(1)),
                {'r': 0, 'v': 1}.get(m.group(2), 2),
                {'': 0, 'a': 1, 'b': 2}.get(m.group(3) or '', 0))
    m2 = re.match(r'(\d+)', str(folio))
    return (int(m2.group(1)), 999, 999) if m2 else (999999, 999, 999)


def _build_doc_mappings(annotation_json: str, doc_fields=("gp", "folio")):
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings   = defaultdict(dict)
    conflicts  = defaultdict(lambda: defaultdict(set))
    for key, entry in data.items():
        doc = "_".join(key.replace(".png", "").split("_")[:2])
        for field in doc_fields:
            val = entry.get(field)
            if val is None:
                continue
            if doc in mappings[field] and mappings[field][doc] != val:
                conflicts[field][doc].add(mappings[field][doc])
                conflicts[field][doc].add(val)
            else:
                mappings[field][doc] = val
    return mappings, conflicts


def _build_line_mappings(annotation_json: str, line_fields=("line", "zone")):
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings  = defaultdict(dict)
    conflicts = defaultdict(lambda: defaultdict(set))
    for key, entry in data.items():
        lk = key.replace(".png", "")
        for field in line_fields:
            val = entry.get(field)
            if val is None:
                continue
            if lk in mappings[field] and mappings[field][lk] != val:
                conflicts[field][lk].add(mappings[field][lk])
                conflicts[field][lk].add(val)
            else:
                mappings[field][lk] = val
    return mappings, conflicts


def _build_doc_line_order(annotation_json: str) -> Dict[str, List[str]]:
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    order: Dict[str, List[str]] = defaultdict(list)
    for key in data:
        doc = "_".join(key.replace(".png", "").split("_")[:2])
        order[doc].append(key.replace(".png", ""))
    return order


def _select_files(doc_folder, doc_path, doc_line_order, line_mappings, folio,
                  line_selection_mode) -> List[str]:
    """Return ordered list of .json filenames for this doc, filtered by zone."""
    json_files = {f.replace(".json", ""): f
                  for f in os.listdir(doc_path) if f.endswith(".json")}
    ordered = [json_files[k] for k in doc_line_order.get(doc_folder, []) if k in json_files]
    default_lines = [f for f in ordered
                     if line_mappings["line"].get(f.replace(".json", "")) == "DefaultLine"]

    if line_selection_mode == "MainZone#1":
        return [f for f in default_lines
                if line_mappings["zone"].get(f.replace(".json", "")) == "MainZone#1"]
    if line_selection_mode == "MainZone#2":
        return [f for f in default_lines
                if line_mappings["zone"].get(f.replace(".json", "")) == "MainZone#2"]
    if line_selection_mode == "recto_verso" and folio:
        if folio.endswith(("r", "ra", "rb")):
            zone = "MainZone#2"
        elif folio.endswith(("v", "va", "vb")):
            zone = "MainZone#1"
        else:
            zone = None
        return ([f for f in default_lines
                 if line_mappings["zone"].get(f.replace(".json", "")) == zone]
                if zone else default_lines)
    return default_lines


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — METRICS  (each function adds columns to the wide DataFrame)
# ══════════════════════════════════════════════════════════════════════════════

def compute_letter_metrics(
    df: pd.DataFrame,
    corpus: dict,
    char: str,
    config: StudyConfig,
) -> pd.DataFrame:
    """
    Add letter-level metrics for `char` to df.

    New columns added
    -----------------
    mean_ar_{char}   — mean aspect ratio (w/h)
    std_ar_{char}    — std of aspect ratio
    cv_ar_{char}     — coefficient of variation of aspect ratio
    count_{char}     — number of filtered instances

    Parameters
    ----------
    df     : base DataFrame from FolioIndex.base_df(), or already enriched df
    corpus : dict returned by build_and_crop_working_corpus()
    char   : single character, e.g. "n"
    config : StudyConfig
    """
    col_mean  = f"mean_ar_{char}"
    col_std   = f"std_ar_{char}"
    col_cv    = f"cv_ar_{char}"
    col_count = f"count_{char}"

    # Gather per-doc data from corpus
    rows: Dict[str, dict] = {}
    for (doc_folder, c), data in corpus.items():
        if c != char:
            continue
        ratios = data.get("ratios", [])
        if not ratios:
            continue
        r = np.array(ratios)

        # Apply bbox filter from config
        if config.bbox_filter_mode == "threshold" and config.outlier_std_threshold:
            ws = np.array(data.get("widths", []))
            hs = np.array(data.get("heights", []))
            if len(ws):
                mw, sw = np.mean(ws), np.std(ws)
                mh, sh = np.mean(hs), np.std(hs)
                t = config.outlier_std_threshold
                mask = ((ws >= mw - t*sw) & (ws <= mw + t*sw) &
                        (hs >= mh - t*sh) & (hs <= mh + t*sh))
                r = r[mask]

        elif config.bbox_filter_mode == "percentage":
            ws = np.array(data.get("widths", []))
            hs = np.array(data.get("heights", []))
            wt, ht = config.width_height_tol
            if len(ws):
                mask = np.ones(len(ws), dtype=bool)
                if wt:
                    mask &= np.abs(ws - np.median(ws)) / np.median(ws) <= wt
                if ht:
                    mask &= np.abs(hs - np.median(hs)) / np.median(hs) <= ht
                r = r[mask]

        if len(r) == 0:
            continue

        mean = float(np.mean(r))
        std  = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        cv   = std / mean if mean != 0 else np.nan

        rows[doc_folder] = {
            col_mean:  mean,
            col_std:   std,
            col_cv:    cv,
            col_count: len(r),
        }

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})

    return df.merge(metric_df, on="folder", how="left")


def compute_bigram_metrics(
    df: pd.DataFrame,
    bigram: str,
    config: StudyConfig,
    idx: Optional[FolioIndex] = None,
) -> pd.DataFrame:
    """
    Add bigram-level spacing metrics for `bigram` to df.
    Re-reads JSON files (sequence information required).

    New columns added
    -----------------
    mean_b_distance_m_{bigram}          — distance norm. by ½m width
    mean_b_distance_avg_letter_{bigram} — distance norm. by avg letter width
    mean_b_distance_px_{bigram}         — raw pixel distance
    mean_b_ar_{bigram}                  — bigram aspect ratio
    cv_b_distance_{bigram}              — CV of distances
    count_{bigram}                      — number of filtered instances
    """
    # Build index if not provided
    if idx is None:
        idx = FolioIndex(config)

    l1, l2 = bigram[0].lower(), bigram[1].lower()

    first_pass:     Dict[str, list] = defaultdict(list)
    folio_m_widths: Dict[str, list] = defaultdict(list)
    folio_lw:       Dict[str, list] = defaultdict(list)

    for doc_folder in os.listdir(config.character_measurements):
        folio = idx.doc_mappings["folio"].get(doc_folder)
        if doc_folder in config.exclude_doc_ids or folio in config.exclude_doc_ids:
            continue
        doc_path = os.path.join(config.character_measurements, doc_folder)
        if not os.path.isdir(doc_path) or doc_folder not in idx.doc_line_order:
            continue

        files = _select_files(doc_folder, doc_path, idx.doc_line_order,
                              idx.line_mappings, folio, config.line_selection_mode)

        for fname in files:
            try:
                with open(os.path.join(doc_path, fname)) as f:
                    data = json.load(f)
            except Exception:
                continue

            preds = sorted(data.get("predictions", []),
                           key=lambda x: x.get("bbox", {}).get("cx", 0))

            for p in preds:
                char = p.get("character", "")
                if p.get("error_label") != "match" or "bbox" not in p:
                    continue
                w = p["bbox"].get("w", 0)
                if w <= 0:
                    continue
                if char.lower() == "m":
                    folio_m_widths[doc_folder].append(w)
                if char in VALID_AVG_WIDTH_CHARS:
                    folio_lw[doc_folder].append(w)

            for i in range(len(preds) - 1):
                p1, p2 = preds[i], preds[i + 1]
                if (p1.get("character", "").lower() == l1 and
                        p2.get("character", "").lower() == l2 and
                        p1.get("error_label") == "match" and
                        p2.get("error_label") == "match"):
                    lo  = (i == 0)           or (preds[i - 1].get("error_label") == "match")
                    ro  = (i + 2 >= len(preds)) or (preds[i + 2].get("error_label") == "match")
                    if not (lo and ro):
                        continue
                    b1, b2 = p1["bbox"], p2["bbox"]
                    if all(k in b1 for k in ("cx","w","h")) and all(k in b2 for k in ("cx","w","h")):
                        dist = (b2["cx"] - b2["w"]/2) - (b1["cx"] + b1["w"]/2)
                        cw   = (b2["cx"] + b2["w"]/2) - (b1["cx"] - b1["w"]/2)
                        ch   = max(b1["h"], b2["h"])
                        first_pass[doc_folder].append({
                            "distance": dist, "combined_w": cw, "combined_h": ch,
                            "w1": b1["w"], "h1": b1["h"], "w2": b2["w"], "h2": b2["h"]
                        })

    # Outlier filtering
    bounds: dict = {}
    if config.bbox_filter_mode == "threshold" and config.outlier_std_threshold:
        all_vals = {k: [e[k] for ex in first_pass.values() for e in ex]
                    for k in ("w1","h1","w2","h2")}
        for k, arr in all_vals.items():
            if arr:
                m, s = np.mean(arr), np.std(arr)
                t    = config.outlier_std_threshold
                bounds[k] = (m - t*s, m + t*s)

    sfx = bigram  # column suffix
    rows: Dict[str, dict] = {}

    for doc, examples in first_pass.items():
        keep = [e for e in examples
                if all(bounds[k][0] <= e[k] <= bounds[k][1] for k in bounds)]
        if not keep:
            continue

        dists     = [e["distance"] for e in keep]
        mean_px   = float(np.mean(dists))
        std_px    = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
        cv        = std_px / mean_px if mean_px != 0 else np.nan
        mean_ar   = float(np.mean([e["combined_w"] / e["combined_h"] for e in keep]))

        mean_norm_m   = None
        mean_norm_avg = None

        if folio_m_widths.get(doc):
            m_half = np.mean(folio_m_widths[doc]) / 2
            if m_half > 0:
                mean_norm_m = mean_px / m_half

        if folio_lw.get(doc):
            avg_lw = np.mean(folio_lw[doc])
            if avg_lw > 0:
                mean_norm_avg = mean_px / avg_lw

        rows[doc] = {
            f"mean_b_distance_m_{sfx}":          mean_norm_m,
            f"mean_b_distance_avg_letter_{sfx}":  mean_norm_avg,
            f"mean_b_distance_px_{sfx}":           mean_px,
            f"mean_b_ar_{sfx}":                    mean_ar,
            f"cv_b_distance_{sfx}":                cv,
            f"count_{sfx}":                        len(keep),
        }

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})

    return df.merge(metric_df, on="folder", how="left")


def compute_word_metrics(
    df: pd.DataFrame,
    config: StudyConfig,
    idx: Optional[FolioIndex] = None,
) -> pd.DataFrame:
    """
    Add word-spacing metrics per folio to df.
    Re-reads JSON files.

    New columns added (all three normalisations × mean/std/cv)
    ----------------------------------------------------------
    mean_distance, std_distance, cv_distance
    mean_w_distance_normalized_m, std_distance_normalized_m, cv_distance_normalized_m
    mean_distance_normalized_avg, std_distance_normalized_avg, cv_distance_normalized_avg
    n_word_distances
    """
    if idx is None:
        idx = FolioIndex(config)

    def _word_distances(predictions):
        lines: Dict[int, list] = defaultdict(list)
        for p in predictions:
            if "bbox" in p:
                lines[p.get("line_id", 0)].append(p)
        dists = []
        for chars in lines.values():
            chars = sorted(chars, key=lambda p: p["bbox"]["cx"])
            words, cur = [], []
            for ch in chars:
                if ch.get("character") == " ":
                    if cur:
                        words.append(cur); cur = []
                else:
                    cur.append(ch)
            if cur:
                words.append(cur)
            for w1, w2 in zip(words, words[1:]):
                last, first = w1[-1]["bbox"], w2[0]["bbox"]
                d = (first["cx"] - first["w"]/2) - (last["cx"] + last["w"]/2)
                if d > 0:
                    dists.append(d)
        return dists

    rows: Dict[str, dict] = {}

    for doc_folder in os.listdir(config.character_measurements):
        folio = idx.doc_mappings["folio"].get(doc_folder)
        if doc_folder in config.exclude_doc_ids or folio in config.exclude_doc_ids:
            continue
        doc_path = os.path.join(config.character_measurements, doc_folder)
        if not os.path.isdir(doc_path) or doc_folder not in idx.doc_line_order:
            continue

        files = _select_files(doc_folder, doc_path, idx.doc_line_order,
                              idx.line_mappings, folio, config.line_selection_mode)

        all_distances, all_m_widths, all_lw = [], [], []

        for jf in files:
            try:
                with open(os.path.join(doc_path, jf)) as f:
                    data = json.load(f)
            except Exception:
                continue

            predictions = data.get("predictions", [])
            valid_preds = []
            for i, p in enumerate(predictions):
                if p.get("error_label") != "match" or "bbox" not in p:
                    continue
                lo = (i == 0)                  or (predictions[i-1].get("error_label") == "match")
                ro = (i+1 >= len(predictions)) or (predictions[i+1].get("error_label") == "match")
                if lo and ro:
                    valid_preds.append(p)

            for p in valid_preds:
                char = p.get("character", "")
                w    = p["bbox"].get("w", 0)
                if w > 0:
                    if char == "m":
                        all_m_widths.append(w)
                    if char in VALID_AVG_WIDTH_CHARS:
                        all_lw.append(w)

            all_distances.extend(_word_distances(valid_preds))

        if not all_distances:
            continue

        mean_px = float(np.mean(all_distances))
        std_px  = float(np.std(all_distances, ddof=1)) if len(all_distances) > 1 else 0.0
        cv_px   = std_px / mean_px if mean_px > 0 else np.nan

        mean_norm_m = std_norm_m = cv_norm_m = None
        if all_m_widths:
            m_half = np.mean(all_m_widths) / 2
            if m_half > 0:
                nd = [d / m_half for d in all_distances]
                mean_norm_m = float(np.mean(nd))
                std_norm_m  = float(np.std(nd, ddof=1)) if len(nd) > 1 else 0.0
                cv_norm_m   = std_norm_m / mean_norm_m if mean_norm_m > 0 else np.nan

        mean_norm_avg = std_norm_avg = cv_norm_avg = None
        if all_lw:
            avg_lw = np.mean(all_lw)
            if avg_lw > 0:
                nd = [d / avg_lw for d in all_distances]
                mean_norm_avg = float(np.mean(nd))
                std_norm_avg  = float(np.std(nd, ddof=1)) if len(nd) > 1 else 0.0
                cv_norm_avg   = std_norm_avg / mean_norm_avg if mean_norm_avg > 0 else np.nan

        rows[doc_folder] = {
            "mean_distance":                mean_px,
            "std_distance":                 std_px,
            "cv_distance":                  cv_px,
            "mean_w_distance_normalized_m":   mean_norm_m,
            "std_distance_normalized_m":    std_norm_m,
            "cv_distance_normalized_m":     cv_norm_m,
            "mean_distance_normalized_avg": mean_norm_avg,
            "std_distance_normalized_avg":  std_norm_avg,
            "cv_distance_normalized_avg":   cv_norm_avg,
            "n_word_distances":             len(all_distances),
        }

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})
    return df.merge(metric_df, on="folder", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — SHARED VIZ HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_alpha_colorbar(fig, base_color, vmin, vmax, label, position,
                        label_fontsize=14, tick_fontsize=11):
    N        = 256
    base_rgb = np.array(mcolors.to_rgb(base_color))
    alphas   = np.linspace(0.2, 1.0, N)
    blended  = (1 - alphas)[:,None] * np.ones(3) + alphas[:,None] * base_rgb
    cmap     = mcolors.ListedColormap(blended)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm       = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax  = fig.add_axes(position)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(label, fontsize=label_fontsize, color='black', labelpad=8)
    cbar.ax.tick_params(labelsize=tick_fontsize, colors=base_color)
    return cbar


def _draw_colorbars(fig, df_sub_fn, gp_colors, count_min, count_max, config,
                    bar_width=0.02, left=1.02):
    """Draw stacked alpha colorbars for each GP."""
    gps_cb     = ["GP4", "GP3", "GP2", "GP1"]
    bar_height = 0.20
    spacing    = 0.02
    total_h    = len(gps_cb) * bar_height + (len(gps_cb) - 1) * spacing
    for i, gp in enumerate(gps_cb):
        if df_sub_fn(gp).empty:
            continue
        bottom = (0.5 - total_h / 2) + i * (bar_height + spacing)
        _add_alpha_colorbar(
            fig, gp_colors[gp], count_min, count_max,
            f"{gp} Occ.",
            [left, bottom, bar_width, bar_height],
            label_fontsize=config.colorbar_label_size,
            tick_fontsize=config.colorbar_tick_size,
        )

'''
def _draw_gp_
(ax, gp_colors, config, loc="upper left", bbox_to_anchor=None):
    """Horizontal single-row GP legend."""
    elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=gp_colors[gp], markersize=10, label=gp)
        for gp in gp_colors
    ]
    kw = dict(
        handles=elements, ncol=len(elements), frameon=True, framealpha=0.9,
        fontsize=config.legend_fontsize, handletextpad=0.3, columnspacing=0.6, loc=loc,
    )
    if bbox_to_anchor:
        kw["bbox_to_anchor"] = bbox_to_anchor
    return ax.legend(**kw)
'''

def _draw_highlight_ranges(ax, config: StudyConfig, idx: FolioIndex):
    """Draw shaded gathering ranges + dashed edges + gathering labels."""
    for beg, end in config.highlight_ranges:
        x_beg = idx.x_positions.get(beg)
        x_end = idx.x_positions.get(end)
        color, _ = idx.get_range_gp_color(beg, end)
        label = config.gathering_labels.get((beg, end), f"{beg}–{end}")

        if x_beg is not None and x_end is not None:
            x_lo, x_hi = min(x_beg, x_end), max(x_beg, x_end)
            ax.axvspan(x_lo, x_hi, color=color, alpha=0.10, zorder=0, linewidth=0)
            for xe in [x_lo, x_hi]:
                ax.axvline(x=xe, color=color, linestyle='--',
                           alpha=0.55, linewidth=1.4, zorder=1)
            ax.annotate(
                label,
                xy=((x_lo + x_hi) / 2, 0.95),
                xycoords=(ax.get_xaxis_transform(), 'axes fraction'),
                xytext=(0, 0), textcoords='offset points',
                ha='center', va='bottom',
                fontsize=config.annotation_fontsize,
                color=color, fontstyle='italic', zorder=5
            )
        elif x_beg is not None:
            ax.axvline(x=x_beg, color=color, linestyle='--', alpha=0.55, linewidth=1.4, zorder=1)
        elif x_end is not None:
            ax.axvline(x=x_end, color=color, linestyle='--', alpha=0.55, linewidth=1.4, zorder=1)


def _draw_folio_xaxis(ax, idx: FolioIndex, highlight_folios: Optional[List[str]] = None):
    """Set x-ticks to ordered folios, colour excluded ones red."""
    highlight_folios = highlight_folios or []
    ax.set_xlim(-0.5, len(idx.all_folios_sorted) - 0.5)
    ax.set_xticks(range(len(idx.all_folios_sorted)))
    ax.set_xticklabels(idx.all_folios_sorted, rotation=90, fontsize=9)
    for tick_label, folio in zip(ax.get_xticklabels(), idx.all_folios_sorted):
        if idx.folio_status.get(folio) == 'excluded':
            tick_label.set_color('red')
    for folio in highlight_folios:
        if folio in idx.x_positions:
            ax.axvline(x=idx.x_positions[folio], color='darkgrey',
                       linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    for folio, status in idx.folio_status.items():
        if status == 'excluded' and folio in idx.x_positions:
            ax.axvline(x=idx.x_positions[folio], color='red',
                       linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — plot_linear
# ══════════════════════════════════════════════════════════════════════════════

def plot_linear(
    df: pd.DataFrame,
    y_metric,                               # str OR list[str] — one or multiple series
    config: StudyConfig,
    idx: FolioIndex,
    count_col: Optional[str] = None,        # column driving alpha; auto-detected if None
    highlight_folios: Optional[List[str]] = None,
    y_label_override: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 6),
):
    """
    Folio-progression line plot.

    Single series
    -------------
    plot_linear(df, y_metric="mean_ar_n", ...)

    Multiple series on shared axes (same metric type, different units)
    ------------------------------------------------------------------
    plot_linear(df, y_metric=["mean_ar_n", "mean_ar_a", "mean_ar_i"], ...)
    plot_linear(df, y_metric=["mean_b_distance_m_en", "mean_b_distance_m_et"], ...)

    Each series gets its own marker (from config.letter_markers /
    config.bigram_markers) and appears in a series legend.
    GP color and count-based alpha still apply.
    """
    # ── normalise to list ─────────────────────────────────────────────────────
    if isinstance(y_metric, str):
        y_metrics = [y_metric]
    else:
        y_metrics = list(y_metric)

    for ym in y_metrics:
        if ym not in df.columns:
            raise ValueError(f"Column '{ym}' not found. Available: {list(df.columns)}")

    gp_colors        = config.gp_colors
    highlight_folios = highlight_folios or []
    multi            = len(y_metrics) > 1

    # ── per-series: series_key used to look up marker ────────────────────────
    def _series_key(col: str) -> str:
        """Extract the letter/bigram suffix from a column name, e.g. mean_ar_n → n"""
        # letter: mean_ar_X, std_ar_X, cv_ar_X
        for prefix in ("mean_ar_", "std_ar_", "cv_ar_"):
            if col.startswith(prefix):
                return col[len(prefix):]
        # bigram: mean_b_distance_m_XY, mean_b_ar_XY, etc.
        for prefix in ("mean_b_distance_m_", "mean_b_distance_avg_letter_",
                       "mean_b_distance_px_", "mean_b_ar_", "cv_b_distance_"):
            if col.startswith(prefix):
                return col[len(prefix):]
        # word: no suffix needed — use full col as key
        return col

    # ── shared count column — use first available across all metrics ──────────
    if count_col is None:
        for ym in y_metrics:
            candidates = [c for c in df.columns if c.startswith("count")]
            if candidates:
                count_col = candidates[0]
                break
    if count_col is None or count_col not in df.columns:
        df = df.copy()
        df["_count_ones"] = 1
        count_col = "_count_ones"

    # ── global count range across all series (consistent alpha scale) ─────────
    all_valid = df.dropna(subset=y_metrics[:1] + [count_col])
    if all_valid.empty:
        print(f"⚠️ No data for metric(s) {y_metrics}.")
        return

    count_min  = df[count_col].dropna().min()
    count_max  = df[count_col].dropna().max()
    count_norm = mcolors.Normalize(vmin=count_min, vmax=count_max)

    # Shared folio→gp/count maps (use first metric's clean rows as base)
    folio_to_gp    = dict(zip(df["folio"], df["gp"]))
    folio_to_count = dict(zip(df["folio"], df[count_col]))

    fig, ax = plt.subplots(figsize=figsize)

    # ── 1. Gathering highlight ranges ─────────────────────────────────────────
    _draw_highlight_ranges(ax, config, idx)

    # ── 2. X-axis ─────────────────────────────────────────────────────────────
    _draw_folio_xaxis(ax, idx, highlight_folios)

    # ── 3 & 4. Per-series: connecting lines + scatter ─────────────────────────
    series_legend_elements = []

    for s_idx, ym in enumerate(y_metrics):
        sk     = _series_key(ym)
        marker = config.get_marker(sk, fallback_idx=s_idx)
        df_s   = df.dropna(subset=[ym, count_col])

        folio_to_y     = dict(zip(df_s["folio"], df_s[ym]))
        folios_w_data  = [f for f in idx.all_folios_sorted if f in folio_to_y]

        # Connecting lines (grey, per-series)
        for i in range(len(folios_w_data) - 1):
            f1, f2 = folios_w_data[i], folios_w_data[i + 1]
            gap = idx.all_folios_sorted.index(f2) - idx.all_folios_sorted.index(f1)
            ls  = '--' if gap > 1 else '-'
            ax.plot([idx.x_positions[f1], idx.x_positions[f2]],
                    [folio_to_y[f1], folio_to_y[f2]],
                    ls, color='grey', alpha=0.25, zorder=2)

        # Scatter points — GP color, count alpha, per-series marker
        for gp, color in gp_colors.items():
            for folio in folios_w_data:
                if folio_to_gp.get(folio) != gp:
                    continue
                cnt   = folio_to_count.get(folio, count_min)
                alpha = 0.2 + 0.8 * count_norm(cnt if pd.notna(cnt) else count_min)
                ax.scatter(idx.x_positions[folio], folio_to_y[folio],
                           color=color, alpha=alpha, s=80, marker=marker,
                           zorder=3, linewidths=0.5)

        # Series legend element (only when multi-series)
        if multi:
                label = _series_key(ym)  # only letter or bigram
                series_legend_elements.append(
                Line2D([0], [0], marker=marker, linestyle='',
                       markersize=9, markeredgecolor='0.35',
                       markerfacecolor='dimgray', markeredgewidth=1.2,
                       label=label)
            )

    # ── 5. Colorbars ──────────────────────────────────────────────────────────
    df_for_cb = df.dropna(subset=[y_metrics[0], count_col])
    _draw_colorbars(fig, lambda gp: df_for_cb[df_for_cb["gp"] == gp],
                    gp_colors, count_min, count_max, config, bar_width=0.015, left=1.01)

    # ── 6. Series marker legend (multi-series only) ───────────────────────────
    if multi and series_legend_elements:
        ax.legend(
            handles       = series_legend_elements,
            loc           = 'lower center',
            ncol          = len(series_legend_elements),
            frameon       = True,
            framealpha    = 0.9,
            fontsize      = config.legend_fontsize,
            handletextpad = 0.3,
            columnspacing = 0.6,
        )

    # ── 7. Axis labels ────────────────────────────────────────────────────────
    if y_label_override:
        y_label = y_label_override
    elif multi:
        y_label = "Metric value"   # generic when multiple series share the axis
    else:
        y_label = get_metric_label(y_metrics[0])

    ax.set_xlabel("Ordered Pages", fontsize=config.xlabel_fontsize)
    ax.set_ylabel(y_label,          fontsize=config.ylabel_fontsize)

    plt.tight_layout()

    if save:
        safe = "_".join(y_metrics) if multi else y_metrics[0]
        # cap filename length
        if len(safe) > 80:
            safe = safe[:80]
        fname = f"linear_{safe}_{config.line_selection_mode}.jpeg"
        plt.savefig(os.path.join(config.fig_output_dir, fname), dpi=300)
        print(f"✅ Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — plot_cross
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    config: StudyConfig,
    count_col: Optional[str] = None,
    x_label_override: Optional[str] = None,
    y_label_override: Optional[str] = None,
    overlay_prototypes: bool = False,       # True → show prototype image at each point
    proto_char: Optional[str] = None,       # which character's sprite to show (e.g. "n")
    character_sprite_map: Optional[Dict[str, str]] = None,  # char → sprite filename
    save: bool = True,
    show: bool = True,
    figsize: Tuple[int, int] = (9, 8),
):
    """
    Scatter cross-analysis: x_metric vs y_metric, one point per folio.
    Color = GP, alpha = observation count.

    Prototype overlay
    -----------------
    Set overlay_prototypes=True and proto_char="n" (or whichever character).
    config.prototypes must point to the folder containing per-doc subfolders
    with sprite images (as produced by build_and_crop_working_corpus).
    The corpus must have been built with a character_sprite_map.
    Pass the corpus dict as corpus= if you want the sprite filenames looked up
    automatically; otherwise the function looks for proto_char + ".png" directly.
    """
    for col in (x_metric, y_metric):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    gp_colors = config.gp_colors

    df_clean = df.dropna(subset=[x_metric, y_metric])
    if df_clean.empty:
        print(f"⚠️ No data for '{x_metric}' vs '{y_metric}'.")
        return

    # ── auto-detect count column ──────────────────────────────────────────────
    if count_col is None:
        x_suf = x_metric.split("_")[-1]
        y_suf = y_metric.split("_")[-1]
        cx = f"count_{x_suf}" if f"count_{x_suf}" in df_clean.columns else None
        cy = f"count_{y_suf}" if f"count_{y_suf}" in df_clean.columns else None
        if cx and cy and cx != cy:
            df_clean = df_clean.copy()
            df_clean["_combined_count"] = df_clean[cx].fillna(0) + df_clean[cy].fillna(0)
            count_col = "_combined_count"
        elif cx:
            count_col = cx
        elif cy:
            count_col = cy
        else:
            df_clean = df_clean.copy()
            df_clean["_count_ones"] = 1
            count_col = "_count_ones"

    df_clean = df_clean.dropna(subset=[count_col])
    count_min  = df_clean[count_col].min()
    count_max  = df_clean[count_col].max()
    count_norm = mcolors.Normalize(vmin=count_min, vmax=count_max)

    x_range = df_clean[x_metric].max() - df_clean[x_metric].min()
    y_range = df_clean[y_metric].max() - df_clean[y_metric].min()
    x_off   = x_range * 0.008 if x_range > 0 else 0
    y_off   = y_range * 0.008 if y_range > 0 else 0

    fig, ax = plt.subplots(figsize=figsize)

    # ── 1. Scatter + folio labels  (zorder 3 / 4) ────────────────────────────
    for gp, color in gp_colors.items():
        sub = df_clean[df_clean["gp"] == gp]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            alpha = 0.2 + 0.8 * count_norm(r[count_col])
            ax.scatter(r[x_metric], r[y_metric],
                       color=color, alpha=alpha, s=80, linewidths=0.4, zorder=3)
            ax.text(r[x_metric] + x_off, r[y_metric] + y_off,
                    r["folio"] if pd.notna(r.get("folio")) else r.get("folder", ""),
                    fontsize=8, alpha=0.6, color="black", zorder=4)

    # ── 2. Prototype overlay  (drawn last → on top of markers and labels) ────
    if overlay_prototypes and config.prototypes and proto_char:
        try:
            from skimage.io import imread
            from skimage.color import rgb2gray
            from skimage.transform import resize
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            _proto_imports_ok = True
        except ImportError:
            print("⚠️ scikit-image not installed — prototype overlay disabled. "
                  "Run: pip install scikit-image")
            _proto_imports_ok = False

        if _proto_imports_ok:
            image_shape = (48, 48)
            proto_x_off = x_range * 0.03 if x_range > 0 else 0.01
            proto_y_off = y_range * 0.01 if y_range > 0 else 0.01

            sprite = character_sprite_map.get(proto_char) if character_sprite_map else None
            if not sprite:
                print(f"⚠️ No sprite found for '{proto_char}' in character_sprite_map. "
                      f"Pass character_sprite_map= to plot_cross.")
            else:
                for _, row in df_clean.iterrows():
                    doc_folder = row.get("folder", "")
                    img_path = os.path.join(str(config.prototypes), doc_folder, sprite)
                    if not os.path.exists(img_path):
                        continue
                    img = imread(img_path)
                    if img.ndim == 3:
                        img = rgb2gray(img)
                    img = resize(img, image_shape)
                    ab = AnnotationBbox(
                        OffsetImage(img, zoom=0.35, cmap="gray", alpha=0.85),
                        (row[x_metric] - proto_x_off, row[y_metric] - proto_y_off),
                        frameon=False,
                        zorder=5,
                    )
                    ax.add_artist(ab)


    # ── Scatter + folio labels ────────────────────────────────────────────────
    for gp, color in gp_colors.items():
        sub = df_clean[df_clean["gp"] == gp]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            alpha = 0.2 + 0.8 * count_norm(r[count_col])
            ax.scatter(r[x_metric], r[y_metric],
                       color=color, alpha=alpha, s=80, linewidths=0.4, zorder=3)
            ax.text(r[x_metric] + x_off, r[y_metric] + y_off,
                    r["folio"] if pd.notna(r.get("folio")) else r.get("folder", ""),
                    fontsize=8, alpha=0.6, color="black")

    # ── Colorbars ─────────────────────────────────────────────────────────────
    _draw_colorbars(fig, lambda gp: df_clean[df_clean["gp"] == gp],
                    gp_colors, count_min, count_max, config)

    # ── GP legend ─────────────────────────────────────────────────────────────
    #_draw_gp_legend(ax, gp_colors, config)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(x_label_override or get_metric_label(x_metric), fontsize=config.xlabel_fontsize)
    ax.set_ylabel(y_label_override or get_metric_label(y_metric), fontsize=config.ylabel_fontsize)

    plt.tight_layout()

    if save:
        fname = f"cross_{x_metric}_vs_{y_metric}_{config.line_selection_mode}.jpeg"
        plt.savefig(os.path.join(config.fig_output_dir, fname), dpi=300)
        print(f"✅ Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close(fig)
