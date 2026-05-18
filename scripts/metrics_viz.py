"""
metrics_viz.py
======================
Three-layer analysis pipeline for manuscript studies.

USAGE IN NOTEBOOK
-----------------

# -- 0. Config (define once) --------------------------------------------------
from metrics_viz import StudyConfig, FolioIndex
from metrics_viz import compute_letter_metrics, compute_bigram_metrics, compute_word_metrics
from metrics_viz import plot_linear, plot_cross

config = StudyConfig(
    annotation_json        = "path/to/master.json",
    character_measurements = "path/to/character_measurements/",
    fig_output_dir         = "path/to/figures/",
    highlight_ranges = [
        ('1r',   '7v'),
        ('170r', '178r'), ('178r', '185v'),
    ],
    gathering_labels = {
        ('1r',   '7v'):   "Gath. I",
        ('170r', '178r'): "Gath. XXII",
        ('178r', '185v'): "Gath. XXIII",
    },
)

# -- 1. Build corpus (once per session) ---------------------------------------
# corpus, build_stats, metadata = build_and_crop_working_corpus(...)

# -- 2. Build folio index (once per session) ----------------------------------
idx = FolioIndex(config)

# -- 3. Compute metrics -------------------------------------------------------
df = idx.base_df()
df = compute_letter_metrics(df, corpus, "n", config)
df = compute_bigram_metrics(df, "en", config, idx)
df = compute_word_metrics(df, config, idx)

# -- 4. Visualise -------------------------------------------------------------
plot_linear(df, y_metric="mean_ar_n",            config=config, idx=idx)
plot_linear(df, y_metric="mean_b_distance_m_en", config=config, idx=idx)
plot_cross( df, x_metric="mean_ar_n",
               y_metric="mean_b_distance_m_en",  config=config, idx=idx)
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


# =============================================================================
# CONSTANTS
# =============================================================================

# Default GP colours — kept for backward compatibility.
# When group_field != "gp" or new group values appear, colours are auto-assigned
# from _DEFAULT_PALETTE below.
GP_COLORS: Dict[str, str] = {
    "GP1": "#009E73",
    "GP2": "#E69F00",
    "GP3": "#0072B2",
    "GP4": "#CC79A7",
}

# Fallback palette for auto-assigned group colours (colour-blind friendly)
_DEFAULT_PALETTE: List[str] = [
    "#009E73", "#E69F00", "#0072B2", "#CC79A7",
    "#D55E00", "#56B4E9", "#F0E442", "#999999",
    "#000000", "#88CCEE",
]

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

# Metric label registry (column_name -> human label for axis / colorbar)
METRIC_LABELS: Dict[str, str] = {
    # letter
    "mean_ar_{char}":   "Aspect Ratio of '{char}'",
    "std_ar_{char}":    "Std of Aspect Ratio of '{char}'",
    "cv_ar_{char}":     "Coefficient of Variation of Aspect Ratio of '{char}'",
    # bigram
    "mean_b_distance_m_{bigram}":          "'{bigram}' Bigram Distance",
    "mean_b_distance_avg_letter_{bigram}": "'{bigram}' Bigram Distance (norm. avg letter)",
    "mean_b_distance_px_{bigram}":         "'{bigram}' Bigram Distance (pixels)",
    "mean_b_ar_{bigram}":                  "'{bigram}' Bigram Aspect Ratio",
    "cv_b_distance_{bigram}":              "'{bigram}' Bigram Distance Coefficient of Variation",
    # word
    "mean_distance":                "Mean Word Distance (px)",
    "std_distance":                 "Std Word Distance (px)",
    "cv_distance":                  "Coefficient of Variation of Word Distance (px)",
    "mean_w_distance_normalized_m":   "Word Distance",
    "std_distance_normalized_m":    "Std Word Distance",
    "cv_distance_normalized_m":     "Coefficient of Variation of Word Distance",
    "mean_distance_normalized_avg": "Word Distance (norm. avg letter)",
    "std_distance_normalized_avg":  "Std Word Distance (norm. avg letter)",
    "cv_distance_normalized_avg":   "Coefficient of Variation of Word Distance (norm. avg letter)",
}


def get_metric_label(col: str) -> str:
    """Return a human-readable label for a metric column name."""
    if col in METRIC_LABELS:
        return METRIC_LABELS[col]
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
    return col  # fallback


# =============================================================================
# FRIENDLY METRIC NAMES
# =============================================================================

# Aliases: every key on the right is one accepted spelling.
# Add new spellings here without touching the resolver.
_LETTER_METRIC_ALIASES = {
    "AR":          "mean_ar",      "aspect ratio":         "mean_ar",
    "mean AR":     "mean_ar",      "mean aspect ratio":    "mean_ar",
    "AR std":      "std_ar",       "std AR":               "std_ar",
    "std aspect ratio": "std_ar",
    "AR CV":       "cv_ar",        "CV AR":                "cv_ar",
    "coefficient of variation":      "cv_ar",
    "CV of aspect ratio":            "cv_ar",
    "count":       "count",
}

_BIGRAM_METRIC_ALIASES = {
    "distance":              "mean_b_distance_px",      # raw pixels
    "px distance":           "mean_b_distance_px",
    "pixel distance":        "mean_b_distance_px",
    "normalized distance":   "mean_b_distance_m",       # by 1/2 m-width
    "normalised distance":   "mean_b_distance_m",       # British
    "m-normalized distance": "mean_b_distance_m",
    "distance normalized m": "mean_b_distance_m",
    "distance avg":          "mean_b_distance_avg_letter",  # by avg letter
    "distance avg letter":   "mean_b_distance_avg_letter",
    "AR":                    "mean_b_ar",
    "aspect ratio":          "mean_b_ar",
    "distance CV":           "cv_b_distance",
    "CV distance":           "cv_b_distance",
    "count":                 "count",
}

_WORD_METRIC_ALIASES = {
    "distance":              "mean_distance",
    "px distance":           "mean_distance",
    "std distance":          "std_distance",
    "distance CV":           "cv_distance",
    "CV distance":           "cv_distance",
    "normalized distance":   "mean_w_distance_normalized_m",
    "normalised distance":   "mean_w_distance_normalized_m",
    "std normalized":        "std_distance_normalized_m",
    "CV normalized":         "cv_distance_normalized_m",
    "distance avg":          "mean_distance_normalized_avg",
    "std avg":               "std_distance_normalized_avg",
    "CV avg":                "cv_distance_normalized_avg",
    "n":                     "n_word_distances",
    "n word distances":      "n_word_distances",
}


def _resolve_alias(metric: str, table: dict, kind: str) -> str:
    """Look up `metric` in `table` case-insensitively; raise with helpful message."""
    if metric in table:
        return table[metric]
    lower = {k.lower(): v for k, v in table.items()}
    if metric.lower() in lower:
        return lower[metric.lower()]
    available = sorted({k for k in table.keys()})
    raise ValueError(
        f"Unknown {kind} metric: {metric!r}. Available aliases: {available}"
    )


def m(kind: str, *args) -> str:
    """
    Build a metric column name from a friendly description.
    Returns the underlying DataFrame column name as a string, so it can be
    passed directly to plot_linear / plot_cross.

    Three signatures, one per kind
    ------------------------------
    Letter:   m("letter", char,   metric)   e.g. m("letter", "n", "AR")
    Bigram:   m("bigram", bigram, metric)   e.g. m("bigram", "en", "normalized distance")
    Word:     m("word",          metric)    e.g. m("word", "normalized distance")

    Metric vocabulary
    -----------------
    Letter:  "AR" / "aspect ratio" | "AR std" | "AR CV" | "count"
    Bigram:  "distance" (pixels) | "normalized distance" (1/2 m-width) |
             "distance avg" (avg letter width) | "AR" | "distance CV" | "count"
    Word:    "distance" | "std distance" | "distance CV" |
             "normalized distance" | "std normalized" | "CV normalized" |
             "distance avg" | "std avg" | "CV avg" | "n"

    Spellings are case-insensitive and accept both "normalized" / "normalised".
    Pass an unknown metric to see the list of accepted aliases for that kind.

    Examples
    --------
    >>> m("letter", "n", "AR")
    'mean_ar_n'
    >>> m("bigram", "en", "normalized distance")
    'mean_b_distance_m_en'
    >>> m("word", "normalized distance")
    'mean_w_distance_normalized_m'

    >>> plot_linear(df, y_metric=m("letter", "n", "AR"), config=config, idx=idx)
    >>> plot_cross(df,
    ...            x_metric=m("letter", "n", "AR"),
    ...            y_metric=m("letter", "a", "AR"),
    ...            config=config, idx=idx)
    """
    kind_lower = kind.lower()

    if kind_lower == "letter":
        if len(args) != 2:
            raise ValueError("m('letter', char, metric) takes 2 arguments after kind")
        char, metric = args
        prefix = _resolve_alias(metric, _LETTER_METRIC_ALIASES, "letter")
        return f"{prefix}_{char}"

    if kind_lower == "bigram":
        if len(args) != 2:
            raise ValueError("m('bigram', bigram, metric) takes 2 arguments after kind")
        bigram, metric = args
        prefix = _resolve_alias(metric, _BIGRAM_METRIC_ALIASES, "bigram")
        return f"{prefix}_{bigram}"

    if kind_lower == "word":
        if len(args) != 1:
            raise ValueError("m('word', metric) takes 1 argument after kind")
        (metric,) = args
        return _resolve_alias(metric, _WORD_METRIC_ALIASES, "word")

    raise ValueError(f"Unknown kind: {kind!r}. Expected 'letter' | 'bigram' | 'word'.")


# =============================================================================
# STUDY CONFIG
# =============================================================================

@dataclass
class StudyConfig:
    """
    All manuscript-level constants in one place.
    Pass this object to every compute_ and plot_ function.

    Generalisation parameters
    -------------------------
    group_field : str or None
        The annotation.json field used to colour-code documents in plots.
        Default "gp" (Graphic Profile, original analysis).
        Set to another field name ("script", "scribe", "century" ...) or None to
        disable grouping (each doc folder becomes its own group).

    doc_key_parts : int
        How many underscore-separated parts of the line filename form the document key.
        Default 2 matches standard DTLR/BnF naming.

    line_type_filter : str or None
        Which value of the "line" annotation to keep when selecting files for
        bigram and word metric computation.
        Default "DefaultLine" (matches original analysis).
        Set to None to keep all line types (use this when your data has no "line" field).

    line_selection_mode : str or None
        Zone / column filter for bigram and word metrics.
        None           -- no zone filtering (default)
        "MainZone#1"   -- keep only MainZone#1 lines
        "MainZone#2"   -- keep only MainZone#2 lines
        "recto_verso"  -- right column on recto, left column on verso
    """
    annotation_json:        str
    character_measurements: str
    fig_output_dir:         str = "."

    # -- GP / grouping ----------------------------------------------------------
    gp_colors: Dict[str, str] = field(default_factory=lambda: dict(GP_COLORS))

    # -- Generalisation ---------------------------------------------------------
    group_field:      Optional[str] = "gp"          # annotation field for colour grouping
    doc_key_parts:    int           = 2              # parts forming the doc key
    line_type_filter: Optional[str] = "DefaultLine" # line-type filter for metric fns

    # -- Folio annotation -------------------------------------------------------
    highlight_ranges: List[Tuple[str, str]]      = field(default_factory=list)
    gathering_labels: Dict[Tuple[str, str], str] = field(default_factory=dict)
    exclude_doc_ids:  List[str]                  = field(default_factory=list)

    # -- Line selection (zone) --------------------------------------------------
    line_selection_mode: Optional[str] = None  # None | MainZone#1 | MainZone#2 | recto_verso

    # -- Filtering --------------------------------------------------------------
    bbox_filter_mode:      Optional[str]   = "threshold"
    outlier_std_threshold: Optional[float] = None
    width_height_tol:      Tuple           = (None, None)

    # -- Prototype overlay (for plot_cross) ------------------------------------
    prototypes: Optional[str] = None

    # -- Markers ---------------------------------------------------------------
    letter_markers: Dict[str, str] = field(default_factory=lambda: {
        'a': 's', 't': 'x', 'd': '*', 'e': 'P', 'n': 'o',
        'i': '^', 'r': 'v', 'u': 'D', 'm': 'h', 's': 'p',
    })
    bigram_markers: Dict[str, str] = field(default_factory=lambda: {
        'en': 'o', 'et': 's', 'de': 'p', 'er': '^', 'es': 'v',
    })
    _default_markers: List[str] = field(
        default_factory=lambda: ['o','s','^','D','v','P','X','*','h','p'],
        repr=False
    )

    # -- Plot defaults ---------------------------------------------------------
    xlabel_fontsize:      int = 25
    ylabel_fontsize:      int = 25
    colorbar_label_size:  int = 18
    colorbar_tick_size:   int = 15
    legend_fontsize:      int = 15
    annotation_fontsize:  int = 15

    def __post_init__(self):
        os.makedirs(self.fig_output_dir, exist_ok=True)

    def get_marker(self, series_key: str, fallback_idx: int = 0) -> str:
        """Return marker for a letter or bigram series key."""
        if series_key in self.letter_markers:
            return self.letter_markers[series_key]
        if series_key in self.bigram_markers:
            return self.bigram_markers[series_key]
        return self._default_markers[fallback_idx % len(self._default_markers)]


# =============================================================================
# FOLIO INDEX  — built once, reused everywhere
# =============================================================================

class FolioIndex:
    """
    Builds and caches all folio-level mappings from the master JSON.
    Pass to plot_linear() so it knows the global folio order.

    Key attributes
    --------------
    all_folios_sorted : list of page labels in natural sort order
    x_positions       : dict mapping page label -> integer x position
    group_colors      : dict mapping group value -> hex colour string
                        (resolved from config.gp_colors + auto-assigned palette)
    folio_to_doc      : dict mapping page label -> doc folder name
    doc_to_gp         : dict mapping doc folder -> group value
    """

    def __init__(self, config: StudyConfig):
        self.config = config

        # Decide which doc-level fields to load
        fields_needed = ["folio"]
        if config.group_field and config.group_field not in fields_needed:
            fields_needed = [config.group_field] + fields_needed

        self.doc_mappings,  _ = _build_doc_mappings(
            config.annotation_json,
            doc_fields=tuple(fields_needed),
            doc_key_parts=config.doc_key_parts,
        )
        self.line_mappings, _ = _build_line_mappings(config.annotation_json)
        self.doc_line_order   = _build_doc_line_order(
            config.annotation_json,
            doc_key_parts=config.doc_key_parts,
        )

        # folio <-> doc <-> group
        self.folio_to_doc: Dict[str, str] = {}
        self.doc_to_gp:    Dict[str, str] = {}

        for doc, folio in self.doc_mappings["folio"].items():
            if folio:
                self.folio_to_doc[folio] = doc
                # Resolve group value for this doc
                if config.group_field:
                    gp = self.doc_mappings[config.group_field].get(doc, "ungrouped")
                else:
                    gp = doc  # no group field -> use doc folder as group identity
                self.doc_to_gp[doc] = gp

        self.all_folios_sorted: List[str] = sorted(
            self.folio_to_doc.keys(), key=folio_sort_key
        )
        self.folio_status: Dict[str, str] = {
            folio: ("excluded" if self.folio_to_doc[folio] in config.exclude_doc_ids
                    else "included")
            for folio in self.all_folios_sorted
        }
        self.x_positions: Dict[str, int] = {
            folio: i for i, folio in enumerate(self.all_folios_sorted)
        }

        # Build a resolved colour map for all actual group values in the data.
        # config.gp_colors is used when the group value is in it (e.g. GP1-GP4).
        # Any other group values get colours auto-assigned from _DEFAULT_PALETTE.
        all_groups = sorted(
            set(self.doc_to_gp.values()),
            key=lambda x: (x is None, str(x) if x is not None else ""),
        )
        self.group_colors: Dict[str, str] = {}
        auto_idx = 0
        for g in all_groups:
            if g in config.gp_colors:
                self.group_colors[g] = config.gp_colors[g]
            else:
                self.group_colors[g] = _DEFAULT_PALETTE[auto_idx % len(_DEFAULT_PALETTE)]
                auto_idx += 1

    def base_df(self) -> pd.DataFrame:
        """
        Return an empty-metrics DataFrame: one row per folio,
        with columns [folder, folio, gp].
        All compute_*_metrics() functions add columns to this.
        """
        rows = []
        for folio, doc in self.folio_to_doc.items():
            gp = self.doc_to_gp.get(doc)
            # Include all docs; gp may be None or "ungrouped" — that is fine
            rows.append({"folder": doc, "folio": folio, "gp": gp})
        return pd.DataFrame(rows).sort_values("folio", key=lambda s: s.map(folio_sort_key))

    def get_range_gp_color(self, beg_folio: str, end_folio: str) -> Tuple[str, Optional[str]]:
        """Resolve the group colour for a highlight range."""
        colors  = self.group_colors
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


# =============================================================================
# PRIVATE HELPERS  — JSON loading, file selection
# =============================================================================

def folio_sort_key(folio) -> tuple:
    """Natural sort key: '12v' -> (12, 1, 0), '3ra' -> (3, 0, 1)."""
    m = re.match(r'(\d+)([rv])([ab])?', str(folio).lower())
    if m:
        return (int(m.group(1)),
                {'r': 0, 'v': 1}.get(m.group(2), 2),
                {'': 0, 'a': 1, 'b': 2}.get(m.group(3) or '', 0))
    m2 = re.match(r'(\d+)', str(folio))
    return (int(m2.group(1)), 999, 999) if m2 else (999999, 999, 999)


def _build_doc_mappings(annotation_json: str,
                        doc_fields=("gp", "folio"),
                        doc_key_parts: int = 2):
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings  = defaultdict(dict)
    conflicts = defaultdict(lambda: defaultdict(set))
    for key, entry in data.items():
        doc = "_".join(key.replace(".png", "").split("_")[:doc_key_parts])
        for fld in doc_fields:
            val = entry.get(fld)
            if val is None:
                continue
            if doc in mappings[fld] and mappings[fld][doc] != val:
                conflicts[fld][doc].add(mappings[fld][doc])
                conflicts[fld][doc].add(val)
            else:
                mappings[fld][doc] = val
    return mappings, conflicts


def _build_line_mappings(annotation_json: str, line_fields=("line", "zone")):
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings  = defaultdict(dict)
    conflicts = defaultdict(lambda: defaultdict(set))
    for key, entry in data.items():
        lk = key.replace(".png", "")
        for fld in line_fields:
            val = entry.get(fld)
            if val is None:
                continue
            if lk in mappings[fld] and mappings[fld][lk] != val:
                conflicts[fld][lk].add(mappings[fld][lk])
                conflicts[fld][lk].add(val)
            else:
                mappings[fld][lk] = val
    return mappings, conflicts


def _build_doc_line_order(annotation_json: str,
                          doc_key_parts: int = 2) -> Dict[str, List[str]]:
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    order: Dict[str, List[str]] = defaultdict(list)
    for key in data:
        doc = "_".join(key.replace(".png", "").split("_")[:doc_key_parts])
        order[doc].append(key.replace(".png", ""))
    return order


def _select_files(doc_folder, doc_path, doc_line_order, line_mappings, folio,
                  config: StudyConfig) -> List[str]:
    """
    Return ordered list of .json filenames for this doc, filtered by line type
    and zone according to config.line_type_filter and config.line_selection_mode.

    Line-type filtering (config.line_type_filter)
    ---------------------------------------------
    None          -> keep all lines regardless of the "line" annotation
    "DefaultLine" -> keep only lines annotated as "DefaultLine" (original behaviour)
    any string    -> keep only lines matching that exact value

    Zone filtering (config.line_selection_mode)
    -------------------------------------------
    None / "all"    -> no zone filtering
    "MainZone#1"    -> keep only MainZone#1
    "MainZone#2"    -> keep only MainZone#2
    "recto_verso"   -> right col on recto, left col on verso (requires folio labels)
    """
    json_files = {f.replace(".json", ""): f
                  for f in os.listdir(doc_path) if f.endswith(".json")}
    # Preserve annotation order when available
    ordered = [json_files[k] for k in doc_line_order.get(doc_folder, []) if k in json_files]

    # -- 1. Line-type filter --------------------------------------------------
    ltype = config.line_type_filter
    if ltype is None:
        candidates = ordered  # no filtering at all
    else:
        candidates = [f for f in ordered
                      if line_mappings["line"].get(f.replace(".json", "")) == ltype]

    # -- 2. Zone filter -------------------------------------------------------
    mode = config.line_selection_mode
    if mode is None or mode == "all":
        return candidates
    if mode == "MainZone#1":
        return [f for f in candidates
                if line_mappings["zone"].get(f.replace(".json", "")) == "MainZone#1"]
    if mode == "MainZone#2":
        return [f for f in candidates
                if line_mappings["zone"].get(f.replace(".json", "")) == "MainZone#2"]
    if mode == "recto_verso" and folio:
        if folio.endswith(("r", "ra", "rb")):
            zone = "MainZone#2"
        elif folio.endswith(("v", "va", "vb")):
            zone = "MainZone#1"
        else:
            zone = None
        return ([f for f in candidates
                 if line_mappings["zone"].get(f.replace(".json", "")) == zone]
                if zone else candidates)
    return candidates


# =============================================================================
# LAYER 2 — METRICS  (each function adds columns to the wide DataFrame)
# =============================================================================

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
    mean_ar_{char}  -- mean aspect ratio (w/h)
    std_ar_{char}   -- std of aspect ratio
    cv_ar_{char}    -- coefficient of variation of aspect ratio
    count_{char}    -- number of filtered instances

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

    rows: Dict[str, dict] = {}
    char_present_in_corpus = False
    for (doc_folder, c), data in corpus.items():
        if c != char:
            continue
        char_present_in_corpus = True
        ratios = data.get("ratios", [])
        if not ratios:
            continue
        r = np.array(ratios)

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

    if not char_present_in_corpus:
        print(f"Warning: character {char!r} not found in corpus. "
              f"Did you restrict target_characters when building the corpus? "
              f"Columns mean_ar_{char}, std_ar_{char}, cv_ar_{char}, count_{char} "
              f"will NOT be added to the DataFrame.")
        return df

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})

    return df.merge(metric_df, on="folder", how="left")


def compute_bigram_metrics(
    df: pd.DataFrame,
    bigram: str,
    config: StudyConfig,
    idx: Optional[FolioIndex] = None,
    metadata_rows: Optional[list] = None,
) -> pd.DataFrame:
    """
    Add bigram-level spacing metrics for `bigram` to df.
    Re-reads JSON files (sequence information required).

    Parameters
    ----------
    metadata_rows : list of dicts | None
        The metadata_rows returned by build_and_crop_working_corpus.
        When provided, a bigram instance is only kept when BOTH bboxes
        survived the nomatch + std filter (kept=True).
        When None, falls back to the original global std filter behaviour.

    New columns added
    -----------------
    mean_b_distance_m_{bigram}          -- distance norm. by 1/2 m width
    mean_b_distance_avg_letter_{bigram} -- distance norm. by avg letter width
    mean_b_distance_px_{bigram}         -- raw pixel distance
    mean_b_ar_{bigram}                  -- bigram aspect ratio
    cv_b_distance_{bigram}              -- CV of distances
    count_{bigram}                      -- number of filtered instances
    """
    if idx is None:
        idx = FolioIndex(config)

    l1, l2 = bigram[0], bigram[1]

    # Build kept lookup: (json_file, cx_rounded) -> bool
    kept_lookup: Dict[tuple, bool] = {}
    if metadata_rows is not None:
        for row in metadata_rows:
            key = (row["json_file"], round(row["bbox_cx"], 1))
            kept_lookup[key] = bool(row.get("kept", False))

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
                              idx.line_mappings, folio, config)

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
                # For normalisation widths, only use kept bboxes when lookup available
                if kept_lookup:
                    if not kept_lookup.get((fname, round(p["bbox"]["cx"], 1)), False):
                        continue
                w = p["bbox"].get("w", 0)
                if w <= 0:
                    continue
                if char == "m":
                    folio_m_widths[doc_folder].append(w)
                if char in VALID_AVG_WIDTH_CHARS:
                    folio_lw[doc_folder].append(w)

            for i in range(len(preds) - 1):
                p1, p2 = preds[i], preds[i + 1]
                if (p1.get("character", "") == l1 and
                        p2.get("character", "") == l2 and
                        p1.get("error_label") == "match" and
                        p2.get("error_label") == "match"):
                    lo  = (i == 0)              or (preds[i - 1].get("error_label") == "match")
                    ro  = (i + 2 >= len(preds)) or (preds[i + 2].get("error_label") == "match")
                    if not (lo and ro):
                        continue
                    b1, b2 = p1["bbox"], p2["bbox"]
                    if not (all(k in b1 for k in ("cx","w","h")) and
                            all(k in b2 for k in ("cx","w","h"))):
                        continue
                    # Require both bboxes to be kept when lookup is available
                    if kept_lookup:
                        k1 = (fname, round(b1["cx"], 1))
                        k2 = (fname, round(b2["cx"], 1))
                        if not (kept_lookup.get(k1, False) and kept_lookup.get(k2, False)):
                            continue
                    dist = (b2["cx"] - b2["w"]/2) - (b1["cx"] + b1["w"]/2)
                    cw   = (b2["cx"] + b2["w"]/2) - (b1["cx"] - b1["w"]/2)
                    ch   = max(b1["h"], b2["h"])
                    first_pass[doc_folder].append({
                        "distance": dist, "combined_w": cw, "combined_h": ch,
                        "w1": b1["w"], "h1": b1["h"], "w2": b2["w"], "h2": b2["h"],
                    })

    # When metadata_rows supplied, bboxes already filtered — skip global std pass.
    # When None, apply original global std filter for backward compatibility.
    if kept_lookup:
        bounds: dict = {}
    else:
        bounds = {}
        if config.bbox_filter_mode == "threshold" and config.outlier_std_threshold:
            all_vals = {k: [e[k] for ex in first_pass.values() for e in ex]
                        for k in ("w1","h1","w2","h2")}
            for k, arr in all_vals.items():
                if arr:
                    mv, s = np.mean(arr), np.std(arr)
                    t     = config.outlier_std_threshold
                    bounds[k] = (mv - t*s, mv + t*s)

    sfx  = bigram
    rows: Dict[str, dict] = {}

    for doc, examples in first_pass.items():
        keep = [e for e in examples
                if all(bounds[k][0] <= e[k] <= bounds[k][1] for k in bounds)]
        if not keep:
            continue

        dists   = [e["distance"] for e in keep]
        mean_px = float(np.mean(dists))
        std_px  = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
        cv      = std_px / mean_px if mean_px != 0 else np.nan
        mean_ar = float(np.mean([e["combined_w"] / e["combined_h"] for e in keep]))

        mean_norm_m = mean_norm_avg = None

        if folio_m_widths.get(doc):
            m_half = np.mean(folio_m_widths[doc]) / 2
            if m_half > 0:
                mean_norm_m = mean_px / m_half

        if folio_lw.get(doc):
            avg_lw = np.mean(folio_lw[doc])
            if avg_lw > 0:
                mean_norm_avg = mean_px / avg_lw

        rows[doc] = {
            f"mean_b_distance_m_{sfx}":           mean_norm_m,
            f"mean_b_distance_avg_letter_{sfx}":  mean_norm_avg,
            f"mean_b_distance_px_{sfx}":          mean_px,
            f"mean_b_ar_{sfx}":                   mean_ar,
            f"cv_b_distance_{sfx}":               cv,
            f"count_{sfx}":                       len(keep),
        }

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})

    return df.merge(metric_df, on="folder", how="left")
    if idx is None:
        idx = FolioIndex(config)

    # Case-sensitive: "de" matches only lowercase d+e, "De" only D+e, etc.
    # This keeps compute_bigram_metrics consistent with compute_letter_metrics
    # and the m-width normalisation in compute_word_metrics.
    l1, l2 = bigram[0], bigram[1]

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
                              idx.line_mappings, folio, config)

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
                if char == "m":
                    folio_m_widths[doc_folder].append(w)
                if char in VALID_AVG_WIDTH_CHARS:
                    folio_lw[doc_folder].append(w)

            for i in range(len(preds) - 1):
                p1, p2 = preds[i], preds[i + 1]
                if (p1.get("character", "") == l1 and
                        p2.get("character", "") == l2 and
                        p1.get("error_label") == "match" and
                        p2.get("error_label") == "match"):
                    lo  = (i == 0)              or (preds[i - 1].get("error_label") == "match")
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
                            "w1": b1["w"], "h1": b1["h"], "w2": b2["w"], "h2": b2["h"],
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

    sfx  = bigram
    rows: Dict[str, dict] = {}

    for doc, examples in first_pass.items():
        keep = [e for e in examples
                if all(bounds[k][0] <= e[k] <= bounds[k][1] for k in bounds)]
        if not keep:
            continue

        dists   = [e["distance"] for e in keep]
        mean_px = float(np.mean(dists))
        std_px  = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
        cv      = std_px / mean_px if mean_px != 0 else np.nan
        mean_ar = float(np.mean([e["combined_w"] / e["combined_h"] for e in keep]))

        mean_norm_m = mean_norm_avg = None

        if folio_m_widths.get(doc):
            m_half = np.mean(folio_m_widths[doc]) / 2
            if m_half > 0:
                mean_norm_m = mean_px / m_half

        if folio_lw.get(doc):
            avg_lw = np.mean(folio_lw[doc])
            if avg_lw > 0:
                mean_norm_avg = mean_px / avg_lw

        rows[doc] = {
            f"mean_b_distance_m_{sfx}":           mean_norm_m,
            f"mean_b_distance_avg_letter_{sfx}":  mean_norm_avg,
            f"mean_b_distance_px_{sfx}":          mean_px,
            f"mean_b_ar_{sfx}":                   mean_ar,
            f"cv_b_distance_{sfx}":               cv,
            f"count_{sfx}":                       len(keep),
        }

    metric_df = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "folder"})

    return df.merge(metric_df, on="folder", how="left")


def compute_word_metrics(
    df: pd.DataFrame,
    config: StudyConfig,
    idx: Optional[FolioIndex] = None,
    metadata_rows: Optional[list] = None,
) -> pd.DataFrame:
    """
    Add word-spacing metrics per folio to df.
    Re-reads JSON files.

    Parameters
    ----------
    metadata_rows : list of dicts | None
        The metadata_rows returned by build_and_crop_working_corpus.
        When provided, word gaps are only counted when BOTH boundary bboxes
        (last char of word 1, first char of word 2) survived the nomatch +
        std filter (kept=True).
        When None, only the nomatch filter is applied (original behaviour).

    New columns added (all three normalisations x mean/std/cv)
    ----------------------------------------------------------
    mean_distance, std_distance, cv_distance
    mean_w_distance_normalized_m, std_distance_normalized_m, cv_distance_normalized_m
    mean_distance_normalized_avg, std_distance_normalized_avg, cv_distance_normalized_avg
    n_word_distances
    """
    if idx is None:
        idx = FolioIndex(config)

    # Build kept lookup: (json_file, cx_rounded) -> bool
    kept_lookup: Dict[tuple, bool] = {}
    if metadata_rows is not None:
        for row in metadata_rows:
            key = (row["json_file"], round(row["bbox_cx"], 1))
            kept_lookup[key] = bool(row.get("kept", False))

    def _word_distances(predictions, jf):
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
                last_char  = w1[-1]
                first_char = w2[0]
                # Require both boundary bboxes to be kept when lookup available
                if kept_lookup:
                    lk = (jf, round(last_char["bbox"]["cx"],  1))
                    fk = (jf, round(first_char["bbox"]["cx"], 1))
                    if not (kept_lookup.get(lk, False) and kept_lookup.get(fk, False)):
                        continue
                last, first = last_char["bbox"], first_char["bbox"]
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
                              idx.line_mappings, folio, config)

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

            all_distances.extend(_word_distances(valid_preds, jf))

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
            "mean_w_distance_normalized_m":  mean_norm_m,
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


# =============================================================================
# LAYER 3 — SHARED VIZ HELPERS
# =============================================================================

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


def _draw_colorbars(fig, df_sub_fn, group_colors, count_min, count_max, config,
                    bar_width=0.02, left=1.02):
    """
    Draw stacked alpha colorbars — one per group value.
    group_colors is the resolved dict from FolioIndex.group_colors or equivalent.
    The list is reversed so the first group appears at the top.
    """
    groups_cb  = list(reversed(list(group_colors.keys())))
    bar_height = 0.20
    spacing    = 0.02
    total_h    = len(groups_cb) * bar_height + (len(groups_cb) - 1) * spacing
    for i, gp in enumerate(groups_cb):
        if df_sub_fn(gp).empty:
            continue
        bottom = (0.5 - total_h / 2) + i * (bar_height + spacing)
        _add_alpha_colorbar(
            fig, group_colors[gp], count_min, count_max,
            f"{gp} Occ.",
            [left, bottom, bar_width, bar_height],
            label_fontsize=config.colorbar_label_size,
            tick_fontsize=config.colorbar_tick_size,
        )


def _draw_highlight_ranges(ax, config: StudyConfig, idx: FolioIndex):
    """Draw shaded gathering ranges + dashed edges + gathering labels."""
    for beg, end in config.highlight_ranges:
        x_beg = idx.x_positions.get(beg)
        x_end = idx.x_positions.get(end)
        color, _ = idx.get_range_gp_color(beg, end)
        label = config.gathering_labels.get((beg, end), f"{beg}-{end}")

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


def _resolve_group_colors(df_clean: pd.DataFrame,
                          config: StudyConfig,
                          idx: Optional[FolioIndex]) -> Dict[str, str]:
    """
    Return a group -> colour dict for the groups actually present in df_clean.
    Prefers idx.group_colors when idx is provided; falls back to auto-assignment.
    """
    if idx is not None:
        return idx.group_colors

    actual_groups = sorted(
        df_clean["gp"].dropna().unique(),
        key=lambda x: (x is None, str(x) if x is not None else ""),
    )
    colors: Dict[str, str] = {}
    auto_idx = 0
    for g in actual_groups:
        if g in config.gp_colors:
            colors[g] = config.gp_colors[g]
        else:
            colors[g] = _DEFAULT_PALETTE[auto_idx % len(_DEFAULT_PALETTE)]
            auto_idx += 1
    return colors


# =============================================================================
# LAYER 3 — plot_linear
# =============================================================================

def plot_linear(
    df: pd.DataFrame,
    y_metric,                               # str OR list[str]
    config: StudyConfig,
    idx: FolioIndex,
    count_col: Optional[str] = None,
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

    Multiple series on shared axes
    --------------------------------
    plot_linear(df, y_metric=["mean_ar_n", "mean_ar_a", "mean_ar_i"], ...)
    plot_linear(df, y_metric=["mean_b_distance_m_en", "mean_b_distance_m_et"], ...)

    Each series gets its own marker (from config.letter_markers / config.bigram_markers).
    Group colour and count-based alpha still apply per point.
    """
    if isinstance(y_metric, str):
        y_metrics = [y_metric]
    else:
        y_metrics = list(y_metric)

    for ym in y_metrics:
        if ym not in df.columns:
            raise ValueError(f"Column '{ym}' not found. Available: {list(df.columns)}")

    # Use the resolved group colours from the index (works for any grouping)
    gp_colors        = idx.group_colors
    highlight_folios = highlight_folios or []
    multi            = len(y_metrics) > 1

    def _series_key(col: str) -> str:
        for prefix in ("mean_ar_", "std_ar_", "cv_ar_"):
            if col.startswith(prefix):
                return col[len(prefix):]
        for prefix in ("mean_b_distance_m_", "mean_b_distance_avg_letter_",
                       "mean_b_distance_px_", "mean_b_ar_", "cv_b_distance_"):
            if col.startswith(prefix):
                return col[len(prefix):]
        return col

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

    all_valid = df.dropna(subset=y_metrics[:1] + [count_col])
    if all_valid.empty:
        print(f"No data for metric(s) {y_metrics}.")
        return

    count_min  = df[count_col].dropna().min()
    count_max  = df[count_col].dropna().max()
    count_norm = mcolors.Normalize(vmin=count_min, vmax=count_max)

    folio_to_gp    = dict(zip(df["folio"], df["gp"]))
    folio_to_count = dict(zip(df["folio"], df[count_col]))

    fig, ax = plt.subplots(figsize=figsize)

    _draw_highlight_ranges(ax, config, idx)
    _draw_folio_xaxis(ax, idx, highlight_folios)

    series_legend_elements = []

    for s_idx, ym in enumerate(y_metrics):
        sk     = _series_key(ym)
        marker = config.get_marker(sk, fallback_idx=s_idx)
        df_s   = df.dropna(subset=[ym, count_col])

        folio_to_y    = dict(zip(df_s["folio"], df_s[ym]))
        folios_w_data = [f for f in idx.all_folios_sorted if f in folio_to_y]

        # Connecting lines
        for i in range(len(folios_w_data) - 1):
            f1, f2 = folios_w_data[i], folios_w_data[i + 1]
            gap = idx.all_folios_sorted.index(f2) - idx.all_folios_sorted.index(f1)
            ls  = '--' if gap > 1 else '-'
            ax.plot([idx.x_positions[f1], idx.x_positions[f2]],
                    [folio_to_y[f1], folio_to_y[f2]],
                    ls, color='grey', alpha=0.25, zorder=2)

        # Scatter points
        for gp, color in gp_colors.items():
            for folio in folios_w_data:
                if folio_to_gp.get(folio) != gp:
                    continue
                cnt   = folio_to_count.get(folio, count_min)
                alpha = 0.2 + 0.8 * count_norm(cnt if pd.notna(cnt) else count_min)
                ax.scatter(idx.x_positions[folio], folio_to_y[folio],
                           color=color, alpha=alpha, s=80, marker=marker,
                           zorder=3, linewidths=0.5)

        if multi:
            label = _series_key(ym)
            series_legend_elements.append(
                Line2D([0], [0], marker=marker, linestyle='',
                       markersize=9, markeredgecolor='0.35',
                       markerfacecolor='dimgray', markeredgewidth=1.2,
                       label=label)
            )

    df_for_cb = df.dropna(subset=[y_metrics[0], count_col])
    _draw_colorbars(fig, lambda gp: df_for_cb[df_for_cb["gp"] == gp],
                    gp_colors, count_min, count_max, config, bar_width=0.015, left=1.01)

    if multi and series_legend_elements:
        ax.legend(
            handles=series_legend_elements,
            loc='lower center', ncol=len(series_legend_elements),
            frameon=True, framealpha=0.9,
            fontsize=config.legend_fontsize,
            handletextpad=0.3, columnspacing=0.6,
        )

    if y_label_override:
        y_label = y_label_override
    elif multi:
        y_label = "Metric value"
    else:
        y_label = get_metric_label(y_metrics[0])

    ax.set_xlabel("Ordered Pages", fontsize=config.xlabel_fontsize)
    ax.set_ylabel(y_label,         fontsize=config.ylabel_fontsize)

    plt.tight_layout()

    if save:
        safe = "_".join(y_metrics) if multi else y_metrics[0]
        if len(safe) > 80:
            safe = safe[:80]
        fname = f"linear_{safe}_{config.line_selection_mode}.jpeg"
        plt.savefig(os.path.join(config.fig_output_dir, fname), dpi=300)
        print(f"Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# LAYER 3 — plot_cross
# =============================================================================

def plot_cross(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    config: StudyConfig,
    idx: Optional[FolioIndex] = None,       # pass for correct colour resolution
    count_col: Optional[str] = None,
    x_label_override: Optional[str] = None,
    y_label_override: Optional[str] = None,
    overlay_prototypes: bool = False,
    proto_char: Optional[str] = None,
    character_sprite_map: Optional[Dict[str, str]] = None,
    save: bool = True,
    show: bool = True,
    figsize: Tuple[int, int] = (9, 8),
):
    """
    Scatter cross-analysis: x_metric vs y_metric, one point per folio.
    Color = group, alpha = observation count.

    idx is optional but recommended: passing it ensures group colours are resolved
    from the actual data (required when group_field != "gp" or new group values appear).
    When idx is not passed, colours are resolved from config.gp_colors with auto-fallback.

    Prototype overlay
    -----------------
    Set overlay_prototypes=True and proto_char="n".
    config.prototypes must point to the folder containing per-doc subfolders with sprite images.
    Pass character_sprite_map= from build_character_index_map().
    """
    for col in (x_metric, y_metric):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    df_clean = df.dropna(subset=[x_metric, y_metric])
    if df_clean.empty:
        print(f"No data for '{x_metric}' vs '{y_metric}'.")
        return

    # Resolve group colours from idx when available, otherwise auto-assign
    gp_colors = _resolve_group_colors(df_clean, config, idx)

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

    df_clean   = df_clean.dropna(subset=[count_col])
    count_min  = df_clean[count_col].min()
    count_max  = df_clean[count_col].max()
    count_norm = mcolors.Normalize(vmin=count_min, vmax=count_max)

    x_range = df_clean[x_metric].max() - df_clean[x_metric].min()
    y_range = df_clean[y_metric].max() - df_clean[y_metric].min()
    x_off   = x_range * 0.008 if x_range > 0 else 0
    y_off   = y_range * 0.008 if y_range > 0 else 0

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter + folio labels
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

    # Prototype overlay
    if overlay_prototypes and config.prototypes and proto_char:
        try:
            from skimage.io import imread
            from skimage.color import rgb2gray
            from skimage.transform import resize
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            _proto_imports_ok = True
        except ImportError:
            print("scikit-image not installed -- prototype overlay disabled. "
                  "Run: pip install scikit-image")
            _proto_imports_ok = False

        if _proto_imports_ok:
            image_shape = (48, 48)
            proto_x_off = x_range * 0.03 if x_range > 0 else 0.01
            proto_y_off = y_range * 0.01 if y_range > 0 else 0.01

            sprite = character_sprite_map.get(proto_char) if character_sprite_map else None
            if not sprite:
                print(f"No sprite found for '{proto_char}' in character_sprite_map. "
                      f"Pass character_sprite_map= to plot_cross.")
            else:
                for _, row in df_clean.iterrows():
                    doc_folder = row.get("folder", "")
                    img_path   = os.path.join(str(config.prototypes), doc_folder, sprite)
                    if not os.path.exists(img_path):
                        continue
                    img = imread(img_path)
                    if img.ndim == 3:
                        img = rgb2gray(img)
                    img = resize(img, image_shape)
                    ab  = AnnotationBbox(
                        OffsetImage(img, zoom=0.35, cmap="gray", alpha=0.85),
                        (row[x_metric] - proto_x_off, row[y_metric] - proto_y_off),
                        frameon=False,
                        zorder=5,
                    )
                    ax.add_artist(ab)

    _draw_colorbars(fig, lambda gp: df_clean[df_clean["gp"] == gp],
                    gp_colors, count_min, count_max, config)

    ax.set_xlabel(x_label_override or get_metric_label(x_metric), fontsize=config.xlabel_fontsize)
    ax.set_ylabel(y_label_override or get_metric_label(y_metric), fontsize=config.ylabel_fontsize)

    plt.tight_layout()

    if save:
        fname = f"cross_{x_metric}_vs_{y_metric}_{config.line_selection_mode}.jpeg"
        plt.savefig(os.path.join(config.fig_output_dir, fname), dpi=300)
        print(f"Saved: {fname}")
    if show:
        plt.show()
    else:
        plt.close(fig)
