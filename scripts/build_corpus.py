import os
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# metadata mappings

def build_doc_mappings(annotation_json, doc_fields=["gp", "folio"]):
    """
    Extract document-level mappings from master JSON.
    Each doc_folder is the prefix of the line keys.
    
    Returns
    -------
    doc_mappings : dict
        doc_mappings[field][doc_folder] -> value
    doc_conflicts : dict
        doc_conflicts[field][doc_folder] -> set of conflicting values
    """
    with open(annotation_json, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    doc_mappings = defaultdict(dict)
    doc_conflicts = defaultdict(lambda: defaultdict(set))

    for key, entry in master_data.items():
        basename = key.replace(".png", "")
        doc_folder = "_".join(basename.split("_")[:2])

        for field in doc_fields:
            value = entry.get(field)
            if value is None:
                continue

            if doc_folder in doc_mappings[field]:
                if doc_mappings[field][doc_folder] != value:
                    doc_conflicts[field][doc_folder].add(doc_mappings[field][doc_folder])
                    doc_conflicts[field][doc_folder].add(value)
            else:
                doc_mappings[field][doc_folder] = value

    # Report conflicts
    for field, field_conflicts in doc_conflicts.items():
        if field_conflicts:
            print(f"⚠️ {field.upper()} conflicts detected per document:")
            for doc_folder, values in field_conflicts.items():
                print(f"  {doc_folder}: {sorted(values)}")

    return doc_mappings, doc_conflicts


def build_line_mappings(annotation_json, line_fields=None):
    """
    Extract line-level mappings from master JSON.
    Each line key is the full basename (e.g., btv1b84472995_f940_eSc_line_5f967ab4)
    
    Parameters
    ----------
    annotation_json : str
        Path to the master annotation JSON.
    line_fields : iterable[str] or None
        Which fields to extract per line.
        Defaults to ["line", "zone"].
    
    Returns
    -------
    line_mappings : dict
        line_mappings[field][line_key] -> value
    conflicts : dict
        conflicts[field][line_key] -> set(values)
    """
    if line_fields is None:
        line_fields = ["line", "zone"]  # default fields to extract

    with open(annotation_json, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    line_mappings = defaultdict(dict)
    conflicts = defaultdict(lambda: defaultdict(set))

    for key, entry in master_data.items():
        line_key = key.replace(".png", "")
        for field in line_fields:
            value = entry.get(field)
            if value is None:
                continue

            if line_key in line_mappings[field]:
                if line_mappings[field][line_key] != value:
                    conflicts[field][line_key].add(line_mappings[field][line_key])
                    conflicts[field][line_key].add(value)
            else:
                line_mappings[field][line_key] = value

    # Print conflicts if any
    for field, field_conflicts in conflicts.items():
        if field_conflicts:
            print(f"⚠️ {field.upper()} conflicts detected:")
            for line_key, values in field_conflicts.items():
                print(f"  {line_key}: {sorted(values)}")

    return line_mappings, conflicts

def build_character_index_map(transcribe_json_path):
    """
    Build mapping: character → sprite filename
    using transcribe.json where keys are sprite indices.
    """

    with open(transcribe_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    character_sprite_map = {}

    for idx, entry in data.items():
        char = entry.get("character")

        if char is None:
            continue

        # If sprite filenames are 1.png, 2.png, ...
        sprite_filename = f"{idx}.png"

        character_sprite_map[char] = sprite_filename

    return character_sprite_map

# Function to build working corpus from dataset (column/line filter + bbox filtering). Optionally, crop valid and discarded bboxes

def build_and_crop_working_corpus(
    character_measurements,
    text_line_images,
    cropped_character_bboxes,
    annotation_json,
    target_characters=None,
    character_sprite_map=None,
    zone_mode="all",
    line_mode="DefaultLine",
    bbox_filter_mode="threshold",
    outlier_std_threshold=4.0,
    exclude_doc_ids=None,
    crop_images=True
):
    """
    Build a working corpus AND optionally crop characters from line images.
    If crop_images=False, no folders or images are saved, only the corpus and metadata in memory.
    
    Returns:
      corpus : dict
      build_stats : dict
      metadata_rows : list of dicts
    """

    def safe_crop(image, bbox):
        cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
        img_w, img_h = image.size
        x1 = max(int(cx - w / 2), 0)
        y1 = max(int(cy - h / 2), 0)
        x2 = min(int(cx + w / 2), img_w)
        y2 = min(int(cy + h / 2), img_h)
        if x2 <= x1 or y2 <= y1:
            return None
        return image.crop((x1, y1, x2, y2))

    # Only create output folders if cropping is enabled
    if crop_images:
        kept_dir = os.path.join(cropped_character_bboxes, "kept_bboxes")
        std_dir = os.path.join(cropped_character_bboxes, "discarded_std")
        nomatch_dir = os.path.join(cropped_character_bboxes, "discarded_nomatch")
        os.makedirs(kept_dir, exist_ok=True)
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(nomatch_dir, exist_ok=True)

    # Load mappings
    doc_mappings, _ = build_doc_mappings(annotation_json)
    line_mappings, _ = build_line_mappings(annotation_json, line_fields=["line", "zone"])

    if target_characters is not None and not isinstance(target_characters, (list, set)):
        target_characters = [target_characters]

    corpus = {}
    metadata_rows = []

    build_stats = {
        "gp_total_predictions": defaultdict(int),
        "gp_match_predictions": defaultdict(int),
        "gp_neighbor_valid": defaultdict(int),
        "gp_after_std_filter": defaultdict(int),
        "gp_default_lines": defaultdict(int),
        "gp_nomatch": defaultdict(int)
    }

    # ======== Pass 1: Collect bboxes & build corpus ========
    folders = [f for f in os.listdir(character_measurements) if os.path.isdir(os.path.join(character_measurements, f))]
    for doc_folder in tqdm(folders, desc="Building working corpus"):
        doc_path = os.path.join(character_measurements, doc_folder)
        if exclude_doc_ids and doc_folder in exclude_doc_ids:
            continue

        gp = doc_mappings["gp"].get(doc_folder)
        folio = doc_mappings["folio"].get(doc_folder)
        json_files = [f for f in os.listdir(doc_path) if f.endswith(".json")]

        for jf in json_files:
            line_key = jf.replace(".json", "")
            # Line filter
            if line_mode == "DefaultLine" and line_mappings["line"].get(line_key) != "DefaultLine":
                continue
            # Zone filter
            zone = line_mappings["zone"].get(line_key)
            if zone_mode == "MainZone#1" and zone != "MainZone#1":
                continue
            elif zone_mode == "MainZone#2" and zone != "MainZone#2":
                continue
            elif zone_mode == "recto_verso" and folio is not None:
                if folio.endswith(("r", "ra", "rb")) and zone != "MainZone#2":
                    continue
                if folio.endswith(("v", "va", "vb")) and zone != "MainZone#1":
                    continue

            build_stats["gp_default_lines"][gp] += 1

            with open(os.path.join(doc_path, jf)) as f:
                data_json = json.load(f)

            preds = data_json.get("predictions", [])
            for i, p in enumerate(preds):
                build_stats["gp_total_predictions"][gp] += 1
                char = p.get("character")
                if char is None:
                    continue
                if target_characters and char not in target_characters:
                    continue

                # Determine if match
                nomatch_flag = False
                if p.get("error_label") != "match":
                    nomatch_flag = True
                    build_stats["gp_nomatch"][gp] += 1
                else:
                    neighbors = []
                    if i > 0:
                        neighbors.append(preds[i - 1].get("error_label"))
                    if i < len(preds) - 1:
                        neighbors.append(preds[i + 1].get("error_label"))
                    if any(n != "match" for n in neighbors):
                        nomatch_flag = True
                        build_stats["gp_nomatch"][gp] += 1
                    else:
                        build_stats["gp_match_predictions"][gp] += 1

                w = p["bbox"]["w"]
                h = p["bbox"]["h"]

                key = (doc_folder, char)
                if key not in corpus:
                    corpus[key] = {
                        "folio": folio,
                        "gp": gp,
                        "ratios": [],
                        "widths": [],
                        "heights": [],
                        "char_count": 0,
                        "character": char,
                        "sprite_filename": character_sprite_map.get(char) if character_sprite_map else None,
                        "records": []
                    }

                if not nomatch_flag:
                    corpus[key]["widths"].append(w)
                    corpus[key]["heights"].append(h)

                corpus[key]["records"].append({
                    "line_key": line_key,
                    "json_file": jf,
                    "bbox": p["bbox"],
                    "gp": gp,
                    "folio": folio,
                    "kept": not nomatch_flag,
                    "nomatch": nomatch_flag
                })

    # ======== Pass 2: Std / percentage filter & optional cropping ========
    if crop_images:
        print("\nCropping images...")
    for key, data in tqdm(corpus.items(), desc="Cropping corpus" if crop_images else "Processing corpus"):
        widths = np.array(data["widths"])
        heights = np.array(data["heights"])
        keep_mask = np.ones_like(widths, dtype=bool) if len(widths) > 0 else []

        if len(widths) > 0:
            if bbox_filter_mode == "threshold":
                mw, sw = np.mean(widths), np.std(widths)
                mh, sh = np.mean(heights), np.std(heights)
                w_bounds = (mw - outlier_std_threshold * sw, mw + outlier_std_threshold * sw)
                h_bounds = (mh - outlier_std_threshold * sh, mh + outlier_std_threshold * sh)
                keep_mask &= (widths >= w_bounds[0]) & (widths <= w_bounds[1])
                keep_mask &= (heights >= h_bounds[0]) & (heights <= h_bounds[1])

        data["widths"] = widths.tolist()
        data["heights"] = heights.tolist()
        data["ratios"] = (widths / heights).tolist()
        data["char_count"] = len(widths)
        build_stats["gp_after_std_filter"][data["gp"]] += len(widths)

        rec_idx = 0
        for rec in data["records"]:
            line_key = rec["line_key"]
            bbox = rec["bbox"]
            doc_folder = key[0]
            char = key[1]

            crop = None
            out_path = None
            if crop_images:
                img_path = os.path.join(text_line_images, doc_folder, f"{line_key}.png")
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                    crop = safe_crop(image, bbox)

                # Determine folder
                if rec.get("nomatch", False):
                    save_base = nomatch_dir
                else:
                    keep = keep_mask[rec_idx] if len(keep_mask) > 0 and rec.get("kept", True) else True
                    save_base = kept_dir if keep else std_dir
                    rec_idx += 1

                save_path = os.path.join(save_base, doc_folder, char)
                os.makedirs(save_path, exist_ok=True)
                out_name = f"{line_key}_{rec_idx:04d}_{char}.png"
                out_path = os.path.join(save_path, out_name)
                if crop is not None:
                    crop.save(out_path)

            metadata_rows.append({
                "doc_folder": doc_folder,
                "character": char,
                "gp": rec["gp"],
                "folio": rec["folio"],
                "json_file": rec["json_file"],
                "bbox_cx": bbox["cx"],
                "bbox_cy": bbox["cy"],
                "width": bbox["w"],
                "height": bbox["h"],
                "kept": rec.get("kept", True) and not rec.get("nomatch", False),
                "nomatch": rec.get("nomatch", False),
                "crop_path": out_path
            })

    # Save metadata CSV only if cropping is enabled
    if crop_images and metadata_rows:
        df = pd.DataFrame(metadata_rows)
        csv_path = os.path.join(cropped_character_bboxes, "crop_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved crop metadata CSV to {csv_path}")

    return corpus, build_stats, metadata_rows


# Compute statistics of corpus

def compute_working_corpus_statistics(corpus, build_stats):

    gp_docs = defaultdict(set)
    gp_folios = defaultdict(set)
    gp_characters = defaultdict(int)

    char_counts = defaultdict(int)
    char_ratios = defaultdict(list)

    for (doc_id, char), data in corpus.items():

        gp = data["gp"]
        folio = data["folio"]

        gp_docs[gp].add(doc_id)
        gp_folios[gp].add(folio)

        count = data["char_count"]
        ratios = data["ratios"]

        gp_characters[gp] += count

        char_counts[char] += count
        char_ratios[char].extend(ratios)

    rows = []

    for gp in sorted(gp_docs):

        total = build_stats["gp_total_predictions"][gp]
        match = build_stats["gp_match_predictions"][gp]
        neighbor = build_stats["gp_neighbor_valid"][gp]
        after_std = build_stats["gp_after_std_filter"][gp]

        rows.append({
            "GP": gp,
            "Docs": len(gp_docs[gp]),
            "Folios": len(gp_folios[gp]),
            "DefaultLines": build_stats["gp_default_lines"][gp],

            "Total predictions": total,
            "Match predictions": match,

            "% discarded (no match)": round(100 * (total - match) / total, 2) if total else 0,
            #"% discarded (neighbor)": round(100 * (match - neighbor) / match, 2) if match else 0,
            #"% discarded (std filter)": round(100 * (neighbor - after_std) / neighbor, 2) if neighbor else 0,

            "Final characters": gp_characters[gp]
        })

    df_gp = pd.DataFrame(rows).sort_values("GP")

    rows = []

    for char in sorted(char_counts):

        ratios = np.array(char_ratios[char])

        rows.append({
            "Character": char,
            "Count": char_counts[char],
            #"Mean ratio": ratios.mean(),
            #"Std ratio": ratios.std(),
            #"Median ratio": np.median(ratios)
        })

    df_char = pd.DataFrame(rows)

    print("\nWORKING CORPUS — GP SUMMARY\n")
    display(df_gp)

    print("\nWORKING CORPUS — CHARACTER SUMMARY\n")
    display(df_char)

    return df_gp, df_char
