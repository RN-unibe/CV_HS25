#!/usr/bin/env python3
"""
select_best_templates_fast.py

Greedy search for the best Sudoku digit templates (max 2 per digit) with a
MUCH faster inner loop than repeatedly calling evaluate_all():

- Cache sudoku_cells once per image.
- Precompute is_empty mask per cell.
- Build candidate pools from ALL_CELL_COORDINATES.
- Precompute correlation scores between EVERY candidate template and EVERY
  sudoku cell using the SAME call as in recognition.get_digit_correlations
  (match_template(image=cell, template=tmpl, pad_input=True, constant_values=0)).
- Do greedy selection entirely in memory and only write the final templates
  to disk for normal downstream use.

Outputs:
- Writes chosen templates to ./templates/<digit>/*.jpg
- Saves chosen coordinates to chosen_templates.json

Run:
    python select_best_templates_fast.py
"""

import os
import json
import shutil
from typing import Dict, List, Tuple

import numpy as np
from skimage.io import imsave
from skimage.feature import match_template

# Project imports
from const import (
    TRAIN_IMAGES_PATH,
    TEMPLATES_PATH,
    ALL_CELL_COORDINATES,
    MAX_TEMPLATES_FOR_DIGIT,
    TRUTH,
)
from utils import read_image
from template import get_template_pipeline
from recognition import is_empty


Digit = int
Coord = Tuple[int, int]
ImageName = str
Cell = np.ndarray

# Candidate = (img_name, coord, cell)
CandidatesByDigit = Dict[Digit, List[Tuple[ImageName, Coord, Cell]]]
Selection = Dict[Digit, List[Tuple[ImageName, Coord]]]


def ensure_clean_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def backup_templates_if_any(templates_path: str, backup_path: str):
    if os.path.isdir(templates_path):
        shutil.rmtree(backup_path, ignore_errors=True)
        shutil.copytree(templates_path, backup_path)


def cache_sudoku_cells(images: List[str]) -> Dict[ImageName, np.ndarray]:
    pipeline = get_template_pipeline()
    cache: Dict[ImageName, np.ndarray] = {}

    for file_name in images:
        image_path = os.path.join(TRAIN_IMAGES_PATH, file_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path} (skipping)")
            continue
        img = read_image(image_path=image_path)
        _, sudoku_cells = pipeline(img)
        cache[file_name] = sudoku_cells
    return cache


def build_candidates(cells_cache: Dict[ImageName, np.ndarray]) -> CandidatesByDigit:
    candidates: CandidatesByDigit = {d: [] for d in range(1, 10)}
    for img_name, per_digit in ALL_CELL_COORDINATES.items():
        if img_name not in cells_cache:
            continue
        sudoku_cells = cells_cache[img_name]
        for d, coords_list in per_digit.items():
            for coord in coords_list:
                i, j = coord
                cell = sudoku_cells[i, j].copy()
                candidates[d].append((img_name, coord, cell))
    return candidates


def precompute_empty_mask(cells_cache: Dict[ImageName, np.ndarray]) -> Dict[ImageName, np.ndarray]:
    empty: Dict[ImageName, np.ndarray] = {}
    for img_name, cells in cells_cache.items():
        mask = np.zeros((cells.shape[0], cells.shape[1]), dtype=bool)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                mask[i, j] = is_empty(cells[i, j])
        empty[img_name] = mask
    return empty


def flatten_truth_cells() -> List[Tuple[ImageName, int, int, int]]:
    items: List[Tuple[ImageName, int, int, int]] = []
    for img_name, gt in TRUTH.items():
        nz = np.argwhere(gt != 0)
        for i, j in nz:
            items.append((img_name, int(i), int(j), int(gt[i, j])))
    return items


def precompute_candidate_correlations(
    cells_cache: Dict[ImageName, np.ndarray],
    candidates: CandidatesByDigit,
) -> Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]]:
    """
    For each digit and each candidate template, compute correlation with EVERY cell
    (for ALL images), using the exact call pattern from recognition.get_digit_correlations.
    Returns: digit -> list (per candidate) of dicts mapping (img,i,j) -> corr.
    """
    corr: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]] = {d: [] for d in range(1, 10)}

    # Flatten cells
    all_cells_index: List[Tuple[ImageName, int, int]] = []
    all_cells_data: List[np.ndarray] = []
    for img_name, cells in cells_cache.items():
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                all_cells_index.append((img_name, i, j))
                all_cells_data.append(cells[i, j])

    # Compute correlations
    for d in range(1, 10):
        cand_list = candidates[d]
        for (_img, _coord, tmpl) in cand_list:
            per_cell: Dict[Tuple[ImageName, int, int], float] = {}
            for idx, cell in enumerate(all_cells_data):
                c = match_template(image=cell, template=tmpl, pad_input=True, constant_values=0).max()
                per_cell[all_cells_index[idx]] = float(abs(c))
            corr[d].append(per_cell)
    return corr


def coords_to_candidate_index_map(candidates: CandidatesByDigit) -> Dict[Digit, Dict[Tuple[str, Tuple[int, int]], int]]:
    idx_map: Dict[Digit, Dict[Tuple[str, Tuple[int, int]], int]] = {d: {} for d in range(1, 10)}
    for d in range(1, 10):
        for idx, (img, coord, _cell) in enumerate(candidates[d]):
            idx_map[d][(img, tuple(coord))] = idx
    return idx_map


def evaluate_selection(
    selection: Selection,
    candidates: CandidatesByDigit,
    corr_cache: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]],
    empty_mask: Dict[ImageName, np.ndarray],
    threshold: float = 0.35,
):
    """
    Compute predictions for ALL cells using precomputed correlations.
    Recognition rule mirrors recognition.recognize_digits.
    Returns preds: dict img_name -> 9x9 uint8 matrix.
    """
    # Build quick lookup of selected candidate indices per digit
    idx_map = coords_to_candidate_index_map(candidates)
    selected_idx: Dict[int, List[int]] = {}
    for d in range(1, 10):
        sel = selection.get(d, [])
        selected_idx[d] = []
        for (img, coord) in sel:
            idx = idx_map[d].get((img, tuple(coord)))
            if idx is not None:
                selected_idx[d].append(idx)

    # Build predictions
    preds: Dict[ImageName, np.ndarray] = {}
    for img_name, mask in empty_mask.items():
        H, W = mask.shape
        out = np.zeros((H, W), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                if mask[i, j]:  # empty cell
                    out[i, j] = 0
                    continue

                # For each digit, take max corr over selected templates
                best = 0.0
                best_d = 0
                for d in range(1, 10):
                    cands = selected_idx.get(d, [])
                    if not cands:
                        continue
                    m = 0.0
                    for idx in cands:
                        m = max(m, corr_cache[d][idx][(img_name, i, j)])
                    if m < threshold:
                        m = 0.0
                    if m > best:
                        best = m
                        best_d = d

                out[i, j] = best_d if best > 0.0 else 0
        preds[img_name] = out
    return preds


def accuracy_from_preds(preds: Dict[str, np.ndarray]) -> Tuple[int, int, float]:
    total_correct = 0
    total = 0
    for img_name, gt in TRUTH.items():
        if img_name not in preds:
            continue
        P = preds[img_name]
        nz = np.argwhere(gt != 0)
        for i, j in nz:
            total += 1
            if int(P[i, j]) == int(gt[i, j]):
                total_correct += 1
    acc = (total_correct / total) if total else 0.0
    return total_correct, total, acc


def greedy_select_best_templates(
    candidates: CandidatesByDigit,
    corr_cache: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]],
    empty_mask: Dict[ImageName, np.ndarray],
) -> Selection:
    selection: Selection = {d: [] for d in range(1, 10)}

    # Baseline
    preds = evaluate_selection(selection, candidates, corr_cache, empty_mask)
    best_correct, best_total, best_acc = accuracy_from_preds(preds)
    print(f"Baseline: correct={best_correct}/{best_total} ({best_acc:.2%})")

    for d in range(1, 10):
        used = set((img, tuple(coord)) for img, coord in selection[d])
        for _k in range(MAX_TEMPLATES_FOR_DIGIT):
            best_improvement = None  # (gain, (img,coord), metrics)
            for (img_name, coord, _cell) in candidates[d]:
                key = (img_name, tuple(coord))
                if key in used:
                    continue

                trial = {dd: sel.copy() for dd, sel in selection.items()}
                trial[d] = trial[d] + [(img_name, coord)]

                preds_t = evaluate_selection(trial, candidates, corr_cache, empty_mask)
                total_correct, total_samples, acc = accuracy_from_preds(preds_t)
                gain = total_correct - best_correct

                if (best_improvement is None) or (gain > best_improvement[0]) or (
                    gain == best_improvement[0] and acc > best_improvement[2][2]
                ):
                    best_improvement = (gain, (img_name, coord), (total_correct, total_samples, acc))

            if best_improvement is None or best_improvement[0] <= 0:
                break

            chosen_img, chosen_coord = best_improvement[1]
            selection[d].append((chosen_img, chosen_coord))
            used.add((chosen_img, tuple(chosen_coord)))
            best_correct, best_total, best_acc = best_improvement[2]
            print(f"[digit {d}] + {chosen_img} {chosen_coord} -> {best_correct}/{best_total} ({best_acc:.2%})")

    return selection


def write_templates_folder_from_selection(
    selection: Selection,
    candidates: CandidatesByDigit,
    out_dir: str,
):
    ensure_clean_dir(out_dir)
    for d in range(1, 10):
        ddir = os.path.join(out_dir, str(d))
        os.makedirs(ddir, exist_ok=True)
        chosen = selection.get(d, [])
        idx = 0
        # Save the exact cell crops that were chosen
        for img_name, coord in chosen:
            # Look up the candidate to fetch the cell image
            found = None
            for c_img, c_coord, c_cell in candidates[d]:
                if c_img == img_name and tuple(c_coord) == tuple(coord):
                    found = c_cell
                    break
            if found is None:
                continue
            fname = f"{os.path.splitext(img_name)[0]}_{d}_{idx}.jpg"
            imsave(os.path.join(ddir, fname), found)
            idx += 1


def main():
    # Backup templates
    backup_path = "./templates.backup_autosave"
    if os.path.isdir(TEMPLATES_PATH):
        print(f"Backing up existing templates to {backup_path} ...")
    backup_templates_if_any(TEMPLATES_PATH, backup_path)

    # Determine which images to use (all that appear in TRUTH)
    images = sorted(TRUTH.keys())

    # 1) Cache sudoku cells
    print("Caching sudoku cells...")
    cells_cache = cache_sudoku_cells(images)
    print(f"  cached: {len(cells_cache)} images")

    # 2) Precompute empty mask
    print("Precomputing empty masks...")
    empty_mask = precompute_empty_mask(cells_cache)

    # 3) Build candidates
    print("Building candidates...")
    candidates = build_candidates(cells_cache)
    for d in range(1, 10):
        print(f"  digit {d}: {len(candidates[d])} candidates")

    # 4) Precompute candidate-to-cell correlations
    print("Precomputing candidate correlations (this is the heavy step, done once)...")
    corr_cache = precompute_candidate_correlations(cells_cache, candidates)
    print("  correlations computed.")

    # 5) Greedy selection using fast evaluation
    print("Selecting templates (greedy, up to 2 per digit)...")
    selection = greedy_select_best_templates(candidates, corr_cache, empty_mask)

    # 6) Final in-memory evaluation
    preds = evaluate_selection(selection, candidates, corr_cache, empty_mask)
    total_correct, total_samples, acc = accuracy_from_preds(preds)
    print("\nFinal selection:")
    for d in range(1, 10):
        print(f"  {d}: {selection[d]}")
    print(f"\nFinal accuracy: {total_correct}/{total_samples} ({acc:.2%})")

    # 7) Materialize templates to ./templates for normal downstream code
    write_templates_folder_from_selection(selection, candidates, TEMPLATES_PATH)
    print("Wrote selected templates into ./templates/")

    # 8) Save JSON
    with open("chosen_templates.json", "w", encoding="utf-8") as f:
        json.dump({str(d): selection[d] for d in range(1, 10)}, f, indent=2)
    print("Saved chosen_templates.json")

    print("\nDone.")
    

if __name__ == "__main__":
    main()
