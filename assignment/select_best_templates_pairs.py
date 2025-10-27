#!/usr/bin/env python3
"""
select_best_templates_pairs.py

Fast joint search that selects EXACTLY 2 templates per digit (9 digits × 2 = 18 templates total).
It optimizes GLOBAL recognition accuracy via coordinate ascent:

1) Cache sudoku_cells once per image (no repeated pipeline runs).
2) Precompute is_empty mask per cell.
3) Build candidate pools from ALL_CELL_COORDINATES.
4) Precompute correlation scores between every candidate template and every cell,
   using the same call as in recognition.get_digit_correlations:
   match_template(image=cell, template=tmpl, pad_input=True, constant_values=0).max()
5) Build all 2-combinations per digit.
6) Coordinate ascent:
   - Initialize each digit with a reasonable starting pair.
   - For several passes: for each digit d, try all its pairs while holding other digits fixed,
     pick the pair that maximizes GLOBAL accuracy. Stop early when no digit changes.

Finally, write the 18 chosen crops to ./templates/<digit>/*.jpg and save chosen_templates.json.

Run:
    python select_best_templates_pairs.py
"""

import os
import json
import math
import shutil
import itertools
from typing import Dict, List, Tuple

import numpy as np
from skimage.io import imsave
from skimage.feature import match_template

# Project imports
from const import (
    TRAIN_IMAGES_PATH,
    TEMPLATES_PATH,
    ALL_CELL_COORDINATES,
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
PairIndex = Tuple[int, int]  # indices into candidates[d]
PairsByDigit = Dict[Digit, List[PairIndex]]
PairsSelection = Dict[Digit, PairIndex]


# ------------------------------- IO helpers -------------------------------

def ensure_clean_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def backup_templates_if_any(templates_path: str, backup_path: str):
    if os.path.isdir(templates_path):
        shutil.rmtree(backup_path, ignore_errors=True)
        shutil.copytree(templates_path, backup_path)


# ---------------------------- Pipeline caching ----------------------------

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


def precompute_empty_mask(cells_cache: Dict[ImageName, np.ndarray]) -> Dict[ImageName, np.ndarray]:
    empty: Dict[ImageName, np.ndarray] = {}
    for img_name, cells in cells_cache.items():
        mask = np.zeros((cells.shape[0], cells.shape[1]), dtype=bool)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                mask[i, j] = is_empty(cells[i, j])
        empty[img_name] = mask
    return empty


# --------------------------- Candidates & truth ---------------------------

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


def flatten_truth_cells() -> List[Tuple[ImageName, int, int, int]]:
    items: List[Tuple[ImageName, int, int, int]] = []
    for img_name, gt in TRUTH.items():
        nz = np.argwhere(gt != 0)
        for i, j in nz:
            items.append((img_name, int(i), int(j), int(gt[i, j])))
    return items


# ----------------------- Correlations precomputation ----------------------

def precompute_candidate_correlations(
    cells_cache: Dict[ImageName, np.ndarray],
    candidates: CandidatesByDigit,
) -> Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]]:
    """
    For each candidate template and for every cell in every image, compute
    match_template(image=cell, template=tmpl, pad_input=True, constant_values=0).max()
    Return corr[d][k][(img,i,j)] = correlation score for digit d, candidate k, cell (img,i,j).
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


# ------------------------------ Pair building -----------------------------

def build_pairs(candidates: CandidatesByDigit) -> PairsByDigit:
    pairs: PairsByDigit = {d: [] for d in range(1, 10)}
    for d in range(1, 10):
        n = len(candidates[d])
        if n == 0:
            pairs[d] = []
        elif n == 1:
            pairs[d] = [(0, 0)]  # duplicate the single candidate
        else:
            pairs[d] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return pairs


# ------------------------------ Evaluation --------------------------------

def evaluate_pairs_selection(
    sel: PairsSelection,
    candidates: CandidatesByDigit,
    corr_cache: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]],
    empty_mask: Dict[ImageName, np.ndarray],
    threshold: float = 0.35,
) -> Tuple[int, int, float]:
    """
    Compute GLOBAL accuracy for the given selection of pairs (two template indices per digit).
    Returns (total_correct, total, accuracy).
    """
    total_correct = 0
    total = 0

    for img_name, gt in TRUTH.items():
        if img_name not in empty_mask:
            continue
        H, W = gt.shape
        for i in range(H):
            for j in range(W):
                if gt[i, j] == 0:
                    continue  # only evaluate non-zero cells from ground truth
                if empty_mask[img_name][i, j]:
                    # should not happen for non-zero GT, but keep robust
                    pred = 0
                else:
                    best = 0.0
                    best_d = 0
                    for d in range(1, 10):
                        p = sel.get(d)
                        if p is None:
                            continue
                        a, b = p
                        # handle degenerate (one candidate only) pairs
                        if a == b:
                            m = corr_cache[d][a][(img_name, i, j)]
                        else:
                            m = max(corr_cache[d][a][(img_name, i, j)],
                                    corr_cache[d][b][(img_name, i, j)])
                        if m < threshold:
                            m = 0.0
                        if m > best:
                            best = m
                            best_d = d
                    pred = best_d if best > 0.0 else 0

                total += 1
                if int(pred) == int(gt[i, j]):
                    total_correct += 1

    acc = (total_correct / total) if total else 0.0
    return total_correct, total, acc


# ---------------------------- Initialization ------------------------------

def initial_pairs_selection(
    candidates: CandidatesByDigit,
    corr_cache: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]],
) -> PairsSelection:
    """
    Heuristic init: for each digit d, score each candidate by the sum of correlations
    on all GT cells with value d. Pick the top-2 candidates as the initial pair.
    If fewer than 2 candidates, duplicate the single one.
    """
    sel: PairsSelection = {}
    # Precompute GT locations per digit
    gt_pos_by_digit: Dict[int, List[Tuple[ImageName, int, int]]] = {d: [] for d in range(1, 10)}
    for img_name, gt in TRUTH.items():
        nz = np.argwhere(gt == 0)
        # we only need positions with gt == d, so we compute per digit explicitly
        pass
    # Actually compute per-digit positions
    for d in range(1, 10):
        locs = []
        for img_name, gt in TRUTH.items():
            nz = np.argwhere(gt == d)
            for i, j in nz:
                locs.append((img_name, int(i), int(j)))
        gt_pos_by_digit[d] = locs

    for d in range(1, 10):
        n = len(candidates[d])
        if n == 0:
            continue
        if n == 1:
            sel[d] = (0, 0)
            continue

        # Score each candidate by sum of correlations on GT cells of this digit
        scores = []
        for k in range(n):
            s = 0.0
            per_cell = corr_cache[d][k]
            for (img_name, i, j) in gt_pos_by_digit[d]:
                s += per_cell.get((img_name, i, j), 0.0)
            scores.append((s, k))
        scores.sort(reverse=True)  # highest first
        best_two = [scores[0][1], scores[1][1]] if len(scores) >= 2 else [scores[0][1], scores[0][1]]
        sel[d] = (best_two[0], best_two[1])

    return sel


# -------------------------- Coordinate ascent -----------------------------

def coordinate_ascent_pairs(
    pairs: PairsByDigit,
    candidates: CandidatesByDigit,
    corr_cache: Dict[Digit, List[Dict[Tuple[ImageName, int, int], float]]],
    empty_mask: Dict[ImageName, np.ndarray],
    max_iters: int = 5,
    threshold: float = 0.35,
) -> PairsSelection:
    """
    Optimize global accuracy by iteratively re-choosing the best pair for each digit
    while holding the other digits fixed. Stop if no change in a full pass or when
    max_iters reached.
    """
    # Initialize
    sel = initial_pairs_selection(candidates, corr_cache)
    best_correct, best_total, best_acc = evaluate_pairs_selection(sel, candidates, corr_cache, empty_mask, threshold)
    print(f"Init: {best_correct}/{best_total} ({best_acc:.2%})")

    for it in range(max_iters):
        changed = False
        print(f"\nPass {it+1}/{max_iters}")
        for d in range(1, 10):
            current = sel.get(d)
            best_local = (best_correct, best_total, best_acc)
            best_pair = current

            for p in pairs[d] if pairs[d] else [(0, 0)]:
                if p == current:
                    continue
                trial = dict(sel)
                trial[d] = p
                tc, tt, acc = evaluate_pairs_selection(trial, candidates, corr_cache, empty_mask, threshold)
                if (tc > best_local[0]) or (tc == best_local[0] and acc > best_local[2]):
                    best_local = (tc, tt, acc)
                    best_pair = p

            if best_pair != current:
                sel[d] = best_pair
                best_correct, best_total, best_acc = best_local
                changed = True
                print(f"  [digit {d}] -> {best_pair}  now {best_correct}/{best_total} ({best_acc:.2%})")

        if not changed:
            print("Converged (no changes this pass).")
            break

    return sel


# ------------------------------ Finalization ------------------------------

def write_templates_from_pairs_selection(
    sel: PairsSelection,
    candidates: CandidatesByDigit,
    out_dir: str,
):
    ensure_clean_dir(out_dir)
    for d in range(1, 10):
        ddir = os.path.join(out_dir, str(d))
        os.makedirs(ddir, exist_ok=True)
        p = sel.get(d)
        if p is None:
            continue
        a, b = p
        chosen_idxs = [a, b]
        idx = 0
        for k in chosen_idxs:
            if k < 0 or k >= len(candidates[d]):
                continue
            img_name, coord, cell = candidates[d][k]
            fname = f"{os.path.splitext(img_name)[0]}_{d}_{idx}.jpg"
            imsave(os.path.join(ddir, fname), cell)
            idx += 1


def pairs_selection_to_coords(sel: PairsSelection, candidates: CandidatesByDigit):
    out = {str(d): [] for d in range(1, 10)}
    for d in range(1, 10):
        if d not in sel:
            continue
        a, b = sel[d]
        for k in [a, b]:
            if k < 0 or k >= len(candidates[d]):
                continue
            img_name, coord, _ = candidates[d][k]
            out[str(d)].append([img_name, list(map(int, coord))])
    return out


# ---------------------------------- Main ----------------------------------

def main():
    # Backup current templates
    backup_path = "./templates.backup_autosave"
    if os.path.isdir(TEMPLATES_PATH):
        print(f"Backing up existing templates to {backup_path} ...")
    backup_templates_if_any(TEMPLATES_PATH, backup_path)

    # Use all images that have TRUTH
    images = sorted(TRUTH.keys())

    # 1) Cache sudoku cells
    print("Caching sudoku cells...")
    cells_cache = cache_sudoku_cells(images)
    print(f"  cached: {len(cells_cache)} images")

    # 2) Empty mask
    print("Precomputing empty masks...")
    empty_mask = precompute_empty_mask(cells_cache)

    # 3) Candidates
    print("Building candidates...")
    candidates = build_candidates(cells_cache)
    for d in range(1, 10):
        print(f"  digit {d}: {len(candidates[d])} candidates")

    # 4) Correlations
    print("Precomputing candidate correlations (heavy step, once)...")
    corr_cache = precompute_candidate_correlations(cells_cache, candidates)
    print("  correlations computed.")

    # 5) Pairs per digit
    pairs = build_pairs(candidates)
    for d in range(1, 10):
        print(f"  digit {d}: {len(pairs[d])} pairs")

    # 6) Coordinate ascent joint optimization
    print("Optimizing 2-per-digit selection (joint, coordinate ascent)...")
    sel = coordinate_ascent_pairs(pairs, candidates, corr_cache, empty_mask, max_iters=6, threshold=0.35)

    # 7) Final score
    tc, tt, acc = evaluate_pairs_selection(sel, candidates, corr_cache, empty_mask, threshold=0.35)
    print(f"\nFinal: {tc}/{tt} ({acc:.2%})")

    # 8) Materialize to ./templates
    write_templates_from_pairs_selection(sel, candidates, TEMPLATES_PATH)
    print("Wrote selected templates into ./templates/")

    # 9) Save coordinates JSON
    coords = pairs_selection_to_coords(sel, candidates)
    with open("chosen_templates.json", "w", encoding="utf-8") as f:
        json.dump(coords, f, indent=2)
    print("Saved chosen_templates.json")

    print("\nDone.")

if __name__ == "__main__":
    main()
