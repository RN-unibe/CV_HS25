#!/usr/bin/env python3
"""
select_best_templates.py

Greedy search for the best Sudoku digit templates (max 2 per digit) using
evaluate_recognition.evaluate_all as the scoring function.

It uses ALL_CELL_COORDINATES to propose candidate cells, runs your existing
pipeline once per image to cache sudoku_cells, then greedily adds templates
that most improve the evaluation score, up to MAX_TEMPLATES_FOR_DIGIT (2).

Outputs:
- Writes chosen templates to ./templates/<digit>/*.jpg (overwrites!)
- Saves chosen coordinates to chosen_templates.json (for reproducibility).
- Leaves a backup of the previous ./templates at ./templates.backup_autosave

Run:
    python select_best_templates.py

Requirements:
    Keep your repository layout the same as in your assignment. This script
    imports your modules (const, utils, template, evaluate_recognition) and
    assumes TRAIN_IMAGES_PATH contains the training puzzles.

Author: ChatGPT
"""

import os
import json
import shutil
from typing import Dict, List, Tuple

import numpy as np
from skimage.io import imsave

# Project imports (must be runnable from repo root)
from const import (
    TRAIN_IMAGES_PATH,
    TEMPLATES_PATH,
    ALL_CELL_COORDINATES,
    MAX_TEMPLATES_FOR_DIGIT,
)
from utils import read_image
from template import get_template_pipeline
from evaluate_recognition import evaluate_all


Digit = int
Coord = Tuple[int, int]
ImageName = str
Cell = np.ndarray
Selection = Dict[Digit, List[Tuple[ImageName, Coord]]]


def ensure_clean_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def backup_templates_if_any(templates_path: str, backup_path: str):
    if os.path.isdir(templates_path):
        shutil.rmtree(backup_path, ignore_errors=True)
        shutil.copytree(templates_path, backup_path)


def cache_sudoku_cells() -> Dict[ImageName, np.ndarray]:
    """
    Runs the pipeline once per image used for candidates and caches sudoku_cells.
    Returns a dict mapping image file name -> sudoku_cells array [9,9,S,S].
    """
    pipeline = get_template_pipeline()
    cache: Dict[ImageName, np.ndarray] = {}

    image_names = sorted(ALL_CELL_COORDINATES.keys())
    for file_name in image_names:
        image_path = os.path.join(TRAIN_IMAGES_PATH, file_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path} (skipping)")
            continue
        img = read_image(image_path=image_path)
        _, sudoku_cells = pipeline(img)
        cache[file_name] = sudoku_cells
    return cache


def build_candidates(
    cells_cache: Dict[ImageName, np.ndarray],
) -> Dict[Digit, List[Tuple[ImageName, Coord, Cell]]]:
    """
    From ALL_CELL_COORDINATES and the cached sudoku_cells, collect all
    candidate cell crops keyed by digit.
    """
    candidates: Dict[Digit, List[Tuple[ImageName, Coord, Cell]]] = {d: [] for d in range(1, 10)}
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


def write_templates_folder_from_selection(
    selection: Selection,
    candidates: Dict[Digit, List[Tuple[ImageName, Coord, Cell]]],
    out_dir: str,
):
    """
    Write current selection to templates/<digit>/*.jpg
    """
    ensure_clean_dir(out_dir)
    for d in range(1, 10):
        ddir = os.path.join(out_dir, str(d))
        os.makedirs(ddir, exist_ok=True)
        chosen = selection.get(d, [])
        idx = 0
        for img_name, coord in chosen:
            # locate candidate tuple
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


def evaluate_current_selection(selection: Selection, candidates, templates_path: str):
    """
    Write selection to templates_path and call evaluate_all().
    Returns tuple: (total_correct, total_samples, mean_accuracy).
    """
    write_templates_folder_from_selection(selection, candidates, templates_path)
    res = evaluate_all(save_csv=False)
    if not isinstance(res, tuple) or len(res) != 2:
        return (0, 0, 0.0)
    results_df, summary_df = res
    total_correct = int(results_df["correct"].sum())
    total_samples = int(results_df.shape[0])
    mean_acc = float(summary_df["accuracy"].mean()) if not summary_df.empty else 0.0
    return (total_correct, total_samples, mean_acc)


def greedy_select_best_templates(
    candidates: Dict[Digit, List[Tuple[ImageName, Coord, Cell]]],
    templates_path: str,
) -> Selection:
    """
    Greedy forward selection per digit:
    - For each digit 1..9, add up to MAX_TEMPLATES_FOR_DIGIT templates.
    - At each addition, try all remaining candidates for that digit, keep the one
      that maximizes total_correct across all images evaluated by evaluate_all.
    """
    selection: Selection = {d: [] for d in range(1, 10)}

    # Baseline
    best_total_correct, best_total, best_mean_acc = evaluate_current_selection(selection, candidates, templates_path)

    for d in range(1, 10):
        used = set((img, tuple(coord)) for img, coord in selection[d])
        for _k in range(MAX_TEMPLATES_FOR_DIGIT):
            best_improvement = None  # (gain_correct, candidate_tuple, metrics)
            for (img_name, coord, _cell) in candidates[d]:
                key = (img_name, tuple(coord))
                if key in used:
                    continue

                trial = {dd: sel.copy() for dd, sel in selection.items()}
                trial[d] = trial[d] + [(img_name, coord)]

                total_correct, total_samples, mean_acc = evaluate_current_selection(trial, candidates, templates_path)
                gain = total_correct - best_total_correct

                if (best_improvement is None) or (gain > best_improvement[0]) or (
                    gain == best_improvement[0] and mean_acc > best_improvement[2][2]
                ):
                    best_improvement = (gain, (img_name, coord), (total_correct, total_samples, mean_acc))

            if best_improvement is None or best_improvement[0] <= 0:
                break

            chosen_img, chosen_coord = best_improvement[1]
            selection[d].append((chosen_img, chosen_coord))
            used.add((chosen_img, tuple(chosen_coord)))
            best_total_correct, best_total, best_mean_acc = best_improvement[2]
            print(
                f"[digit {d}] Added {chosen_img} {chosen_coord} -> "
                f"correct={best_total_correct}/{best_total} (mean acc {best_mean_acc:.2%})"
            )

    return selection


def main():
    backup_path = "./templates.backup_autosave"
    if os.path.isdir(TEMPLATES_PATH):
        print(f"Backing up existing templates to {backup_path} ...")
    else:
        print("No existing templates directory found.")

    # Backup any current templates
    backup_templates_if_any(TEMPLATES_PATH, backup_path)

    # Build candidate pool
    print("Caching sudoku cells...")
    cells_cache = cache_sudoku_cells()
    print(f"  cached: {len(cells_cache)} images")

    print("Building candidates...")
    candidates = build_candidates(cells_cache)
    for d in range(1, 10):
        print(f"  digit {d}: {len(candidates[d])} candidates")

    print("Selecting templates (greedy, up to 2 per digit)...")
    selection = greedy_select_best_templates(candidates, TEMPLATES_PATH)

    print("\nFinal selection:")
    for d in range(1, 10):
        print(f"  {d}: {selection[d]}")

    # Final evaluation (also ensures templates/ holds the selection files)
    total_correct, total_samples, mean_acc = evaluate_current_selection(selection, candidates, TEMPLATES_PATH)
    overall = (total_correct / total_samples) if total_samples else 0.0
    print(f"\nFinal accuracy: {total_correct}/{total_samples} ({overall:.2%}), mean per-image acc={mean_acc:.2%}")

    # Save JSON
    with open("chosen_templates.json", "w", encoding="utf-8") as f:
        json.dump({str(d): selection[d] for d in range(1, 10)}, f, indent=2)
    print("Saved chosen_templates.json")

    print("\nDone. The ./templates directory now contains the selected templates.")
    print("To restore your previous templates, copy back from ./templates.backup_autosave .")


if __name__ == "__main__":
    main()
