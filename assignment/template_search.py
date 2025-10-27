#!/usr/bin/env python3
"""
Template search for Sudoku digit recognition.

Goal:
- For each digit d in 1..9, and for the provided set of labeled (image, i, j) cells,
  find which subset of templates for digit d yields the best recognition accuracy
  on those cells, while keeping the other digits' templates unchanged.

Method:
1) Load and cache sudoku_cells for every image we'll evaluate.
2) Load templates (dict: digit -> list of template images).
3) For each digit d:
   a) Evaluate each single template alone (for d) to get a ranking.
   b) Greedy forward selection: start from best single, then add the next template
      that improves accuracy the most, up to max_k.
4) Save detailed CSV of per-template results and a summary of chosen subsets.
"""

import os
import copy
import numpy as np
import pandas as pd

from const import TRAIN_IMAGES_PATH, ALL_CELL_COORDINATES
from utils import read_image, load_templates
from template import get_template_pipeline
from recognition import recognize_digits


DIGITS = list(range(1, 10))

def build_digit_index(coords_by_image: dict) -> dict:
    """Invert ALL_CELL_COORDINATES into per-digit list of (image, i, j)."""
    per_digit = {d: [] for d in DIGITS}
    for img, by_digit in coords_by_image.items():
        for d, coords in by_digit.items():
            for (i, j) in coords:
                per_digit[d].append((img, i, j))
    return per_digit

def cache_all_cells(images: set, pipeline):
    """Run pipeline once per image; return {image: sudoku_cells}."""
    cache = {}
    for img in images:
        path = os.path.join(TRAIN_IMAGES_PATH, img)
        if not os.path.exists(path):
            print(f"[WARN] missing image: {path}")
            continue
        sudoku_img = read_image(image_path=path)
        _, sudoku_cells = pipeline(sudoku_img)
        cache[img] = sudoku_cells
    return cache

def recognize_with_templates(cells_cache: dict, templates_dict: dict, threshold: float = 0.5):
    """Run recognition for all cached images. Return {image: recognized_matrix}."""
    results = {}
    for img, cells in cells_cache.items():
        recognized = recognize_digits(cells, templates_dict, threshold=threshold)
        results[img] = recognized
    return results

def accuracy_for_digit(results: dict, per_digit_coords: dict, d: int) -> float:
    """Compute accuracy for digit d over its labeled coords given recognition results."""
    coords = per_digit_coords.get(d, [])
    if not coords:
        return float("nan")
    total = 0
    correct = 0
    for (img, i, j) in coords:
        if img not in results:
            continue
        pred = int(results[img][i, j])
        total += 1
        if pred == d:
            correct += 1
    if total == 0:
        return float("nan")
    return correct / total

def template_subset(base_templates: dict, d: int, idxs: list) -> dict:
    """Return a deep-copied templates dict where digit d's list is replaced by subset specified by idxs."""
    td = copy.deepcopy(base_templates)
    td[d] = [base_templates[d][k] for k in idxs]
    return td

def greedy_select_for_digit(d, base_templates, cells_cache, per_digit_coords, max_k=5, threshold=0.5):
    """Greedy forward selection of templates for digit d. Returns (chosen_idxs, history_df)."""
    n_t = len(base_templates[d])
    if n_t == 0:
        return [], pd.DataFrame()

    # 1) Score each template alone
    rows = []
    single_scores = []
    for idx in range(n_t):
        tdict = template_subset(base_templates, d, [idx])
        res = recognize_with_templates(cells_cache, tdict, threshold)
        acc = accuracy_for_digit(res, per_digit_coords, d)
        rows.append({"digit": d, "stage": "single", "subset": (idx,), "accuracy": acc})
        single_scores.append((idx, acc))

    # Rank singles
    single_scores.sort(key=lambda x: (x[1] if x[1]==x[1] else -1), reverse=True)  # NaN-safe

    # 2) Greedy build
    chosen = [single_scores[0][0]]
    # Evaluate the currently chosen subset
    tdict = template_subset(base_templates, d, chosen)
    res = recognize_with_templates(cells_cache, tdict, threshold)
    best_acc = accuracy_for_digit(res, per_digit_coords, d)
    rows.append({"digit": d, "stage": "greedy", "subset": tuple(chosen), "accuracy": best_acc})

    remaining = [idx for idx in range(n_t) if idx not in chosen]

    while len(chosen) < max_k and remaining:
        best_candidate = None
        best_candidate_acc = best_acc
        for idx in remaining:
            trial = chosen + [idx]
            tdict = template_subset(base_templates, d, trial)
            res = recognize_with_templates(cells_cache, tdict, threshold)
            acc = accuracy_for_digit(res, per_digit_coords, d)
            rows.append({"digit": d, "stage": f"try_add_{len(trial)}", "subset": tuple(trial), "accuracy": acc})
            if acc > best_candidate_acc:
                best_candidate_acc = acc
                best_candidate = idx
        if best_candidate is None:  # no improvement
            break
        chosen.append(best_candidate)
        best_acc = best_candidate_acc
        remaining.remove(best_candidate)
        rows.append({"digit": d, "stage": "greedy", "subset": tuple(chosen), "accuracy": best_acc})

    return chosen, pd.DataFrame(rows)

def main(output_csv_details="template_search_details.csv",
         output_csv_summary="template_search_summary.csv",
         max_k=5, threshold=0.5):
    pipeline = get_template_pipeline()
    base_templates = load_templates()

    # Which images do we need?
    images = set(ALL_CELL_COORDINATES.keys())
    cells_cache = cache_all_cells(images, pipeline)

    # Invert coords to per-digit lists
    per_digit_coords = build_digit_index(ALL_CELL_COORDINATES)

    all_rows = []
    summary_rows = []

    for d in DIGITS:
        if len(base_templates.get(d, [])) == 0:
            print(f"[INFO] digit {d}: no templates found, skipping")
            continue

        chosen, hist = greedy_select_for_digit(
            d, base_templates, cells_cache, per_digit_coords, max_k=max_k, threshold=threshold
        )
        all_rows.append(hist)

        # Final accuracy of chosen subset
        tdict = template_subset(base_templates, d, chosen)
        res = recognize_with_templates(cells_cache, tdict, threshold)
        final_acc = accuracy_for_digit(res, per_digit_coords, d)

        summary_rows.append({
            "digit": d,
            "n_templates_available": len(base_templates[d]),
            "chosen_template_indices": chosen,
            "chosen_count": len(chosen),
            "final_accuracy": final_acc
        })

        print(f"Digit {d}: chosen {chosen} -> accuracy {final_acc:.1%} (from {len(base_templates[d])} available)")

    if all_rows:
        details_df = pd.concat(all_rows, ignore_index=True)
        details_df.to_csv(output_csv_details, index=False)
        print(f"Saved details to {output_csv_details}")
    else:
        details_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).sort_values("digit")
    summary_df.to_csv(output_csv_summary, index=False)
    print(f"Saved summary to {output_csv_summary}")

    return details_df, summary_df

if __name__ == "__main__":
    main()
