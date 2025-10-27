#!/usr/bin/env python3
"""
Evaluate Sudoku digit recognition against ground-truth entries.

- Loads templates from ./templates (via utils.load_templates)
- Builds the recognition pipeline (via template.get_template_pipeline)
- For each image listed in TRUTH, runs the pipeline to get sudoku_cells
- Recognizes digits (recognition.recognize_digits)
- Compares only the NON-ZERO ground-truth cells (given digits) and reports correctness
- Saves a CSV with per-cell results and prints per-image summaries
"""

import os
import sys
import numpy as np
import pandas as pd

from const import TRAIN_IMAGES_PATH, TRUTH
from utils import read_image, load_templates
from template import get_template_pipeline
from recognition import recognize_digits


def evaluate(recognized: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """
    Compare recognized matrix against truth. Returns long-form rows for non-zero truth cells.
    Columns: image, i, j, truth, pred, correct
    """
    rows = []
    for i in range(9):
        for j in range(9):
            t = int(truth[i, j])
            if t == 0:  # only check given digits
                continue
            p = int(recognized[i, j])
            rows.append({"i": i, "j": j, "truth": t, "pred": p, "correct": p == t})
    return pd.DataFrame(rows)

def evaluate_all(save_csv: bool = False, csv_path: str = "recognition_eval.csv"):
    templates = load_templates()
    pipeline = get_template_pipeline()

    all_rows = []
    per_image_summary = []

    for file_name, truth in TRUTH.items():
        image_path = os.path.join(TRAIN_IMAGES_PATH, file_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}. Skipping.")
            continue

        sudoku_img = read_image(image_path=image_path)
        _, sudoku_cells = pipeline(sudoku_img)

        recognized = recognize_digits(sudoku_cells, templates)
        df = evaluate(recognized, truth)
        if df.empty:
            print(f"[WARN] No non-zero ground-truth cells for {file_name}?")
            continue

        df["image"] = file_name
        all_rows.append(df)

        acc = df["correct"].mean()
        n_correct = int(df["correct"].sum())
        n_total = int(df.shape[0])
        per_image_summary.append({"image": file_name, "accuracy": acc, "n_correct": n_correct, "n_total": n_total})

        print(f"{file_name}: {n_correct}/{n_total} correct  (accuracy {acc:.1%})")

        # Also, show mismatches briefly
        mismatches = df.loc[~df["correct"]]
        if not mismatches.empty:
            print("  mismatches (i,j): truth -> pred")
            for _, r in mismatches.iterrows():
                print(f"   ({int(r.i)},{int(r.j)}): {int(r.truth)} -> {int(r.pred)}")

    if not all_rows:
        print("No results to save.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(per_image_summary).sort_values("image")

    if save_csv:
        results.to_csv(csv_path, index=False)
        print(f"\nSaved per-cell results to: {csv_path}")

    # Print summary nicely
    print("\nPer-image summary:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary.to_string(index=False))

    return results, summary

if __name__ == "__main__":
    evaluate_all()
