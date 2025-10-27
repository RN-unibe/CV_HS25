from unittest import main, TestCase
import os
import numpy as np

from const import TRAIN_IMAGES_PATH, TRUTH
from utils import read_image, show_image
from pipeline import Pipeline
from frontalization import find_edges, highlight_edges, find_contours, get_max_contour, find_corners, rescale_image, gaussian_blur, frontalize_image, show_frontalized_images

from const import SUDOKU_SIZE


from utils import load_templates
from recognition import get_sudoku_cells, resize_image, get_sudoku_cells, get_digit_correlations, show_correlations, recognize_digits, show_recognized_digits


from sudoku_solver import matrix_to_puzzle, solve_sudoku
from recognition import show_solved_sudoku, recognize_digits, show_recognized_digits

import itertools

import os
import unittest
import numpy as np
import pandas as pd

from utils import read_image, load_templates
from template import get_template_pipeline, CELL_COORDINATES
from recognition import get_digit_correlations, recognize_digits






PIPELINE = Pipeline(functions=[rescale_image, 
                                            gaussian_blur, 
                                            find_edges, 
                                            highlight_edges, 
                                            find_contours, 
                                            get_max_contour, 
                                            find_corners, 
                                            frontalize_image,
                                            resize_image,
                                            get_sudoku_cells],
                                parameters={"rescale_image": {"scale": 0.6},
                                            "gaussian_blur": {"sigma": .4}, # play with the "sigma" parameter
                                            "find_corners": {"epsilon": 0.9}, # play with the "epsilon" parameter
                                            "resize_image": {"size": SUDOKU_SIZE},
                                            # play with the "crop_factor" parameter and binarization_kwargs
                                            "get_sudoku_cells": {"crop_factor":0.75, "binarization_kwargs": {}}
                                        })


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



class TestRecognission(TestCase) :


    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.templates_dict = load_templates()

 
    def subtest_nr(self, nr='0'):
            image_path = os.path.join(TRAIN_IMAGES_PATH, f"image_{nr}.jpg")
            sudoku_image = read_image(image_path=image_path)
            frontalized_image, sudoku_cells = self.pipeline(sudoku_image, plot=False, figsize=(24, 18))
            sudoku_matrix = recognize_digits(sudoku_cells, self.templates_dict)
            self.assertFalse((sudoku_matrix == 0).all())

            try:
                sudoku_matrix_solved = solve_sudoku(sudoku_matrix)
                show_solved_sudoku(frontalized_image, sudoku_matrix, sudoku_matrix_solved)
            except Exception as e:
                self.fail()



    def test_all(self):
        for nr in range(0, 10):
            with self.subTest(image_number=nr): 
                self.subtest_nr(nr)

    def test_all_possibilities(self):
        pass

        

if __name__=="__main__":
    main(verbosity=2)

    


    



