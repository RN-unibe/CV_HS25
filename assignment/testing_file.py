import os

from const import TRAIN_IMAGES_PATH
from utils import read_image, show_image
from pipeline import Pipeline
from frontalization import find_edges, highlight_edges, find_contours, get_max_contour, find_corners, rescale_image, gaussian_blur, frontalize_image, show_frontalized_images

from const import SUDOKU_SIZE
from recognition import resize_image, get_sudoku_cells

from const import SUDOKU_SIZE
from recognition import resize_image

from recognition import get_sudoku_cells

from utils import load_templates
from recognition import get_digit_correlations, show_correlations, recognize_digits, show_recognized_digits

from utils import load_templates
from recognition import get_digit_correlations, show_correlations

image_path = os.path.join(TRAIN_IMAGES_PATH, "image_1.jpg")
sudoku_image = read_image(image_path=image_path)


if __name__=="__main__":
    pipeline = Pipeline(functions=[gaussian_blur, find_edges, highlight_edges, find_contours, get_max_contour, find_corners, frontalize_image,
                               resize_image, get_sudoku_cells],
                    parameters={"gaussian_blur": {"sigma": 0.42}, # play with the "sigma" parameter
                                "find_corners": {"epsilon": 6.6}, # play with the "epsilon" parameter
                                "resize_image": {"size": SUDOKU_SIZE},
                                # play with the "crop_factor" parameter and binarization_kwargs
                                "get_sudoku_cells": {"crop_factor":0.65, "binarization_kwargs": {}}
                               })
    frontalized_image, sudoku_cells = pipeline(sudoku_image, plot=True, figsize=(24, 18))

    templates_dict = load_templates()

    sudoku_cell = sudoku_cells[6, 1]

    correlations = get_digit_correlations(sudoku_cell, templates_dict)
    show_correlations(sudoku_cell, correlations)