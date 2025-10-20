from pipeline import Pipeline

# BEGIN YOUR IMPORTS
import os

from const import TRAIN_IMAGES_PATH
from utils import read_image

from pipeline import Pipeline
from frontalization import find_edges, highlight_edges, find_contours, get_max_contour, find_corners, rescale_image, gaussian_blur, frontalize_image, show_frontalized_images

from const import SUDOKU_SIZE
from recognition import resize_image, get_sudoku_cells
# END YOUR IMPORTS

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example
"""
"""CELL_COORDINATES = {"image_0.jpg": {3: (1, 3),
                                    8: (0, 5), # "crop_factor":0.7
                                    },
                    "image_1.jpg": {2: (0, 1)},
                    "image_2.jpg": {1: (0, 7), # "crop_factor":0.7
                                    },
                    "image_7.jpg": {1: (0,0),
                                    2: (4,5)},
                    "image_8.jpg": {8: (7, 3), # "crop_factor":0.7
                                    }}
"""

"""CELL_COORDINATES = {"image_0.jpg": {1: (6, 4),
                                    2: (2, 8),
                                    3: (1, 3),
                                    4: (6, 0),
                                    5: (4, 0),
                                    6: (4, 4),
                                    7: (4, 1),
                                    8: (0, 5), 
                                    9: (2, 0)},
                    "image_7.jpg": {1: (0, 0),
                                    2: (4, 5),
                                    3: (2, 1),
                                    4: (1, 0),
                                    5: (2, 2),
                                    6: (1, 4),
                                    7: (5, 3),
                                    8: (0, 1),
                                    9: (8, 0)},
}"""

CELL_COORDINATES = {"image_0.jpg": {6: (4, 4)},
                    "image_1.jpg": {1: (6, 1),
                                    2: (0, 1),
                                    3: (1, 1),
                                    4: (2, 0),
                                    5: (3, 0),
                                    7: (4, 1),
                                    8: (1, 0), 
                                    9: (8, 0)},
                    "image_5.jpg": {1: (3, 2),
                                    2: (2, 3),
                                    3: (1, 7),
                                    4: (0, 0),
                                    5: (1, 1),
                                    6: (0, 6),
                                    7: (0, 5),
                                    8: (4, 1),
                                    9: (3, 3)},
}

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE
    pipeline = Pipeline(functions=[gaussian_blur, 
                                   find_edges, 
                                   highlight_edges, 
                                   find_contours, 
                                   get_max_contour, 
                                   find_corners, 
                                   frontalize_image,
                                   resize_image, 
                                   get_sudoku_cells],
                        parameters={"gaussian_blur": {"sigma": 1}, #0.42
                                    "find_corners": {"epsilon": 3}, #6.6
                                    "resize_image": {"size": SUDOKU_SIZE},
                                    "get_sudoku_cells": {"crop_factor":0.7, "binarization_kwargs": {}}}) #0.65
    return pipeline

    # END YOUR CODE

