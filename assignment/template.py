
from pipeline import Pipeline

# BEGIN YOUR IMPORTS
import os

from const import TRAIN_IMAGES_PATH
from utils import read_image

from pipeline import Pipeline
from frontalization import find_edges, highlight_edges, find_contours, get_max_contour, find_corners, rescale_image, gaussian_blur, frontalize_image, show_frontalized_images

from const import SUDOKU_SIZE
from recognition import resize_image, get_sudoku_cells
import cv2
# END YOUR IMPORTS

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example
"""

CELL_COORDINATES = {"image_0.jpg": {1: (6, 4), 4: (3, 3), 6: (4, 4), 7: (2, 4)},
                    "image_1.jpg": {2: (0, 1)},
                    "image_2.jpg": {5: (2, 6), 7: (0, 3)},
                    "image_4.jpg": {8: (1, 5), 9: (2, 8)},
                    "image_6.jpg": {4: (2, 2), 8: (5, 1)},
                    "image_7.jpg": {1: (4, 2), 3: (3, 3), 5: (3, 5), 9: (4, 7)},
                    "image_8.jpg": {2: (5, 1), 3: (0, 5), 6: (0, 2)},
}

CELL_COORDINATES = {"image_0.jpg": {4: (3, 3), 6: (4, 4), 7: (2, 4)},
                    "image_1.jpg": {1: (1, 3), 2: (0, 1)},
                    "image_2.jpg": {5: (2, 6)},
                    "image_4.jpg": {8: (1, 5), 9: (2, 8)},
                    "image_5.jpg": {4: (0, 0), 7: (5, 3)},
                    "image_6.jpg": {8: (5, 1)},
                    "image_7.jpg": {1: (4, 2), 3: (3, 3), 5: (3, 5), 9: (4, 7)},
                    "image_8.jpg": {2: (5, 1), 3: (0, 5), 6: (0, 2)},
}

CELL_COORDINATES = {"image_0.jpg": {4: (3, 3), 6: (4, 4), 7: (2, 4)},
                    "image_1.jpg": {1: (1, 3), 2: (0, 1)},
                    "image_2.jpg": {5: (2, 6)},
                    "image_4.jpg": {8: (1, 5), 9: (2, 8)},
                    "image_5.jpg": {1: (3, 2), 4: (0, 0), 7: (5, 3)},
                    "image_6.jpg": {8: (5, 1)},
                    "image_7.jpg": {3: (3, 3), 5: (3, 5), 9: (4, 7)},
                    "image_8.jpg": {2: (5, 1), 3: (0, 5), 6: (0, 2)},
}


CELL_COORDINATES = {"image_0.jpg": {4: (3, 3), 6: (4, 4), 7: (2, 4)},
                    "image_1.jpg": {1: (1, 3), 2: (0, 1)},
                    "image_2.jpg": {5: (2, 6)},
                    "image_4.jpg": {8: (1, 5), 9: (2, 8)},
                    "image_5.jpg": {1: (3, 2), 4: (4, 3), 7: (5, 3)},
                    "image_6.jpg": {8: (5, 1)},
                    "image_7.jpg": {3: (3, 3), 5: (3, 5), 9: (4, 7)},
                    "image_8.jpg": {2: (5, 1), 3: (0, 5), 6: (0, 2)},
}

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE
  
    """pipeline = Pipeline(
        functions=[rescale_image, 
                    gaussian_blur, 
                    find_edges, 
                    highlight_edges, 
                    find_contours, 
                    get_max_contour, 
                    find_corners, 
                    frontalize_image,
                    resize_image, 
                    get_sudoku_cells],
        parameters={
            "rescale_image": {"scale": 0.7}, #min 0.9
            "gaussian_blur": {"sigma": 1.1}, #max 1.1      
            "find_corners": {"epsilon": 0.4},       
            "resize_image": {"size": SUDOKU_SIZE},   
            "get_sudoku_cells": {
                "crop_factor": 0.75,                 
                "binarization_kwargs": {}
            }
        }
    )"""

    pipeline = Pipeline(
        functions=[rescale_image, 
                    gaussian_blur, 
                    find_edges, 
                    highlight_edges, 
                    find_contours, 
                    get_max_contour, 
                    find_corners, 
                    frontalize_image,
                    resize_image, 
                    get_sudoku_cells],
        parameters={
            "rescale_image": {"scale": 0.7}, #min 0.9
            "gaussian_blur": {"sigma": 1.1}, #max 1.1
            "find_corners": {"epsilon": 0.4},       
            "resize_image": {"size": SUDOKU_SIZE},   
            "get_sudoku_cells": {
                "crop_factor": 0.75,                 
                "binarization_kwargs": {}
            }
        }
    )

    
    return pipeline

    # END YOUR CODE
