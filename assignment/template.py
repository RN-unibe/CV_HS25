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

"""
CELL_COORDINATES = {"image_0.jpg": {1: (6, 4),
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
"""
CELL_COORDINATES = {"image_0.jpg": {6: (4, 4)},
                    "image_1.jpg": {1: (6, 1),
                                    2: (0, 1),
                                    3: (1, 1),
                                    4: (2, 0),
                                    5: (3, 0),
                                    7: (4, 1),
                                    8: (1, 0), 
                                    9: (4, 3)},
                    "image_5.jpg": {1: (3, 2),
                                    2: (2, 3),
                                    3: (1, 7),
                                    4: (4, 3),
                                    5: (3, 0),
                                    6: (5, 0),
                                    7: (5, 3),
                                    8: (4, 1),
                                    9: (3, 3)},
}"""
"""CELL_COORDINATES = {"image_0.jpg": {6: (4, 4)},
                    "image_1.jpg": {1: (6, 1),
                                    2: (0, 1),
                                    3: (1, 1),
                                    4: (2, 0),
                                    5: (3, 0),
                                    7: (4, 1),
                                    8: (1, 0), 
                                    9: (4, 3)},
                    "image_5.jpg": {
                                    2: (2, 3),},
}"""


CELL_COORDINATES = {"image_0.jpg": {6: (4, 4)},
                    "image_1.jpg": {1: (6, 1),
                                    2: (0, 1),
                                    3: (1, 1),
                                    4: (2, 0),
                                    5: (3, 0),
                                    7: (4, 1),
                                    8: (1, 0), 
                                    9: (4, 3)},
                    "image_1.jpg": {1: (6, 1)},
                    "image_5.jpg": {1: (3, 2),
                                    2: (2, 3),
                                    3: (1, 7),
                                    4: (4, 3),
                                    5: (3, 0),
                                    7: (5, 3),
                                    8: (4, 1),
                                    9: (3, 3)},
                    "image_6.jpg": {6: (6, 4)},
}

CELL_COORDINATES = {"image_0.jpg": {1: (6, 4), 6: (4, 4), 7: (2, 4)},
                    "image_2.jpg": {5: (2, 6), 7: (0, 3)},
                    "image_4.jpg": {8: (1, 5), 9: (2, 8)},
                    "image_6.jpg": {4: (2, 2), 8: (5, 1)},
                    "image_7.jpg": {1: (4, 2), 3: (3, 3), 5: (3, 5), 9: (4, 7)},
                    "image_8.jpg": {2: (5, 1), 3: (0, 5), 6: (0, 2)},
                    "image_9.jpg": {4: (4, 7)}
}





# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE
    pipeline = Pipeline(functions=[rescale_image, 
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
                                    "gaussian_blur": {"sigma": 0.9}, # play with the "sigma" parameter
                                    "find_corners": {"epsilon": 0.9}, # play with the "epsilon" parameter
                                    "resize_image": {"size": SUDOKU_SIZE},
                                    # play with the "crop_factor" parameter and binarization_kwargs
                                    "get_sudoku_cells": {"crop_factor":0.75, "binarization_kwargs": {}}
                                })
    

    
    return pipeline

    # END YOUR CODE

