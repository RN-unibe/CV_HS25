import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from const import TEMPLATES_PATH


def rgb2grayscale(image):
    """
    Args:
        image (np.array): RGB image of shape [H, W, 3] (np.array of np.uint8 type)
    Returns:
        image (np.array): grayscale image of shape [H, W]
    """
    image = np.dot(image, [0.299, 0.587, 0.114]).astype(dtype=np.uint8)
    
    return np.clip(image, 0, 255)


def grayscale2rgb(image):
    """
    Args:
        image (np.array): grayscale image of shape [H, W]
    Returns:
        image (np.array): RGB image of shape [H, W, 3]
    """
    return np.stack((image,)*3, axis=-1)


def read_image(image_path):
    """
    Args:
        image_path (str): path to the image
    Returns:
        image (np.array): image of shape [H, W] (grayscale image)
    """
    image = plt.imread(image_path)

    if len(image.shape) == 3:
        image = rgb2grayscale(image)

    return image


def show_image(image, axis=None, as_gray=False):
    if axis is None:
        axis = plt
    if as_gray:
        axis.imshow(image, 'gray')
    else:
        axis.imshow(image)

    axis.axis('off')


def show_contours(image, contours, axis=None, contour_color=(255, 0, 0)):
    image_with_contours = cv2.drawContours(grayscale2rgb(image.copy()), contours, -1, contour_color, 10)
    show_image(image_with_contours, axis=axis, as_gray=False)


def show_corners(image, corners, axis=None, color='r'):
    show_image(image, axis=axis, as_gray=True)

    if axis is None:
        axis = plt

    axis.scatter(x=[point[0] for point in corners],
                 y=[point[1] for point in corners],
                 c=color, marker="x", s=80)

    axis.plot([point[0] for point in corners] + [corners[0][0]],
              [point[1] for point in corners] + [corners[0][1]],
              c=color)


def show_sudoku_cells(sudoku_cells, axis=None):
    num_cells = sudoku_cells.shape[0]
    size = sudoku_cells.shape[2]

    show_image(sudoku_cells.transpose(0, 2, 1, 3).reshape(num_cells * size, num_cells * size),
               axis=axis, as_gray=True)


def load_templates():
    """
    Returns:
        templates (dict): dict with digits as keys and lists of template images (np.array) as values
    """
    templates = {}
    for folder_name in sorted(os.listdir(TEMPLATES_PATH)):
        if "." in folder_name:
            continue
        
        folder_path = os.path.join(TEMPLATES_PATH, folder_name)
        templates[int(folder_name)] = [read_image(os.path.join(folder_path, file_name))
                                       for file_name in sorted(os.listdir(folder_path))
                                       if os.path.isfile(os.path.join(folder_path, file_name))]
    
    return templates
