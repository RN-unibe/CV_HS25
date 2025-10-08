import os
import shutil
import argparse
import numpy as np

from skimage.io import imsave

from const import TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, FRONTALIZED_IMAGES_PATH
from utils import read_image, load_templates
from template import get_template_pipeline
from recognition import recognize_digits

from sudoku_solver import matrix_to_puzzle, solve_sudoku

CEND = '\33[0m'
CBOLD = '\33[1m'

CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'


def get_recognition_error(sudoku_matrix, gt_sudoku_matrix):
    return np.sum(sudoku_matrix != gt_sudoku_matrix)


def get_recognition_error_str(recognition_error):
    if recognition_error == 0:
        return CBOLD + CGREEN + f"{recognition_error}" + CEND
    elif recognition_error <= 3:
        return CBOLD + CYELLOW + f"{recognition_error}" + CEND
    else:
        return CBOLD + CRED + f"{recognition_error}" + CEND


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    path = TEST_IMAGES_PATH if args.test else TRAIN_IMAGES_PATH
    
    image_paths = [os.path.join(path, file_name) for file_name in sorted(os.listdir(path))
                   if 'jpg' in os.path.splitext(file_name)[1]]
    sudoku_matrix_paths = [os.path.join(path, file_name) for file_name in sorted(os.listdir(path))
                           if 'npy' in os.path.splitext(file_name)[1]]
    shutil.rmtree(os.path.join(FRONTALIZED_IMAGES_PATH, f"{os.path.split(os.path.split(image_paths[0])[0])[1]}"),
                  ignore_errors=True)
    
    pipeline = get_template_pipeline()
    recognition_errors = []
    for image_path, sudoku_matrix_path in zip(image_paths, sudoku_matrix_paths):
        print("-"*20)
        print(f"For Sudoku in the image {image_path}")
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, sudoku_cells = pipeline(sudoku_image)

        frontalized_image_path = os.path.join(FRONTALIZED_IMAGES_PATH,
                                              f"{os.path.split(os.path.split(image_path)[0])[1]}",
                                              f"frontalized_{os.path.split(image_path)[1]}")
        os.makedirs(os.path.split(frontalized_image_path)[0], exist_ok=True)
        imsave(frontalized_image_path, frontalized_image)
        print(f"You can find the frontalized image at {frontalized_image_path}")
        print("-"*20)

        templates_dict = load_templates()
        sudoku_matrix = recognize_digits(sudoku_cells, templates_dict)

        gt_sudoku_matrix = np.load(sudoku_matrix_path)

        print(f"Your Sudoku matrix is")
        print(matrix_to_puzzle(sudoku_matrix))
        print()
        print("Ground truth Sudoku matrix is")
        print(matrix_to_puzzle(gt_sudoku_matrix))
        print()

        recognition_error = get_recognition_error(sudoku_matrix, gt_sudoku_matrix)
        recognition_errors.append(recognition_error)
        print(f"There are {get_recognition_error_str(recognition_error)} cells with recognition error")
        print("-"*20)

        if recognition_error == 0:
            sudoku_matrix_solved = solve_sudoku(sudoku_matrix)
            print("The solved Sudoku puzzle is")
            print(matrix_to_puzzle(sudoku_matrix_solved))
        else:
            print("The recognized Sudoku matrix contains errors and cannot be solved")
        print()

    print(f"Successfully recognized {sum(np.array(recognition_errors) <= 3)} out of {len(recognition_errors)} Sudoku grids")


if __name__ == "__main__":
    main()
