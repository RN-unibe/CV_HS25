import os
import shutil

from tqdm import tqdm
from skimage.io import imsave

from const import TEMPLATES_PATH, TRAIN_IMAGES_PATH
from const import MAX_TEMPLATES_FOR_DIGIT

from utils import read_image
from template import CELL_COORDINATES, get_template_pipeline


def main():
    shutil.rmtree(TEMPLATES_PATH, ignore_errors=True)
    os.makedirs(TEMPLATES_PATH)
    
    pipeline = get_template_pipeline()
    digits_count = {1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0}

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(TRAIN_IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)
        _, sudoku_cells = pipeline(sudoku_image)

        for digit, coordinates_list in tqdm(coordinates_dict.items(), desc=file_name):
            if not isinstance(coordinates_list, list):
                coordinates_list = [coordinates_list]
            
            digit_templates_path = os.path.join(TEMPLATES_PATH, str(digit))
            os.makedirs(digit_templates_path, exist_ok=True)
            
            for i, coordinates in enumerate(coordinates_list):
                if digits_count[digit] < MAX_TEMPLATES_FOR_DIGIT:
                    digits_count[digit] += 1
                    
                    digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}_{i}.jpg")
                    imsave(digit_template_path, sudoku_cells[*coordinates])


if __name__ == "__main__":
    main()
