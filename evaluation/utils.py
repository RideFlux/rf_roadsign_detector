import os
import numpy as np

def create_hyperlink(file_path, text):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_path)
    file_uri = f'file://{file_path}'
    print(f'\n\033]8;;{file_uri}\033\\{text} (ctrl + click)\033]8;;\033\\')

def compute_distance(box1, box2):
    dist = np.linalg.norm((box1[:2]+box1[3:5]/2) - (box2[:2]+box2[3:5]/2))
    return dist