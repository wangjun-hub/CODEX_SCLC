import numpy as np
import cv2 as cv
import os
import pickle
import argparse
import time
from math import *

# Start time
start_time = time.time()

# Capture parameters
parser = argparse.ArgumentParser(description='Script for colony contour visualization')
parser.add_argument('path', type=str, help='your data path')
args = parser.parse_args()
print("First parameter:", args.path)

# Get all folders in this directory
path = args.path
os.chdir(path)

print("Working directory set to:", path)

# Get image directories
image_dir = [i for i in os.listdir("./") if i[3] == "_" and i[0] != "."]

for s in image_dir:
    print(s)

    # Read cell type names
    image_cell=open(f'../result/{s}/{s}_image_cell.csv',"r")
    image_cell_list=[]

    for i in image_cell:
        orig=i.strip().split(".jpg")
        image_cell_list=orig[:-1]



    # Read contour files
    with open(f'../result/{s}/{s}_contours_list.pkl', 'rb') as file:
        contours_list = pickle.load(file)

    # Filter desired cell type combinations
    cell_type1 = 'Macrophage'
    file_names_group_1 = [i for i in image_cell_list if cell_type1 in i]

    print("---------Read coordinate files---------")
    print(s)

    # Read file
    with open(f"./{s}/{s}_celltype_area.csv", "r") as in_put1:
        data_result = [line.strip().split(",") for line in in_put1]

    # Find max and min coordinate ranges
    a = max(float(i[3]) for i in data_result)
    a1 = min(float(i[3]) for i in data_result)
    b = max(float(i[4]) for i in data_result)
    b1 = min(float(i[4]) for i in data_result)
    # Calculate the maximum and minimum coordinate ranges


    y_max = 100 * ceil(round(a / 100, 1)) + 200
    x_max = 100 * ceil(round(b / 100, 1)) + 200

    print("x-max: ", x_max)
    print("y-max: ", y_max)

    # Get cell types
    cells = [i[6] for i in data_result]
    cell_type = list(set(cells))

    # Calculate cell areas and coordinates
    area = [[] for _ in range(len(set(i[6] for i in data_result)))]
    data_points = [[] for _ in range(len(area))]

    for i in data_result:
        data_point = [int(round(float(i[3]))), int(round(float(i[4])))]
        area[cell_type.index(i[6])].append(float(i[5]))
        data_points[cell_type.index(i[6])].append(data_point)

    print("---------Generate corresponding cell point images---------")

    # Create white image
    image = np.ones((x_max, y_max), np.uint8) * 255

    f = f"../result/{s}/Process_text/{s}_Macrophage_show.jpg"
    cv.imwrite(f, image)  # Save image
    img = cv.imread(f)

    contours_1 = contours_list[image_cell_list.index(file_names_group_1[0])]
    cv.drawContours(img, contours_1, -1, (0, 0, 128), 10)

    cell_type_new = [i.strip('"') for i in cell_type]

    for i in range(len(cell_type)):
        r = round(np.sqrt(np.median(area[i]) / np.pi), 1)
        print(cell_type[i], ":", r)
        for data_point in data_points[i]:
            cv.circle(img, tuple(data_point), int(r), (193, 192, 192), -1)

    # Overlay Macrophage cells
    if file_names_group_1:
        i = cell_type_new.index(cell_type1)
        r = round(np.sqrt(np.median(area[i]) / np.pi), 1)
        for data_point in data_points[i]:
            cv.circle(img, tuple(data_point), int(r), (0, 0, 128), -1)

    print("---------Save recognized contour images---------")
    f = f"../result/{s}/Process_text/{s}_Macrophage_show.jpg"
    cv.imwrite(f, img)

    print(file_names_group_1)

    # End time
    end_time = time.time()
    print(s + " Finished!: ", round((end_time - start_time), 2), " s")
