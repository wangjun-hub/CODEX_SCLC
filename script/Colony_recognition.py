import numpy as np
import cv2 as cv
import os
from math import *
import matplotlib.pyplot as plt
import pickle
import argparse
import time

# Start time
start_time = time.time()

# Capture parameters
parser = argparse.ArgumentParser(description='Script for colony contour recognition')
parser.add_argument('path', type=str, help='your data path')
args = parser.parse_args()
print("First parameter:", args.path)

# Get all folders in this directory
path = args.path
os.chdir(path)

print("Set working directory to: ", path)

image_dir = [i for i in os.listdir("./") if i[3] == "_" and i[0] != "."]

print(image_dir)

for s in image_dir:
    print(s)
    # Create corresponding result folders
    result_folder1 = "../result/" + s + "/image"
    result_folder2 = "../result/" + s + "/Process_text"


    os.makedirs(result_folder1, exist_ok=True)
    os.makedirs(result_folder2, exist_ok=True)


    print("---------Read coordinate file---------")
    print(s)
    # Read file
    with open("./" + s + "/" + s + "_celltype_area.csv", "r") as in_put1:
        data_result = [i.strip().split(",") for i in in_put1]

    # Calculate the maximum and minimum coordinate ranges
    a = max(float(i[3]) for i in data_result)
    b = max(float(i[4]) for i in data_result)
    a1 = min(float(i[3]) for i in data_result)
    b1 = min(float(i[4]) for i in data_result)

    y = 100 * ceil(round(a / 100, 1)) + 200
    x = 100 * ceil(round(b / 100, 1)) + 200

    print("x-max: ", x)
    print("y-max: ", y)

    # Count cell types
    cells = [i[6] for i in data_result]
    cell_type = list(set(cells))

    # Store cell area and coordinates
    area = [[] for _ in range(len(cell_type))]
    data_points = [[] for _ in range(len(cell_type))]

    for i in data_result:
        data_point = [int(round(float(i[3]))), int(round(float(i[4])))]
        area[cell_type.index(i[6])].append(float(i[5]))
        data_points[cell_type.index(i[6])].append(data_point)

    x_max = max(max(i) for i in area) + 1

    print(x_max)

    print("---------Draw contour of the entire core---------")

    data_points_list = [[int(round(float(i[3]))), int(round(float(i[4])))] for i in data_result]

    # Draw contour of the entire core
    image = np.zeros((x, y), dtype=np.uint8)
    for data_point in data_points_list:
        cv.circle(image, tuple(data_point), 2, 255, -1)
    f = "../result/" + s + "/" + s + "_core_Whole_Core_Contour.jpg"
    print(f)
    cv.imwrite(f, image)  # Generate in the first-level directory

    print("---------Plot cell type area distribution---------")

    # Calculate the number of rows and columns of subplots
    num_plots = len(cell_type)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Create canvas and set subplot layout
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

    # Draw subplots
    for i, ax in enumerate(axes.flatten()):
        if i < num_plots:
            ax.hist(area[i], bins=range(round(min(area[i]) - 2), round(max(area[i]) + 2)), edgecolor='black')
            ax.set_xlim(right=x_max)
            ax.set_xticks(np.arange(0, x_max, int(x_max / 10) * 5))
            ax.set_title(cell_type[i])
            ax.set_xlabel('Area / µm2')
            ax.set_ylabel('Counts / cells')
        else:
            ax.axis('off')  # Hide extra subplots

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=1, hspace=1)

    f = "../result/" + s + "/" + s + '_cell_type_area.png'

    # Save the image
    plt.savefig(f, dpi=1000, bbox_inches='tight')  # Generate in the first-level directory

    print("---------Generate corresponding cell point maps---------")

    for i in range(len(cell_type)):
        image = np.zeros((x, y), dtype=np.uint8)
        r = round(sqrt(np.median(area[i]) / pi), 1)
        print(cell_type[i], " : ", r)
        for data_point in data_points[i]:
            cv.circle(image, tuple(data_point), int(r), 255, -1)
        f = "../result/" + s + "/image/" + s + '_' + cell_type[i].strip('"') + ".jpg"
        cv.imwrite(f, image)  # Generate in the second-level directory

    print("---------Read corresponding cell point maps---------")
    file_names = [i for i in os.listdir("../result/" + s + "/image/") if i[-3:] == 'jpg']

    contours_list = []

    # Define area threshold
    area_threshold = 2000

    image_cell = open('../result/' + s + '/' + s + '_image_cell.csv', 'w')

    print("---------Contour identification---------")
    for jpg_file in file_names:
        print("../result/" + s + "/image/" + jpg_file)
        image_cell.write(jpg_file)
        # Read the image
        img = cv.imread("../result/" + s + "/image/" + jpg_file)
        # Convert to grayscale
        imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Binarize
        ret, thresh = cv.threshold(imggray, 9, 255, 0, cv.THRESH_OTSU)

        # Dilate
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
        dilated = cv.dilate(thresh, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create blank image for contour drawing
        height, width, _ = img.shape
        background = np.zeros((height, width), dtype=np.uint8)

        for i in range(len(contours)):
            contour = contours[i]
            area = cv.contourArea(contour)
            if area < area_threshold:
                cv.polylines(img, [contour], True, (0, 0, 255), 3)
                moments = cv.moments(contour)
                cx = int(moments['m10'] / moments['m00']) + 20
                cy = int(moments['m01'] / moments['m00']) - 20

                # Label contour number on the image (adjust font size)
                font_scale = 1
                font_thickness = 4
                cv.putText(img, str(i), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness,
                           cv.LINE_AA)

            else:
                moments = cv.moments(contour)
                cx = int(moments['m10'] / moments['m00']) + 20
                cy = int(moments['m01'] / moments['m00']) - 20

                font_scale = 1
                font_thickness = 4
                cv.putText(img, str(i), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness,
                           cv.LINE_AA)

                cv.drawContours(img, [contour], -1, (0, 255, 0), 2)

                area = cv.contourArea(contour)

        contours_list.append(contours)

        print("---------Store recognized contour images---------")
        f = "../result/" + s + "/Process_text/" + jpg_file.split(".jpg")[0] + ".jpg"
        print(f)
        cv.imwrite(f, img)  # Generate in the second-level directory


    # Save contour list to file
    with open('../result/' + s + '/' + s + '_contours_list.pkl', 'wb') as file:
        pickle.dump(contours_list, file)

    # Save cell type order to file
    image_cell.close()

    # End time
    end_time = time.time()

    print(s + " Finished! : ", round((end_time - start_time),2), "s")











