import pandas as pd
import scipy.signal as sig
import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt
from skimage import io
import skimage
import os
import argparse

parser = argparse.ArgumentParser()

# HYPERPARAMETERS DEFAULT
root = ''
output_path = 'results.csv'
kernel = (21, 21)
sigma = 15.0
pad = 50
extra = 100
gamma = 1.75

# PARSER

parser.add_argument('--root', '-r', default=root)
parser.add_argument('--output', '-o', default=output_path)
parser.add_argument('--kernel', '-k', default=kernel)
parser.add_argument('--sigma', '-s', default=sigma)
parser.add_argument('--xypadding', '-xy', default=pad)
parser.add_argument('--hwpadding', '-hw', default=extra)
parser.add_argument('--gamma', '-g', default=gamma)

args = parser.parse_args()


# Initialize Empty DataFrame

result = pd.DataFrame(columns=['path', 'x', 'y', 'w', 'h'])

# Functions
def bounding_box(im, contours):
    cv2.drawContours(im, contours, -1, 255, 3)

    # find the biggest countour (c) by the area

    # print(len(contours))
    temp = contours
    c = max(temp, key = cv2.contourArea)


    # Naively find the second largest bounding box
    c = np.inf
    counter = 0
    cont = {}
    for i in range(len(contours)):
        cont[i] = cv2.contourArea(contours[i])

    sorted_values = sorted(cont.values()) # Sort the values
    sorted_dict = {}

    for i in sorted_values:
        for k in cont.keys():
            if cont[k] == i:
                sorted_dict[k] = cont[k]
                break


    # Pull out second largest contour; form a bounding box
    c = list(sorted_dict.keys())[-2]

    x,y,w,h = cv2.boundingRect(contours[c])
    return x, y, w, h 


if __name__ == "__main__":
    working_directory = [x for x in os.listdir() if 'HAM10000_images' in x]
    
    for directory in working_directory:
        for image_path in os.listdir(directory):
            path = os.path.join(directory, image_path)
            
            ##########################
            #START FIXME
            ##########################
            im = cv2.imread(path, 1)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(imgray, args.kernel, args.sigma)
            high_contrast = skimage.exposure.adjust_log(blurred, args.gamma)
            ret, thresh = cv2.threshold(high_contrast, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = bounding_box(high_contrast, contours)
            x = int(x - args.xypadding)
            y = int(y - args.xypadding)
            w = int(w + args.hwpadding)
            h = int(h + args.hwpadding)
            result.append(pd.DataFrame([[path, x, y, w, h]], columns=result.columns),ignore_index=True)
            ##########################
            # END FIXME
            ##########################

    result.to_csv(args.output)