"""
Authors: Arpit Aggarwal
File: Main file to extract collagen features at patient level
"""

# header files to load
import numpy as np
import cv2
from PIL import Image
import math
import os
import glob
import argparse
import csv


# MAIN CODE
# read patches and masks
parser = argparse.ArgumentParser()
parser.add_argument('--input_files', help='Input patches', default='data/files/')
parser.add_argument('--input_features', help='Input masks', default='results/patches/')
parser.add_argument('--output', help='Output', default='results/features/')
args = parser.parse_args()
files = args.input_files
features = args.input_features
files = glob.glob(files+"*") 

# loop through patients
for file in files:
    filename = file.split("/")[-1][:-4] + "_"
    print(filename)

    patches = glob.glob(features+filename+"*")
    #if len(patches) == 0:
    #    continue
    
    file_features = np.zeros(36)
    for patch in patches:
        flag = -1
        with open(patch, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if flag == -1:
                    array = row
                    for index in range(1, len(array), 2):
                        file_features[index] = max(file_features[index], float(array[index]))
                    for index in range(0, len(array), 2):
                        file_features[index] += float(array[index])
    for index in range(0, len(array), 2):
        file_features[index] = file_features[index] / len(patches)
    with open(args.output+filename[:len(filename)-1]+".csv", mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(file_features)
