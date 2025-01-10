"""
File: Main file to extract collagen features at patient level
"""

# header files to load
import numpy as np
import glob
import os
import argparse
import csv
from tqdm import tqdm

# MAIN CODE
# read patches and masks
parser = argparse.ArgumentParser()
parser.add_argument('--input_files', help='Input slide', default='data/images/')
parser.add_argument('--input_features', help='Input patche-level features', default='results/patch_features/')
parser.add_argument('--output', help='Output patient level features', default='results/patient_features/')
args = parser.parse_args()

# loop through patients
def extract_patient_level_features(slides, patch_features):
    for slide in tqdm(slides):
        filename = slide.split("/")[-1]
        print(filename)
        filename = filename.split(".")[0] + "_"

        patch_feats = glob.glob(os.path.join(patch_features, filename+"*"))
        #if len(patches) == 0:
        #    continue

        file_features = np.zeros(36)
        for feat in patch_feats:
            flag = -1
            with open(feat, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if flag == -1:
                        array = row
                        for index in range(1, len(array), 2):
                            file_features[index] = max(file_features[index], float(array[index]))
                        for index in range(0, len(array), 2):
                            file_features[index] += float(array[index])
        for index in range(0, len(array), 2):
            file_features[index] = file_features[index] / len(patch_feats)
        with open(args.output+filename[:len(filename)-1]+".csv", mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(file_features)

if "__name__" == "__main__":
    slides = args.input_files
    patch_features = args.input_features
    slides = glob.glob(slides+"*")
    extract_patient_level_features(slides, patch_features)