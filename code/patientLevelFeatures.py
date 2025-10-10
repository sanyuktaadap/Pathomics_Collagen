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
def extract_patient_level_features(slides, patch_features, output):
    # Iterate over each slide in the list
    for slide in tqdm(slides):
        # Extract the slide filename (without path or extension)
        filename = slide.split("/")[-1]
        filename = filename.rsplit('.', 1)[0]  # Removes file extension
        print(filename)

        # Find all patch-level feature files for this slide (prefix match)
        patch_feats = glob.glob(os.path.join(patch_features, filename + "*"))

        # Initialize a zero vector to hold aggregated slide-level features
        file_features = np.zeros(12)

        # Iterate over each patch feature file for the current slide
        for feat in patch_feats:
            flag = -1  # Used to only process the first row of the CSV
            with open(feat, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if flag == -1:
                        array = row  # Assume this row is the feature vector (length 36)

                        # For odd indices (1, 3, 5, ...): take maximum value across patches
                        for index in range(1, len(array), 2):
                            file_features[index] = max(file_features[index], float(array[index]))

                        # For even indices (0, 2, 4, ...): accumulate values to compute mean later
                        for index in range(0, len(array), 2):
                            file_features[index] += float(array[index])

                        # Prevent processing any additional rows in the CSV
                        flag = 1

        # After processing all patches, average the even-indexed features
        for index in range(0, len(array), 2):
            file_features[index] = file_features[index] / len(patch_feats)

        # Save the final patient-level features to a CSV file
        output_file = os.path.join(output, filename + ".csv")
        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(file_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', help='Input slide', default='data/hari_BC/Original_renamed')
    parser.add_argument('--input_features', help='Input patche-level features', default='data/hari_BC/otsu/otsu_patch_feats4_60_70')
    parser.add_argument('--output', help='Output patient level features', default='data/hari_BC/otsu/otsu_patient_feats4_60_70/')
    args = parser.parse_args()

    cohorts = ['Black_cohort', 'White_cohort']
    for cohort in cohorts:
        slides = os.path.join(args.input_files, cohort)
        print(slides)
        patch_features = os.path.join(args.input_features, cohort)

        output = os.path.join(args.output, cohort)
        slides = glob.glob(os.path.join(slides, "*"))
        os.makedirs(output, exist_ok=True)
        extract_patient_level_features(slides, patch_features, output)