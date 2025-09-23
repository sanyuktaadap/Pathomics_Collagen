import numpy as np
import glob
import os
import argparse
import csv
from tqdm import tqdm
import math

patient_feats_fold_name = "otsu_patient_feats3_60_70"
cohorts = ["White", "Black"]

for cohort in cohorts:
    files = glob.glob(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/*.csv')
    # get all files that hvae '-' in thier file names
    slides = [f for f in files if '-' in f]
    count = 0
    for slide in slides:
        name = slide.split("/")[-1]
        pid = name.split("-")[0]
        file1 = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{pid}-1_H&E_Breast_XXXXXXXX.csv"
        file2 = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{pid}-2_H&E_Breast_XXXXXXXX.csv"
        feats = [file1, file2]

        # Initialize a zero vector to hold aggregated slide-level features
        file_features = np.zeros(12)
        even_counts = np.zeros(12)  # count valid values at even indices

        # Iterate over each patch feature file for the current slide
        for feat in feats:
            flag = -1  # Used to only process the first row of the CSV
            with open(feat, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if flag == -1:
                        array = row  # Assume this row is the feature vector (length 36)

                        # For odd indices (1, 3, 5, ...): take maximum value across patches
                        for index in range(1, len(array), 2):
                            val = float(array[index])
                            if not np.isnan(val):
                                # Update max
                                file_features[index] = np.nanmax([file_features[index], val])
                            # file_features[index] = np.nanmax([file_features[index], float(array[index])])

                        # # For even indices (0, 2, 4, ...): accumulate values to compute mean later
                        # for index in range(0, len(array), 2):
                        #     file_features[index] += float(array[index])

                        # Sum and count for even indices ignoring nan
                        for index in range(0, len(array), 2):
                            val = float(array[index])
                            if not np.isnan(val):
                                file_features[index] += val
                                even_counts[index] += 1

                        # Prevent processing any additional rows in the CSV
                        flag = 1

        # After processing all patches, average the even-indexed features
        for index in range(0, len(array), 2):
            if even_counts[index] > 0:
                file_features[index] /= even_counts[index]
            else:
                file_features[index] = np.nan  # or 0, if you prefer
            # file_features[index] = file_features[index] / len(feats)

        # Save the final patient-level features to a CSV file
        output_file = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{pid}_H&E_Breast_XXXXXXXX.csv"
        # output_file = f"data/hari_BC/temp_results/{pid}_H&E_Breast_XXXXXXXX.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(file_features)
        print(f"Saved feats for PID: {output_file}")
        count += 1
    print(f"-----{cohort}:{count}-----")