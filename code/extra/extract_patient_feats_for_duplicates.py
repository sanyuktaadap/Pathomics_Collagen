import os
import csv
import numpy as np
import pandas as pd

# ---- config ----
patient_feats_fold_name = "otsu_patient_feats4_60_70"
cohorts = ["White", "Black"]
csv_path = f"data/hari_BC/csv/BnW_combined.csv"

df = pd.read_csv(csv_path)

total_written = 0

for cohort in cohorts:
    cohort_feat_dir = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort"
    sub = df[df["Race"] == cohort].copy()
    count = 0

    for _, row in sub.iterrows():
        updated_name = row["UPDATED_NAME"]
        if "_01.svs" in updated_name:
            file1 = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{updated_name.replace('_01.svs', '_00.csv')}"
            file2 = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{updated_name.replace('.svs', '.csv')}"
            feats = [file1, file2]

            # Initialize a zero vector to hold aggregated slide-level features
            file_features = np.zeros(12)
            even_counts = np.zeros(12)  # count valid values at even indices

            # Iterate over each patient feature file for the current slide
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

                            # Sum and count for even indices ignoring nan
                            for index in range(0, len(array), 2):
                                val = float(array[index])
                                if not np.isnan(val):
                                    file_features[index] += val
                                    even_counts[index] += 1

                            # Prevent processing any additional rows in the CSV
                            flag = 1

            # After processing all csvs, average the even-indexed features
            for index in range(0, len(array), 2):
                if even_counts[index] > 0:
                    file_features[index] /= even_counts[index]
                else:
                    file_features[index] = np.nan  # or 0, if you prefer

            # Save the final patient-level features to a CSV file
            output_file = f"data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{updated_name.replace('_01.svs', '_00.csv')}"
            os.remove(file1)
            os.remove(file2)
            with open(output_file, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(file_features)
            print(f"Saved feats for PID: {output_file}")
            count += 1
    print(f"-----{cohort}:{count}-----")