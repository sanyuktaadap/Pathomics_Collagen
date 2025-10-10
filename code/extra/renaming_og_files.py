import pandas as pd
import os
import shutil
import glob

def rename_cohort_files(dirs):
    for cohort in dirs:
        cohort_files = os.listdir(cohort)
        for cohort_file in cohort_files:
            new_name = None
            if "-" not in cohort_file:
                file_id = cohort_file.split(".")[0]
                new_name = f"{file_id}_00.svs"
            else:
                file_id = cohort_file.split("-")[0]
                num = cohort_file.split("-")[1].split(".")[0]
                if num in ("FF1", "2"):
                    new_name = f"{file_id}_01.svs"
                elif num == "1":
                    new_name = f"{file_id}_00.svs"

            if new_name:  # Only rename if rule matched
                old_path = os.path.join(cohort, cohort_file)
                new_path = os.path.join(cohort, new_name)
                os.rename(old_path, new_path)
                print(f"Old: {cohort_file}, New: {new_name}")


dirs = ["data/hari_BC/Original_renamed/Black_cohort", "data/hari_BC/Original_renamed/White_cohort"]

df = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")

new_rows = []

for d in dirs:
    files = glob.glob(os.path.join(d, "*"))
    for file in files:
        file_id = os.path.basename(file).split("_")[0]

        if file_id in df["Barcode"].values:
            if "_00" in file:
                new_name = df.loc[df["Barcode"] == file_id, "UPDATED_NAME"].values[0]
            else:
                # Get the original row
                row = df.loc[df["Barcode"] == file_id].iloc[0].copy()
                new_name = row["UPDATED_NAME"].replace("_00", "_01")

                # Update the name and store as a new row
                row["UPDATED_NAME"] = new_name
                new_rows.append(row)

            new_path = os.path.join(d, new_name)
            os.rename(file, new_path)   # uncomment to actually rename
            print(f"Renamed {file} → {new_name}")

# Append the new rows to the dataframe
# if new_rows:
#     df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# # Save the updated CSV if you want
# df.to_csv("BnW_combined_updated.csv", index=False)