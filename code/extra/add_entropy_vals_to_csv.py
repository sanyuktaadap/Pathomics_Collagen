import pandas as pd
import numpy as np

num_feats = 12
donor_col = 0
year_col = 4
new_name_col = 5
cohort_col = 6
age_col = 3

patient_feats_fold_name = "otsu_patient_feats4_60_70"

# Load original CSV
process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")
process_list = process_list[~process_list.iloc[:, new_name_col].str.contains("_01.svs", na=False)]
process_list = process_list.reset_index(drop=True)   # <- add this
print(f"Total entries after filtering: {len(process_list)}")

# Initialize new columns
process_list["Stromal_Mean"] = np.nan
process_list["Peritumoral_Mean"] = np.nan

i = 0
while i < len(process_list):
    donor = process_list.iloc[i, donor_col]
    if i+1 >= len(process_list) or process_list.iloc[i+1, donor_col] != donor:
        print(f"Warning: {donor} only appears once or missing pair")
        i += 1
        continue

    year1 = process_list.iloc[i, year_col]
    year2 = process_list.iloc[i+1, year_col]

    # Age at first timepoint only (year2 can be anything)
    age1 = process_list.iloc[i, age_col]

    new_name_1 = process_list.iloc[i, new_name_col].replace(".svs", ".csv")
    new_name_2 = process_list.iloc[i+1, new_name_col].replace(".svs", ".csv")
    print(new_name_1, new_name_2)

    cohort = process_list.iloc[i, cohort_col]

    # Load the CSVs (no headers)
    csv_1 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_1}', header=None)
    csv_2 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_2}', header=None)

    # Flatten and select even indices
    even_indices = list(range(0, num_feats, 2))
    data1_even = csv_1.values.flatten()[even_indices]
    data2_even = csv_2.values.flatten()[even_indices]

    # Compute separate means for data1 and data2
    quarter = num_feats // 4  # used to split stromal vs peritumoral in the even list
    stromal_mean_1 = np.nanmean(data1_even[:quarter])
    stromal_mean_2 = np.nanmean(data2_even[:quarter])
    peritumoral_mean_1 = np.nanmean(data1_even[quarter:])
    peritumoral_mean_2 = np.nanmean(data2_even[quarter:])

    print(f"Stromal_Mean (data1): {stromal_mean_1}, Peritumoral_Mean (data1): {peritumoral_mean_1}")
    print(f"Stromal_Mean (data2): {stromal_mean_2}, Peritumoral_Mean (data2): {peritumoral_mean_2}")

    # Assign to rows: data1 -> row i, data2 -> row i+1
    process_list.loc[i, "Stromal_Mean"] = stromal_mean_1
    process_list.loc[i, "Peritumoral_Mean"] = peritumoral_mean_1
    process_list.loc[i+1, "Stromal_Mean"] = stromal_mean_2
    process_list.loc[i+1, "Peritumoral_Mean"] = peritumoral_mean_2

    i += 2  # Move to next donor pair
    print("----")

# Save updated CSV
process_list.to_csv("data/hari_BC/csv/BnW_combined_with_means.csv", index=False)
print("CSV updated with Stromal_Mean and Peritumoral_Mean!")
