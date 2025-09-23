import pandas as pd
import numpy as np

# Load original CSV
df = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")

num_feats = 12
donor_col = 2
year_col = 4
new_name_col = 5
cohort_col = 6

# Initialize new columns
df["Stromal_Mean"] = np.nan
df["Peritumoral_Mean"] = np.nan

i = 0
while i < len(df):
    donor = df.iloc[i, donor_col]
    if i+1 >= len(df) or df.iloc[i+1, donor_col] != donor:
        print(f"Warning: {donor} only appears once or missing pair")
        i += 1
        continue

    cohort = df.iloc[i, cohort_col]

    # Extract patient names
    p1 = df.iloc[i, new_name_col].split("/")[-1].split("_")[0]
    p2 = df.iloc[i+1, new_name_col].split("/")[-1].split("_")[0]

    # Load the CSVs (no headers)
    csv_1 = pd.read_csv(f'data/hari_BC/otsu/otsu_patient_feats2_60_70/{cohort}_cohort/{p1}_H&E_Breast_XXXXXXXX.csv', header=None)
    csv_2 = pd.read_csv(f'data/hari_BC/otsu/otsu_patient_feats2_60_70/{cohort}_cohort/{p2}_H&E_Breast_XXXXXXXX.csv', header=None)

    # Flatten and select even indices
    even_indices = list(range(0, num_feats, 2))
    data1_even = csv_1.values.flatten()[even_indices]
    data2_even = csv_2.values.flatten()[even_indices]

    # Compute means
    stromal_mean = np.nanmean([np.nanmean(data1_even[:num_feats//4]), np.nanmean(data2_even[:num_feats//4])])
    peritumoral_mean = np.nanmean([np.nanmean(data1_even[num_feats//4:]), np.nanmean(data2_even[num_feats//4:])])

    # Assign to both rows for the donor
    df.loc[i, "Stromal_Mean"] = stromal_mean
    df.loc[i+1, "Stromal_Mean"] = stromal_mean
    df.loc[i, "Peritumoral_Mean"] = peritumoral_mean
    df.loc[i+1, "Peritumoral_Mean"] = peritumoral_mean

    i += 2  # Move to next donor pair

# Save updated CSV
df.to_csv("data/hari_BC/csv/BnW_combined_with_means.csv", index=False)
print("CSV updated with Stromal_Mean and Peritumoral_Mean!")
