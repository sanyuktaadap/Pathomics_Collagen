import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create B & W means and max csv
df_black = {"donor": [],
            "Stromal_Mean": [],
            "Stromal_Max": [],
            "Peritumoral_Mean": [],
            "Peritumoral_Max": []}

df_white = {"donor": [],
            "Stromal_Mean": [],
            "Stromal_Max": [],
            "Peritumoral_Mean": [],
            "Peritumoral_Max": []}

process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")
num_feats = 12
donor_col = 2
year_col = 4
new_name_col = 5
cohort_col = 6

for i in range(0, len(process_list), 2):
    donor = process_list.iloc[i, donor_col]
    print(f"Donor: {donor}")
    if process_list.iloc[i+1, donor_col] != donor:
        print(f"Error: {donor} only appears once")
        exit()

    year1 = process_list.iloc[i, year_col]
    year2 = process_list.iloc[i+1, year_col]

    p1 = process_list.iloc[i, new_name_col].split("/")[-1]
    p2 = process_list.iloc[i+1, new_name_col].split("/")[-1]

    p1 = p1.split("_")[0]
    p2 = p2.split("_")[0]

    cohort = process_list.iloc[i, cohort_col]

    # Load the CSVs (no headers)
    csv_1 = pd.read_csv(f'data/hari_BC/otsu/otsu_patient_feats2/{cohort}_cohort/{p1}_H&E_Breast_XXXXXXXX.csv', header=None)
    csv_2 = pd.read_csv(f'data/hari_BC/otsu/otsu_patient_feats2/{cohort}_cohort/{p2}_H&E_Breast_XXXXXXXX.csv', header=None)

    # Extract values as 1D arrays
    data1 = csv_1.values.flatten()
    data2 = csv_2.values.flatten()

    # Select only even indices (0, 2, ..., 42) - Means
    even_indices = list(range(0, num_feats, 2))
    data1_even = data1[even_indices]
    data2_even = data2[even_indices]

    # Select only odd indices (1, 3, ..., 43) - Max
    odd_indices = list(range(1, num_feats, 2))
    data1_odd = data1[odd_indices]
    data2_odd = data2[odd_indices]

    data1_stromal_even = np.nanmean(data1_even[:num_feats//4])
    data2_stromal_even = np.nanmean(data2_even[:num_feats//4])
    data1_peritumoral_even = np.nanmean(data1_even[num_feats//4:])
    data2_peritumoral_even = np.nanmean(data2_even[num_feats//4:])

    data1_stromal_odd = np.nanmax(data1_odd[:num_feats//4])
    data2_stromal_odd = np.nanmax(data2_odd[:num_feats//4])
    data1_peritumoral_odd = np.nanmax(data1_odd[num_feats//4:])
    data2_peritumoral_odd = np.nanmax(data2_odd[num_feats//4:])

    if year2 > year1:
        stromal_even_diff =  data2_stromal_even - data1_stromal_even
        peritumoral_even_diff =  data2_peritumoral_even - data1_peritumoral_even

        stromal_odd_diff =  data2_stromal_odd - data1_stromal_odd
        peritumoral_odd_diff =  data2_peritumoral_odd - data1_peritumoral_odd

        if cohort == "Black":
            df_black["donor"].append(donor)
            df_black["Stromal_Mean"].append(stromal_even_diff)
            df_black["Peritumoral_Mean"].append(peritumoral_even_diff)
            df_black["Stromal_Max"].append(stromal_odd_diff)
            df_black["Peritumoral_Max"].append(peritumoral_odd_diff)

        elif cohort == "White":
            df_white["donor"].append(donor)
            df_white["Stromal_Mean"].append(stromal_even_diff)
            df_white["Peritumoral_Mean"].append(peritumoral_even_diff)
            df_white["Stromal_Max"].append(stromal_odd_diff)
            df_white["Peritumoral_Max"].append(peritumoral_odd_diff)

    else:
        print(f"Year1: {year1}, Year2: {year2} -- Year2 shuld be > Year1 for Donor: {donor}")
        exit()


df_black = pd.DataFrame(df_black)
df_white = pd.DataFrame(df_white)

# Save to CSV with column names
df_black.to_csv("data/hari_BC/csv/otsu2_black_cohort_differences.csv", index=False)
df_white.to_csv("data/hari_BC/csv/otsu2_white_cohort_differences.csv", index=False)

# 1. Read the CSV
# 1. Read the CSVs
# df_black = pd.read_csv("black_cohort_differences.csv")
# df_white = pd.read_csv("white_cohort_differences.csv")

# 2. Column groups
mean_cols = ["Stromal_Mean", "Peritumoral_Mean"]
max_cols = ["Stromal_Max", "Peritumoral_Max"]

# 3. Prepare the data
mean_data = [
    df_black["Stromal_Mean"].dropna(),
    df_white["Stromal_Mean"].dropna(),
    df_black["Peritumoral_Mean"].dropna(),
    df_white["Peritumoral_Mean"].dropna()
]

max_data = [
    df_black["Stromal_Max"].dropna(),
    df_white["Stromal_Max"].dropna(),
    df_black["Peritumoral_Max"].dropna(),
    df_white["Peritumoral_Max"].dropna()
]

# 4. Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- Means plot ---
box1 = axes[0].boxplot(mean_data,
                       labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
                       patch_artist=True)

# Colors for means
mean_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]  # pinks & violets
for patch, color in zip(box1['boxes'], mean_colors):
    patch.set_facecolor(color)

axes[0].set_title("Mean Values")
axes[0].set_ylabel("Value")

# --- Max plot ---
box2 = axes[1].boxplot(max_data,
                       labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
                       patch_artist=True)

# Colors for max
max_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]
for patch, color in zip(box2['boxes'], max_colors):
    patch.set_facecolor(color)

axes[1].set_title("Max Values")
axes[1].set_ylabel("Value")

plt.suptitle("Black vs White Cohort: Stromal & Peritumoral Boxplots")
plt.tight_layout()
plt.savefig("otsu2_BvW_comparison.png", dpi=300)