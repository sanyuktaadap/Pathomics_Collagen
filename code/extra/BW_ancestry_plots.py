import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")
patient_feats_fold_name = "otsu_patient_feats4_60_70"

num_feats = 12
donor_col = 0
year_col = 5
new_name_col = 7
cohort_col = 8
age_col = 4

# Create B & W means and max csv  + Age
df_black = {"donor": [],
            "Age": [],  # age at first timepoint
            "Stromal_Mean": [],
            "Stromal_Max": [],
            "Peritumoral_Mean": [],
            "Peritumoral_Max": []}

df_white = {"donor": [],
            "Age": [],  # age at first timepoint
            "Stromal_Mean": [],
            "Stromal_Max": [],
            "Peritumoral_Mean": [],
            "Peritumoral_Max": []}

# Drop rows with "_01.svs" in the new_name_col
process_list = process_list[~process_list.iloc[:, new_name_col].str.contains("_01.svs", na=False)]

for i in range(0, len(process_list), 2):
    donor = process_list.iloc[i, donor_col]
    print(f"Donor: {donor}")
    if process_list.iloc[i+1, donor_col] != donor:
        print(f"Error: {donor} only appears once")
        exit()

    year1 = process_list.iloc[i, year_col]
    year2 = process_list.iloc[i+1, year_col]

    # Ensure year2 > year1 and at least 2 years apart (outliers)
    if year2 - year1 < 2:
        continue

    # Age at first timepoint only (year2 can be anything)
    age1 = process_list.iloc[i, age_col]
    # Exclude older subjects (outliers)
    if age1 > 60:
        continue

    new_name_1 = process_list.iloc[i, new_name_col].replace(".svs", ".csv")
    new_name_2 = process_list.iloc[i+1, new_name_col].replace(".svs", ".csv")
    print(new_name_1, new_name_2)

    cohort = process_list.iloc[i, cohort_col]

    # Load the CSVs (no headers)
    csv_1 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_1}', header=None)
    csv_2 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_2}', header=None)

    # Extract values as 1D arrays
    data1 = csv_1.values.flatten()
    data2 = csv_2.values.flatten()

    # Even indices -> Means
    even_indices = list(range(0, num_feats, 2))
    data1_even = data1[even_indices]
    data2_even = data2[even_indices]

    # Odd indices -> Max
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
            df_black["Age"].append(age1)  # store age at timepoint 1
            df_black["Stromal_Mean"].append(stromal_even_diff)
            df_black["Peritumoral_Mean"].append(peritumoral_even_diff)
            df_black["Stromal_Max"].append(stromal_odd_diff)
            df_black["Peritumoral_Max"].append(peritumoral_odd_diff)

        elif cohort == "White":
            df_white["donor"].append(donor)
            df_white["Age"].append(age1)  # store age at timepoint 1
            df_white["Stromal_Mean"].append(stromal_even_diff)
            df_white["Peritumoral_Mean"].append(peritumoral_even_diff)
            df_white["Stromal_Max"].append(stromal_odd_diff)
            df_white["Peritumoral_Max"].append(peritumoral_odd_diff)

    else:
        print(f"Year1: {year1}, Year2: {year2} -- Year2 should be > Year1 for Donor: {donor}")
        exit()

df_black = pd.DataFrame(df_black)
df_white = pd.DataFrame(df_white)

# Save to CSV
df_black.to_csv("data/hari_BC/csv/otsu4_black_cohort_differences_60_70.csv", index=False)
df_white.to_csv("data/hari_BC/csv/otsu4_white_cohort_differences_60_70.csv", index=False)

# ---- Split by age at first timepoint (year1) ----
black_0_40   = df_black[df_black["Age"] <= 40]
black_41_60  = df_black[(df_black["Age"] >= 41) & (df_black["Age"] <= 60)]
white_0_40   = df_white[df_white["Age"] <= 40]
white_41_60  = df_white[(df_white["Age"] >= 41) & (df_white["Age"] <= 60)]

def make_boxplots(df_b, df_w, title_suffix, save_name):
    # If both cohorts empty, skip
    if len(df_b) == 0 and len(df_w) == 0:
        print(f"[WARN] No data for {title_suffix}; skipping plot.")
        return

    mean_data = [
        df_b["Stromal_Mean"].dropna(),
        df_w["Stromal_Mean"].dropna(),
        df_b["Peritumoral_Mean"].dropna(),
        df_w["Peritumoral_Mean"].dropna()
    ]
    max_data = [
        df_b["Stromal_Max"].dropna(),
        df_w["Stromal_Max"].dropna(),
        df_b["Peritumoral_Max"].dropna(),
        df_w["Peritumoral_Max"].dropna()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Means
    box1 = axes[0].boxplot(mean_data,
                           labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1),
                           showmeans=True, meanline=True,
                           meanprops=dict(color="black", linestyle="--", linewidth=1))
    mean_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]
    for patch, color in zip(box1['boxes'], mean_colors):
        patch.set_facecolor(color)
    axes[0].set_title(f"Mean Values ({title_suffix})")
    axes[0].set_ylabel("Value")
    axes[0].legend(handles=[
        plt.Line2D([0], [0], color="black", linewidth=1, label="Median"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    ], loc="upper left")

    # Max
    box2 = axes[1].boxplot(max_data,
                           labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1),
                           showmeans=True, meanline=True,
                           meanprops=dict(color="black", linestyle="--", linewidth=1))
    max_colors = ["#E88FBC", "#C42E7E", "#B69EE7", "#6C19BA"]
    for patch, color in zip(box2['boxes'], max_colors):
        patch.set_facecolor(color)
    axes[1].set_title(f"Max Values ({title_suffix})")
    axes[1].set_ylabel("Value")
    axes[1].legend(handles=[
        plt.Line2D([0], [0], color="black", linewidth=1, label="Median"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    ], loc="upper right")

    plt.suptitle(f"Black vs White Cohort: Stromal & Peritumoral ({title_suffix})")
    plt.tight_layout()
    plt.savefig(f"data/hari_BC/plots/otsu4_BvW_comparison_{save_name}.png", dpi=300)
    plt.close()

# Make the two requested plots (based only on age at timepoint 1)
make_boxplots(black_0_40,  white_0_40,  "Age ≤ 40 ", "age_0-40")
make_boxplots(black_41_60, white_41_60, "Age 41–60 ", "age_41-60")
