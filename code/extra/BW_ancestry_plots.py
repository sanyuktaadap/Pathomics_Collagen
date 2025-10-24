import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def make_boxplots(df_b, df_w, title_suffix, save_name):
    # If both cohorts empty, skip
    if len(df_b) == 0 and len(df_w) == 0:
        print(f"[WARN] No data for {title_suffix}; skipping plot.")
        return

    mean_data = [
        df_b["Stromal_Mean"].dropna(),
        df_w["Stromal_Mean"].dropna(),
    ]

    plt.figure(figsize=(5,7))

    # Create the boxplot directly on the current Axes
    box = plt.boxplot(
        mean_data,
        labels=["Stromal:Black", "Stromal:White"],
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color="black", linewidth=1),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="black", linestyle="--", linewidth=1)
    )

    # Fill box colors
    mean_colors = ["#FA98C9", "#C31170"]
    for patch, color in zip(box["boxes"], mean_colors):
        patch.set_facecolor(color)

    # Add horizontal zero line
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    # Titles, labels, legend
    plt.margins(x=0.02)
    plt.title(f"Mean Entropy")
    plt.ylabel("Entropy")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="black", linewidth=1, label="Median"),
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
        ],
        loc="upper left"
    )

    plt.suptitle(f"Black vs White: Stromal CF Entropy ({title_suffix})")
    plt.tight_layout()
    plt.savefig(f"data/hari_BC/otsu4_BvW_comparison_{save_name}.png", dpi=300)
    plt.close()


process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")
patient_feats_fold_name = "otsu_patient_feats4_60_70"

num_feats = 12
donor_col = 0
year_col = 4
new_name_col = 5
cohort_col = 6
age_col = 3

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
        stromal_even_diff =  data1_stromal_even - data2_stromal_even
        peritumoral_even_diff =  data1_peritumoral_even - data2_peritumoral_even

        stromal_odd_diff =  data1_stromal_odd - data2_stromal_odd
        peritumoral_odd_diff =  data1_peritumoral_odd - data2_peritumoral_odd

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
df_black.to_csv("data/hari_BC/otsu4_black_cohort_differences_60_70_reverse_diff_all_subj.csv", index=False)
df_white.to_csv("data/hari_BC/otsu4_white_cohort_differences_60_70_reverse_diff_all_subj.csv", index=False)

# df_black = pd.read_csv("data/hari_BC/csv/otsu4_black_cohort_differences_60_70_reverse_diff_all_subj.csv")
# df_white = pd.read_csv("data/hari_BC/csv/otsu4_white_cohort_differences_60_70_reverse_diff_all_subj.csv")

# ---- Split by age at first timepoint (year1) ----
black  = df_black[(df_black["Age"] >= 40) & (df_black["Age"] <= 60)]
white  = df_white[(df_white["Age"] >= 40) & (df_white["Age"] <= 60)]

# Make the two requested plots (based only on age at timepoint 1)
# make_boxplots(black_0_40,  white_0_40,  "Age ≤ 40 ", "age_0-40")
# make_boxplots(black_41_60, white_41_60, "Age 41–60 ", "age_41-60")
# make_boxplots(black_25_55, white_25_55, "Age 25_55 ", "age_25_55")
# make_boxplots(df_black, df_white, "all_subjects", "all_subjects")
make_boxplots(black, white, "Age 40–60", "age_40-60")