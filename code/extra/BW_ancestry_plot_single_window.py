import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

win_sizes = [60, 65, 70]
# repeat each element twice
win_sizes = [item for sublist in [[s, s] for s in win_sizes] for item in sublist]
# Repeat this whole list at the end of the list - stromal and peritumoral
win_sizes = win_sizes + win_sizes
print(f"win_sizes: {win_sizes}")

num_feats = 12
donor_col = 0
year_col = 4
new_name_col = 5
cohort_col = 6
age_col = 3

out_csv_dir = "data/hari_BC/csv"
out_plot_dir = "data/hari_BC/plots"
os.makedirs(out_csv_dir, exist_ok=True)
os.makedirs(out_plot_dir, exist_ok=True)

process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")
# Drop rows with "_01.svs" in the new_name_col
process_list = process_list[~process_list.iloc[:, new_name_col].str.contains("_01.svs", na=False)]

patient_feats_fold_name = "otsu_patient_feats4_60_70"


def make_boxplots(df_b, df_w, title_suffix, save_name):
    if len(df_b) == 0 and len(df_w) == 0:
        print(f"[WARN] No data for {title_suffix}; skipping plot.")
        return

    mean_data = [
        df_b["Stromal_Mean"].dropna(),
        df_w["Stromal_Mean"].dropna(),
        # df_b["Peritumoral_Mean"].dropna(),
        # df_w["Peritumoral_Mean"].dropna()
    ]
    # max_data = [
    #     df_b["Stromal_Max"].dropna(),
    #     df_w["Stromal_Max"].dropna(),
    #     # df_b["Peritumoral_Max"].dropna(),
    #     # df_w["Peritumoral_Max"].dropna()
    # ]

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


    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # Means
    # box1 = axes[0].boxplot(mean_data,
    #                        labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
    #                        patch_artist=True,
    #                        medianprops=dict(color="black", linewidth=1),
    #                        showmeans=True, meanline=True,
    #                        meanprops=dict(color="black", linestyle="--", linewidth=1))
    # mean_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]
    # for patch, color in zip(box1['boxes'], mean_colors):
    #     patch.set_facecolor(color)
    # axes[0].set_title(f"Mean Values ({title_suffix})")
    # axes[0].set_ylabel("Value")
    # axes[0].legend(handles=[
    #     plt.Line2D([0], [0], color="black", linewidth=1, label="Median"),
    #     plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    # ], loc="upper left")

    # # Max
    # box2 = axes[1].boxplot(max_data,
    #                        labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
    #                        patch_artist=True,
    #                        medianprops=dict(color="black", linewidth=1),
    #                        showmeans=True, meanline=True,
    #                        meanprops=dict(color="black", linestyle="--", linewidth=1))
    # max_colors = ["#E88FBC", "#C42E7E", "#B69EE7", "#6C19BA"]
    # for patch, color in zip(box2['boxes'], max_colors):
    #     patch.set_facecolor(color)
    # axes[1].set_title(f"Max Values ({title_suffix})")
    # axes[1].set_ylabel("Value")
    # axes[1].legend(handles=[
    #     plt.Line2D([0], [0], color="black", linewidth=1, label="Median"),
    #     plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    # ], loc="upper right")

    plt.suptitle(f"Black vs White: Stromal CF Entropy ({title_suffix})")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot: {save_name}")


for analysis_win in win_sizes:
    prefix = f"win_{analysis_win}_"
    # # find all indices of analysis_win in sizes
    # indices = [i for i, x in enumerate(win_sizes) if x == analysis_win]
    # print(f"indices for win size {analysis_win}: {indices}")

    # # Create B & W means and max csv
    # df_black = {"donor": [],
    #             "Age": [],  # age at first timepoint
    #             "Stromal_Mean": [],
    #             "Stromal_Max": [],
    #             "Peritumoral_Mean": [],
    #             "Peritumoral_Max": []}

    # df_white = {"donor": [],
    #             "Age": [],  # age at first timepoint
    #             "Stromal_Mean": [],
    #             "Stromal_Max": [],
    #             "Peritumoral_Mean": [],
    #             "Peritumoral_Max": []}

    # for i in range(0, len(process_list), 2):
    # # for i in range(0, 3, 2):
    #     donor = process_list.iloc[i, donor_col]
    #     print(f"Donor: {donor}")
    #     if process_list.iloc[i+1, donor_col] != donor:
    #         print(f"Error: {donor} only appears once")
    #         exit()

    #     year1 = process_list.iloc[i, year_col]
    #     year2 = process_list.iloc[i+1, year_col]

    #     # Ensure year2 > year1 and at least 2 years apart (outliers)
    #     if year2 - year1 < 2:
    #         continue

    #     # Age at first timepoint only (year2 can be anything)
    #     age1 = process_list.iloc[i, age_col]
    #     # Exclude older subjects (outliers)
    #     # if age1 > 60:
    #     #     continue

    #     new_name_1 = process_list.iloc[i, new_name_col].replace(".svs", ".csv")
    #     new_name_2 = process_list.iloc[i+1, new_name_col].replace(".svs", ".csv")
    #     print(new_name_1, new_name_2)

    #     cohort = process_list.iloc[i, cohort_col]

    #     # Load the CSVs (no headers)
    #     csv_1 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_1}', header=None)
    #     csv_2 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{new_name_2}', header=None)

    #     # Extract values as 1D arrays
    #     # The first 22 columns are stromal, the next 22 are peritumoral
    #     # Amongst the 22, the first 11 are means, the next 11 are max
    #     data1 = csv_1.values.flatten()
    #     data2 = csv_2.values.flatten()
    #     # print(f"data1: {data1}")
    #     # print(f"data2: {data2}")

    #     # Select only even indices (0, 2, ..., 42) - Means
    #     even_indices = [indices[0], indices[2]] #[0,22]
    #     data1_even = data1[even_indices]
    #     data2_even = data2[even_indices]
    #     # print(f"data1_even: {data1_even}, data2_even: {data2_even}")

    #     # Select only odd indices (1, 3, ..., 43) - Max
    #     odd_indices = [indices[1], indices[3]] # [1,23]
    #     data1_odd = data1[odd_indices]
    #     data2_odd = data2[odd_indices]
    #     # print(f"data1_odd: {data1_odd}, data2_odd: {data2_odd}")

    #     data1_stromal_even = np.nanmean(data1_even[0])
    #     data2_stromal_even = np.nanmean(data2_even[0])
    #     data1_peritumoral_even = np.nanmean(data1_even[1])
    #     data2_peritumoral_even = np.nanmean(data2_even[1])
    #     # print(f"data1_stromal_even: {data1_stromal_even}, data2_stromal_even: {data2_stromal_even}")
    #     # print(f"data1_peritumoral_even: {data1_peritumoral_even}, data2_peritumoral_even: {data2_peritumoral_even}")

    #     data1_stromal_odd = np.nanmax(data1_odd[0])
    #     data2_stromal_odd = np.nanmax(data2_odd[0])
    #     data1_peritumoral_odd = np.nanmax(data1_odd[1])
    #     data2_peritumoral_odd = np.nanmax(data2_odd[1])
    #     # print(f"data1_stromal_odd: {data1_stromal_odd}, data2_stromal_odd: {data2_stromal_odd}")
    #     # print(f"data1_peritumoral_odd: {data1_peritumoral_odd}, data2_peritumoral_odd: {data2_peritumoral_odd}")

    #     if year2 > year1:
    #         stromal_even_diff =  data1_stromal_even - data2_stromal_even
    #         peritumoral_even_diff =  data1_peritumoral_even - data2_peritumoral_even

    #         stromal_odd_diff =  data1_stromal_odd - data2_stromal_odd
    #         peritumoral_odd_diff =  data1_peritumoral_odd - data2_peritumoral_odd

    #         if cohort == "Black":
    #             df_black["donor"].append(donor)
    #             df_black["Age"].append(age1)  # store age at timepoint 1
    #             df_black["Stromal_Mean"].append(stromal_even_diff)
    #             df_black["Peritumoral_Mean"].append(peritumoral_even_diff)
    #             df_black["Stromal_Max"].append(stromal_odd_diff)
    #             df_black["Peritumoral_Max"].append(peritumoral_odd_diff)

    #         elif cohort == "White":
    #             df_white["donor"].append(donor)
    #             df_white["Age"].append(age1)  # store age at timepoint 1
    #             df_white["Stromal_Mean"].append(stromal_even_diff)
    #             df_white["Peritumoral_Mean"].append(peritumoral_even_diff)
    #             df_white["Stromal_Max"].append(stromal_odd_diff)
    #             df_white["Peritumoral_Max"].append(peritumoral_odd_diff)

    #     else:
    #         print(f"Year1: {year1}, Year2: {year2} -- Year2 shuld be > Year1 for Donor: {donor}")
    #         exit()


    df_black = pd.DataFrame(df_black)
    df_white = pd.DataFrame(df_white)

    # Save to CSV with column names
    df_black.to_csv(f"data/hari_BC/csv/{prefix}otsu4_black_cohort_reverse_differences.csv", index=False)
    df_white.to_csv(f"data/hari_BC/csv/{prefix}otsu4_white_cohort_reverse_differences.csv", index=False)

    # df_black = pd.read_csv(f"data/hari_BC/csv/{prefix}otsu4_black_cohort_reverse_differences.csv")
    # df_white = pd.read_csv(f"data/hari_BC/csv/{prefix}otsu4_white_cohort_reverse_differences.csv")

    # ---- Split by age at first timepoint (year1) ----
    black  = df_black[(df_black["Age"] >= 25) & (df_black["Age"] <= 55)]
    white  = df_white[(df_white["Age"] >= 25) & (df_white["Age"] <= 55)]


    # Make plots for each age group
    plot = os.path.join(out_plot_dir, f"{prefix}otsu4_BvW_age_25-55.png")
    make_boxplots(black, white, f"Age 25–55: Win_{analysis_win}", plot)