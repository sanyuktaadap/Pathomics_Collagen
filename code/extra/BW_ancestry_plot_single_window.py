import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sizes = [60, 65, 70]
# repeat each element twice
sizes = [item for sublist in [[s, s] for s in sizes] for item in sublist]
# Repeat this whole list at the end of the list - stromal and peritumoral
sizes = sizes + sizes

process_list = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")

donor_col = 0
year_col = 5
new_name_col = 6
cohort_col = 7

patient_feats_fold_name = "otsu_patient_feats3_60_70"

for analysis_win in sizes:
    prefix = f"win_{analysis_win}_"
    # find all indices of analysis_win in sizes
    indices = [i for i, x in enumerate(sizes) if x == analysis_win]

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



    for i in range(0, len(process_list), 2):
    # for i in range(0, 3, 2):
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
        csv_1 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{p1}_H&E_Breast_XXXXXXXX.csv', header=None)
        csv_2 = pd.read_csv(f'data/hari_BC/otsu/{patient_feats_fold_name}/{cohort}_cohort/{p2}_H&E_Breast_XXXXXXXX.csv', header=None)

        # Extract values as 1D arrays
        # The first 22 columns are stromal, the next 22 are peritumoral
        # Amongst the 22, the first 11 are means, the next 11 are max
        data1 = csv_1.values.flatten()
        data2 = csv_2.values.flatten()
        # print(f"data1: {data1}")
        # print(f"data2: {data2}")

        # Select only even indices (0, 2, ..., 42) - Means
        even_indices = [indices[0], indices[2]] #[0,22]
        data1_even = data1[even_indices]
        data2_even = data2[even_indices]
        # print(f"data1_even: {data1_even}, data2_even: {data2_even}")

        # Select only odd indices (1, 3, ..., 43) - Max
        odd_indices = [indices[1], indices[3]] # [1,23]
        data1_odd = data1[odd_indices]
        data2_odd = data2[odd_indices]
        # print(f"data1_odd: {data1_odd}, data2_odd: {data2_odd}")

        data1_stromal_even = np.nanmean(data1_even[0])
        data2_stromal_even = np.nanmean(data2_even[0])
        data1_peritumoral_even = np.nanmean(data1_even[1])
        data2_peritumoral_even = np.nanmean(data2_even[1])
        # print(f"data1_stromal_even: {data1_stromal_even}, data2_stromal_even: {data2_stromal_even}")
        # print(f"data1_peritumoral_even: {data1_peritumoral_even}, data2_peritumoral_even: {data2_peritumoral_even}")

        data1_stromal_odd = np.nanmax(data1_odd[0])
        data2_stromal_odd = np.nanmax(data2_odd[0])
        data1_peritumoral_odd = np.nanmax(data1_odd[1])
        data2_peritumoral_odd = np.nanmax(data2_odd[1])
        # print(f"data1_stromal_odd: {data1_stromal_odd}, data2_stromal_odd: {data2_stromal_odd}")
        # print(f"data1_peritumoral_odd: {data1_peritumoral_odd}, data2_peritumoral_odd: {data2_peritumoral_odd}")

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
    df_black.to_csv(f"data/hari_BC/csv/{prefix}otsu3_black_cohort_differences.csv", index=False)
    df_white.to_csv(f"data/hari_BC/csv/{prefix}otsu3_white_cohort_differences.csv", index=False)

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
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1),   # black median line
                        showmeans=True, meanline=True,                 # mean line
                        meanprops=dict(color="black", linestyle="--", linewidth=1))  # dotted mean line

    # Colors for means
    mean_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]  # pinks & violets
    for patch, color in zip(box1['boxes'], mean_colors):
        patch.set_facecolor(color)

    # Overlay median markers
    for i, line in enumerate(box1['medians']):
        x, y = line.get_xydata()[1]   # (x, y) of median

    axes[0].set_title("Mean Values")
    axes[0].set_ylabel("Value")

    # Legend for left plot (upper left)
    median_line = plt.Line2D([0], [0], color="black", linewidth=1, label="Median")
    mean_line = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    axes[0].legend(handles=[median_line, mean_line], loc="upper left")

    # --- Max plot ---
    box2 = axes[1].boxplot(max_data,
                        labels=["Stromal:Black", "Stromal:White", "Peritumoral:Black", "Peritumoral:White"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1),   # black median line
                        showmeans=True, meanline=True,
                        meanprops=dict(color="black", linestyle="--", linewidth=1))  # dotted mean line

    # Colors for max
    max_colors = ["#FF69B4", "#FF1493", "#9370DB", "#8A2BE2"]
    for patch, color in zip(box2['boxes'], max_colors):
        patch.set_facecolor(color)

    # Overlay median markers
    for i, line in enumerate(box2['medians']):
        x, y = line.get_xydata()[1]   # (x, y) of median

    axes[1].set_title("Max Values")
    axes[1].set_ylabel("Value")

    # Legend for right plot (upper left)
    median_line = plt.Line2D([0], [0], color="black", linewidth=1, label="Median")
    mean_line = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Mean")
    axes[1].legend(handles=[median_line, mean_line], loc="upper right")

    plt.suptitle("Black vs White Cohort: Stromal & Peritumoral Boxplots")
    plt.tight_layout()
    plt.savefig(f"data/hari_BC/plots/{prefix}otsu3_BvW_comparison.png", dpi=300)