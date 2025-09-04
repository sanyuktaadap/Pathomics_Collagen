import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

sheets = ["White Cohort", "Black Cohort"]
cohorts = [sheet.replace(" Cohort", "") for sheet in sheets]

sheet_path = "data/hari_BC/Original/Sample list.xlsx"

donor_col = 2
year_col = 4
new_name_col = 5

for sheet, cohort in zip(sheets, cohorts):
    # Read a specific sheet by name
    df = pd.read_excel(sheet_path, sheet_name=sheet)
    for i in range(0, len(df), 2):
        donor = df.iloc[i, donor_col]
        if df.iloc[i+1, donor_col] != donor:
            print(f"Error: {donor} only appears once")
            exit()
        else:
            year1 = df.iloc[i, year_col]
            year2 = df.iloc[i+1, year_col]

            p1 = df.iloc[i, new_name_col].split("/")[-1]
            p2 = df.iloc[i+1, new_name_col].split("/")[-1]

            p1 = p1.split("_")[0]
            p2 = p2.split("_")[0]

        # Load the CSVs (no headers)
        csv_1 = pd.read_csv(f'data/hari_BC/patient_feats/{cohort}_cohort/{p1}_H&E_Breast_XXXXXXXX.csv', header=None)
        csv_2 = pd.read_csv(f'data/hari_BC/patient_feats/{cohort}_cohort/{p2}_H&E_Breast_XXXXXXXX.csv', header=None)

        # Extract values as 1D arrays
        data1 = csv_1.values.flatten()
        data2 = csv_2.values.flatten()

        # Select only even indices (0, 2, ..., 42)
        even_indices = list(range(0, 44, 2))
        data1_even = data1[even_indices]
        data2_even = data2[even_indices]

        # Split into stromal and peritumoral features
        stromal_labels = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        x = np.arange(len(stromal_labels))  # 11

        data1_stromal = data1_even[:11]
        data2_stromal = data2_even[:11]

        data1_peritumoral = data1_even[11:]
        data2_peritumoral = data2_even[11:]

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # Stromal
        ax[0].bar(x - 0.2, data1_stromal, width=0.4, label=f'{p1}-{year1}', color='steelblue')
        ax[0].bar(x + 0.2, data2_stromal, width=0.4, label=f'{p2}-{year2}', color='orange')
        ax[0].set_title('Stromal Features')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(stromal_labels, rotation=45)
        ax[0].set_xlabel('Sliding Window Size')
        ax[0].set_ylabel('Feature Value')
        ax[0].legend()

        # Peritumoral
        ax[1].bar(x - 0.2, data1_peritumoral, width=0.4, label=f'{p1}-{year1}', color='steelblue')
        ax[1].bar(x + 0.2, data2_peritumoral, width=0.4, label=f'{p2}-{year2}', color='orange')
        ax[1].set_title('Peritumoral Features')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(stromal_labels, rotation=45)
        ax[1].set_xlabel('Sliding Window Size')
        ax[1].legend()

        fig.suptitle(f'Patient-Level Feature Comparison: {p1}-{year1} vs {p2}-{year2}', fontsize=16)
        plt.tight_layout()
        filename = f"data/hari_BC/age_analysis_plots/{sheet}/{p1}-{year1} vs {p2}-{year2}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')