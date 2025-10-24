import pandas as pd
from scipy.stats import mannwhitneyu

df_black = pd.read_csv("data/hari_BC/csv/otsu4_black_cohort_differences_60_70_reverse_diff_all_subj.csv")
df_white = pd.read_csv("data/hari_BC/csv/otsu4_white_cohort_differences_60_70_reverse_diff_all_subj.csv")

df_black  = df_black[(df_black["Age"] >= 25) & (df_black["Age"] <= 55)]
df_white  = df_white[(df_white["Age"] >= 25) & (df_white["Age"] <= 55)]

stromal_black = list(df_black["Stromal_Mean"].values)
stromal_white = list(df_white["Stromal_Mean"].values)

# Perform the test
stat, p = mannwhitneyu(df_black, stromal_white)

print(f"p-value: {p}")