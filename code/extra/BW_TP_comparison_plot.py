import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

def plot_donor_data(df, x_col):
    """
    Plot donor data side-by-side for Black and White cohorts with separate Y-axis labels.
    Older donation year -> orange circle, newer donation year -> blue circle
    """
    df["PID"] = df["PID"].astype(str)
    races = ["Black", "White"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True)  # remove sharey=True

    for ax, race in zip(axes, races):
        race_df = df[df["Race"] == race].dropna(subset=[x_col])

        for donor, group in race_df.groupby("PID"):
            # Sort by Donation Year to determine older vs newer
            group_sorted = group.sort_values("Donation Year")
            colors = ["orange", "blue"]  # older -> orange, newer -> blue

            # Plot gray lines connecting the points
            ax.plot(group_sorted[x_col], [donor]*len(group_sorted),
                    color="gray", alpha=0.5, linewidth=2, zorder=1)

            # Scatter points with colors based on order
            for x, color in zip(group_sorted[x_col], colors):
                ax.scatter(x, donor, color=color, edgecolor="black", s=150, zorder=2)
                ax.scatter(x, donor, color="black", marker="x", s=100, zorder=3)

        ax.set_title(f"{race} Cohort")
        ax.set_ylabel("PID")
        ax.set_xlabel(x_col)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Add a single legend for the color meaning
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Older Donation Year'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Newer Donation Year'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper left')

    plt.suptitle(f"Donor Trajectories by Race: {x_col}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"data/hari_BC/plots/{x_col.replace(' ', '')}.png", dpi=300)


def plot_age_distribution(df, age_column):
    df["PID"] = df["PID"].astype(str)

    # Keep only the first timepoint per donor
    df_first = df.sort_values(age_column).groupby("PID").first().reset_index()

    races = ["Black", "White"]
    colors = ["blue", "orange"]

    plt.figure(figsize=(8,6))

    # Decide bins automatically (you can adjust bin width here, e.g., 5-year bins)
    all_ages = df_first[age_column].dropna().values
    bins = np.arange(int(np.min(all_ages)), int(np.max(all_ages)) + 5, 5)  # 5-year bins

    for race, color in zip(races, colors):
        race_df = df_first[df_first["Race"] == race]
        ages_data = race_df[age_column].dropna().values

        if len(ages_data) == 0:
            continue

        # Plot histogram
        plt.hist(ages_data, bins=bins, color=color, alpha=0.6, label=f"{race} Cohort", edgecolor='black')

    plt.xlabel(f"{age_column} at First Timepoint")
    plt.ylabel("Number of Donors")
    plt.title(f"Distribution of {age_column} at First Timepoint by Race")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(bins)  # show bin edges on x-axis
    plt.tight_layout()
    plt.savefig(f"data/hari_BC/plots/hist_{age_column.replace(' ', '')}.png", dpi=300)


def plot_donation_year_diff_hist(df):

    df["PID"] = df["PID"].astype(str)

    races = ["Black", "White"]
    colors = ["blue", "orange"]

    # Calculate year difference per donor
    diff_list = []

    for pid, group in df.groupby("PID"):
        group_sorted = group.sort_values("Donation Year")
        if len(group_sorted) < 2:
            continue  # skip donors with only one timepoint
        older = group_sorted.iloc[0]["Donation Year"]
        newer = group_sorted.iloc[1]["Donation Year"]
        diff = newer - older
        race = group_sorted.iloc[0]["Race"]
        diff_list.append({"PID": pid, "Race": race, "Year_Diff": diff})

    diff_df = pd.DataFrame(diff_list)

    plt.figure(figsize=(8,6))

    # Decide bins automatically
    all_diffs = diff_df["Year_Diff"].dropna().values
    if len(all_diffs) == 0:
        print("No donors with multiple timepoints found.")
        return
    bins = np.arange(int(np.min(all_diffs)), int(np.max(all_diffs)) + 1, 1)  # 1-year bins

    for race, color in zip(races, colors):
        race_df = diff_df[diff_df["Race"] == race]
        diffs = race_df["Year_Diff"].dropna().values
        if len(diffs) == 0:
            print("missing!")
            continue
        plt.hist(diffs, bins=bins, color=color, alpha=0.6, label=f"{race} Cohort", edgecolor='black')

    plt.xlabel("Donation Year Difference (Newer - Older)")
    plt.ylabel("Number of Donors")
    plt.title("Histogram of Donation Year Differences by Race")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig("data/hari_BC/plots/donation_year_diff_hist.png", dpi=300)


def plot_donor_data_by_age(df, x_col):
    """
    Plot donor data side-by-side for Black and White cohorts,
    separately for donors with older donation age in 0–40 and 40–60.
    Older donation year -> orange circle, newer donation year -> blue circle.
    """

    df["PID"] = df["PID"].astype(str)

    # Determine the older (first) donation age for each donor
    older_ages = (
        df.sort_values("Donation Year")
          .groupby("PID")
          .first()["Age"]
          .rename("OlderAge")
    )
    df = df.merge(older_ages, on="PID", how="left")

    # Define age bins
    age_bins = [(0, 40), (40, 60)]

    for low, high in age_bins:
        subset = df[(df["OlderAge"] >= low) & (df["OlderAge"] < high)]
        if subset.empty:
            print(f"No donors in age range {low}-{high}")
            continue

        races = ["Black", "White"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True)

        for ax, race in zip(axes, races):
            race_df = subset[subset["Race"] == race].dropna(subset=[x_col])

            for donor, group in race_df.groupby("PID"):
                # Sort by Donation Year
                group_sorted = group.sort_values("Donation Year")
                colors = ["orange", "blue"]

                # Connect points
                ax.plot(group_sorted[x_col], [donor]*len(group_sorted),
                        color="gray", alpha=0.5, linewidth=2, zorder=1)

                # Scatter older/newer points
                for x, color in zip(group_sorted[x_col], colors):
                    ax.scatter(x, donor, color=color, edgecolor="black", s=150, zorder=2)
                    ax.scatter(x, donor, color="black", marker="x", s=100, zorder=3)

            ax.set_title(f"{race} Cohort")
            ax.set_ylabel("PID")
            ax.set_xlabel(x_col)
            ax.grid(True, linestyle="--", alpha=0.5)

        # Legend and title
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Older Donation Year'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Newer Donation Year'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper left')

        plt.suptitle(f"Donor Trajectories by Race: {x_col} (Older Age {low}–{high})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"data/hari_BC/{x_col.replace(' ', '')}_age_{low}-{high}.png", dpi=300)
        plt.close(fig)



# Example usage

# Load your CSV
df = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")

# plot_age_distribution(df, "Age")

plot_donation_year_diff_hist(df)

# plot_donor_data(df, "Donation Year")
# plot_donor_data(df, "ZEB1_Positivity")
# plot_donor_data(df, "ZEB1_H-score")
# plot_donor_data(df, "FOXA1_Positivity")
# plot_donor_data(df, "FOXA1_H-score")


plot_donor_data_by_age(df, "Stromal_Mean")
plot_donor_data_by_age(df, "Peritumoral_Mean")