import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def assign_age_color(row):
    if row["PID"] == "AAAADB":
        return "red"
    age = row["Age"]
    if 20 <= age <= 40:
        return "green"
    elif 40 < age <= 60:
        return "blue"
    else:
        return "red"


def plot_group_with_trend(sub_df, group_name, save_suffix):
    """Helper to plot a subset and add a trend line if possible."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if sub_df.empty:
        ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center")
        ax.set_xlabel("Stromal_Mean")
        ax.set_ylabel("Age")
        ax.set_title(f"Age vs Stromal_Mean — {save_name} — {group_name}")
        out = f"data/hari_BC/plots/otsu4_{save_name}_{save_suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved (empty plot): {out}")
        return

    # Scatter points
    ax.scatter(sub_df["Stromal_Mean"], sub_df["Age"],
            c=sub_df["Color"], alpha=0.8, edgecolors="black", label=group_name)

    # Add linear fit (trend) if at least 2 points
    x = sub_df["Stromal_Mean"].values
    y = sub_df["Age"].values
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, linestyle="--", color="black", linewidth=1, label="Trend")
    else:
        # annotate that trend not plotted
        ax.text(0.02, 0.95, "Trend: n<2", transform=ax.transAxes, fontsize=9,
                verticalalignment='top')

    # Labels, title, legend
    ax.set_xlabel("Stromal_Mean")
    ax.set_ylabel("Age")
    ax.set_title(f"Age vs Stromal_Mean — {save_name} — {group_name}")
    # Build legend so it shows the color patch and trend line (if exists)
    patches = []
    # color patch
    patches.append(mpatches.Patch(color=sub_df["Color"].iloc[0], label=group_name))
    # add trend line handle manually if plotted
    if len(x) >= 2:
        patches.append(Line2D([0], [0], color="black", linestyle="--", label="Trend"))
    ax.legend(handles=patches, loc="best")

    out = f"data/hari_BC/plots/otsu4_{save_name}_{save_suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("data/hari_BC/csv/BnW_combined.csv")

    df_black = df[df["Race"] == "Black"].copy()
    df_white = df[df["Race"] == "White"].copy()

    dfs = [df_black, df_white]
    save_names = ["Black", "White"]

    for df, save_name in zip(dfs, save_names):
        # Drop missing
        df = df.dropna(subset=["Age", "Stromal_Mean"]).copy()

        # Apply color assignment
        df["Color"] = df.apply(assign_age_color, axis=1)

        ################## YEARS VS ENTROPY CHANGE PLOT ##################
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df["Stromal_Mean"], df["Age"],
                    c=df["Color"], alpha=0.7, edgecolors="black")

        # ---- Add trend arrow ----
        x = df["Stromal_Mean"].values
        y = df["Age"].values

        # Fit a linear regression (y = m*x + b)
        m, b = np.polyfit(x, y, 1)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = m * x_min + b, m * x_max + b

        # Draw arrow from (x_min, y_min) → (x_max, y_max)
        plt.arrow(
            x_min, y_min,
            (x_max - x_min), (y_max - y_min),
            color="black", width=0.001, head_width=0.02,
            length_includes_head=True, alpha=0.7,
            label="Trend"
        )

        # Add legend entry manually
        # plt.legend(["Trend (Age vs Stromal_Mean)"], loc="best")

        # Add labels and title
        plt.xlabel("Stromal_Mean")
        plt.ylabel("Age")
        plt.title(f"Age vs Stromal_Mean — Race: {save_name}")

        # Optional: Add grid
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save figure
        plt.savefig(f"data/hari_BC/plots/otsu4_{save_name}_age_vs_entropy.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

        # Ensure numeric types
        df["Donation Year"] = pd.to_numeric(df["Donation Year"], errors="coerce")
        df["Stromal_Mean"] = pd.to_numeric(df["Stromal_Mean"], errors="coerce")

        # Prepare lists to collect per-subject metrics
        pids = []
        year_diffs = []
        stromal_diffs = []
        abs_stromal_diffs = []
        colors = []

        # Group by subject (PID)
        for pid, grp in df.groupby("PID"):
            grp = grp.dropna(subset=["Donation Year", "Stromal_Mean"])
            if len(grp) < 2:
                continue  # need at least two timepoints

            grp_sorted = grp.sort_values("Donation Year", kind="mergesort")
            first = grp_sorted.iloc[0]
            second = grp_sorted.iloc[1]

            year1 = int(first["Donation Year"])
            year2 = int(second["Donation Year"])
            s1 = float(first["Stromal_Mean"])
            s2 = float(second["Stromal_Mean"])

            year_diff = year2 - year1
            stromal_diff = s1 - s2
            abs_diff = abs(stromal_diff)

            if stromal_diff > 0:
                col = "green"   # entropy increased
            elif stromal_diff < 0:
                col = "red"     # entropy decreased

            pids.append(pid)
            year_diffs.append(year_diff)
            stromal_diffs.append(stromal_diff)
            abs_stromal_diffs.append(abs_diff)
            colors.append(col)

        summary = pd.DataFrame({
            "PID": pids,
            "year_diff": year_diffs,
            "stromal_diff": stromal_diffs,
            "abs_stromal_diff": abs_stromal_diffs,
            "color": colors
        })

        if not summary.empty:
            plt.figure(figsize=(8, 6))
            plt.scatter(summary["abs_stromal_diff"], summary["year_diff"],
                        c=summary["color"], alpha=0.8, edgecolors="black")

            # --- Add general trend line (no formula label) ---
            m, b = np.polyfit(summary["abs_stromal_diff"], summary["year_diff"], 1)
            xs = np.linspace(summary["abs_stromal_diff"].min(), summary["abs_stromal_diff"].max(), 100)
            plt.plot(xs, m * xs + b, linestyle="--", linewidth=1, color="black")

            # --- Legend: patches for color categories + line for trend ---
            green_patch = mpatches.Patch(color="green", label="Entropy increased")
            red_patch = mpatches.Patch(color="red", label="Entropy decreased")
            trend_line = Line2D([0], [0], color="black", linestyle="--", label="Trend")
            plt.legend(handles=[green_patch, red_patch], loc="best")

            # Labels and formatting
            plt.xlabel("Stromal entropy change")
            plt.ylabel("Years between two timepoints")
            plt.title(f"Year difference vs Stromal Entropy Change: {save_name}")
            plt.grid(True, linestyle="--", alpha=0.5)

            outpath = f"data/hari_BC/plots/otsu_4_{save_name}_years_vs_entropy_change.png"
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved: {outpath}")
        else:
            print("No subjects with ≥2 timepoints found.")

        ################## END ##################

        ################## AGE VS ENTROPY PLOT ##################

        green_df = df[df["Color"] == "green"].copy()
        blue_df = df[df["Color"] == "blue"].copy()

        # Plot green group
        plot_group_with_trend(green_df, "Age 20-40 (green)", "green_age_vs_entropy")

        # Plot blue group
        plot_group_with_trend(blue_df, "Age 41-60 (blue)", "blue_age_vs_entropy")

        ################## END ##################