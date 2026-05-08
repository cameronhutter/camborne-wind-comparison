# ============================================================
# 3-WIND-INSTRUMENT COMPARISON SCRIPT
# Tier 1 and Tier 2 diagnostics
#
# Instruments:
#   1. THIES
#   2. Ventus
#   3. A100&W200
#
# Data used:
#   - Average wind speed
#   - Maximum wind speed
#   - Wind direction
#
# Author: Cameron Hutter
# ============================================================


# ============================================================
# USER INPUT SECTION
# ============================================================

csv_path = "/home/users/cameron.hutter/VS Code/Test Area/aggregated_wind_data_continuous.csv"

# Optional settings
output_folder = "/home/users/cameron.hutter/VS Code/Camborne Wind Comparison/analysis_plots"

# Wind-speed threshold for direction analysis.
# Direction is often noisy at very low wind speed.
direction_speed_threshold = 1.0  # m/s

# Rolling window size for rolling bias/RMSE plots.
# For minutely data, 60 = 1 hour.
rolling_window_minutes = 60

# Direction sector width for directional plots.
direction_sector_width = 15  # degrees

# Wind-speed bin edges for binned plots.
speed_bins = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30]


# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# BASIC SETUP
# ============================================================

os.makedirs(output_folder, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def circular_difference_deg(angle_a, angle_b):
    """
    Calculate circular difference between two wind directions in degrees.

    Returns:
        Difference in range -180 to +180 degrees.
    """
    return (angle_a - angle_b + 180) % 360 - 180


def circular_mean_deg(angles_deg):
    """
    Calculate circular mean direction from several direction columns.

    Input:
        DataFrame or array-like of directions in degrees.

    Returns:
        Circular mean direction in degrees, range 0 to 360.
    """
    angles_rad = np.deg2rad(angles_deg)

    mean_sin = np.nanmean(np.sin(angles_rad), axis=1)
    mean_cos = np.nanmean(np.cos(angles_rad), axis=1)

    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad)

    return mean_angle_deg % 360


def circular_spread_deg(angles_deg):
    """
    Simple circular spread estimate for three direction sensors.

    This calculates each sensor's circular difference from the
    circular mean direction, then returns the maximum absolute difference.

    Returns:
        Direction spread in degrees.
    """
    mean_dir = circular_mean_deg(angles_deg)

    diffs = []
    for col in angles_deg.columns:
        diffs.append(np.abs(circular_difference_deg(angles_deg[col], mean_dir)))

    diffs = pd.concat(diffs, axis=1)

    return diffs.max(axis=1)


def save_plot(filename):
    """
    Save current matplotlib figure to the output folder.
    """
    path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def add_direction_sector(direction_series, sector_width):
    """
    Convert wind direction into direction sectors.

    Example for 30 degree sectors:
        0, 30, 60, ..., 330
    """
    return ((direction_series / sector_width).round() * sector_width) % 360


def pairwise_pairs():
    """
    Return all sensor pair combinations.
    """
    return [
        ("THIES", "Ventus"),
        ("THIES", "A100&W200"),
        ("Ventus", "A100&W200"),
    ]


def plot_time_series(df, columns, title, ylabel, filename):
    """
    Simple time series plot for selected columns.
    """
    plt.figure()

    for col in columns:
        plt.plot(df["timestamp"], df[col], label=col, linewidth=0.8)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(filename)


def plot_sensor_minus_consensus(df, variable, ylabel, filename):
    """
    Plot each sensor minus the three-sensor consensus.
    """
    plt.figure()

    for sensor in ["THIES", "Ventus", "A100&W200"]:
        residual_col = f"{sensor}_{variable}_minus_consensus"
        plt.plot(df["timestamp"], df[residual_col], label=sensor, linewidth=0.8)

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{variable.upper()} sensor minus three-sensor consensus")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(filename)


def plot_residual_vs_speed(df, variable, ylabel, filename):
    """
    Plot sensor minus consensus against consensus wind speed.
    """
    plt.figure()

    for sensor in ["THIES", "Ventus", "A100&W200"]:
        residual_col = f"{sensor}_{variable}_minus_consensus"
        plt.scatter(
            df[f"{variable}_consensus"],
            df[residual_col],
            s=5,
            alpha=0.35,
            label=sensor
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"{variable.upper()} sensor minus consensus vs consensus speed")
    plt.xlabel("Consensus wind speed, m/s")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(filename)


def plot_three_sensor_spread(df, variable, ylabel, filename):
    """
    Plot three-sensor spread through time.
    """
    plt.figure()
    plt.plot(df["timestamp"], df[f"{variable}_spread"], color="black", linewidth=0.8)
    plt.title(f"Three-sensor spread: {variable.upper()}")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    save_plot(filename)


def plot_spread_vs_speed(df, variable, ylabel, filename):
    """
    Plot three-sensor spread against consensus wind speed.
    """
    plt.figure()
    plt.scatter(
        df[f"{variable}_consensus"],
        df[f"{variable}_spread"],
        s=5,
        alpha=0.35
    )
    plt.title(f"Three-sensor spread vs consensus speed: {variable.upper()}")
    plt.xlabel("Consensus wind speed, m/s")
    plt.ylabel(ylabel)
    save_plot(filename)


def plot_spread_by_direction_sector(df, variable, ylabel, filename):
    """
    Boxplot of three-sensor spread by wind direction sector.
    """
    sector_values = sorted(df["direction_sector"].dropna().unique())
    data_to_plot = []

    for sector in sector_values:
        sector_df = df[df["direction_sector"] == sector]
        data_to_plot.append(sector_df[f"{variable}_spread"].dropna())

    plt.figure(figsize=(12, 6))
    plt.boxplot(data_to_plot, labels=[int(s) for s in sector_values], showfliers=False)
    plt.title(f"Three-sensor spread by wind direction sector: {variable.upper()}")
    plt.xlabel("Direction sector, degrees")
    plt.ylabel(ylabel)
    save_plot(filename)


def plot_rank_summary(df, variable, filename):
    """
    Plot how often each sensor is highest, middle, or lowest.
    """
    sensors = ["THIES", "Ventus", "A100&W200"]
    cols = [f"{sensor}_{variable}" for sensor in sensors]

    rank_counts = pd.DataFrame(index=sensors, columns=["Lowest", "Middle", "Highest"])
    rank_counts[:] = 0

    valid_df = df[cols].dropna()

    for _, row in valid_df.iterrows():
        sorted_sensors = row.sort_values().index

        lowest_sensor = sorted_sensors[0].replace(f"_{variable}", "")
        middle_sensor = sorted_sensors[1].replace(f"_{variable}", "")
        highest_sensor = sorted_sensors[2].replace(f"_{variable}", "")

        rank_counts.loc[lowest_sensor, "Lowest"] += 1
        rank_counts.loc[middle_sensor, "Middle"] += 1
        rank_counts.loc[highest_sensor, "Highest"] += 1

    rank_percent = rank_counts.div(rank_counts.sum(axis=1), axis=0) * 100

    plt.figure()
    rank_percent.plot(kind="bar")
    plt.title(f"Rank frequency for {variable.upper()}")
    plt.xlabel("Sensor")
    plt.ylabel("Percentage of valid records, %")
    plt.xticks(rotation=0)
    plt.legend(title="Rank")
    save_plot(filename)


def plot_odd_one_out_by_direction(df, variable, filename):
    """
    Plot which sensor is furthest from the three-sensor median consensus
    by direction sector.
    """
    sensors = ["THIES", "Ventus", "A100&W200"]

    rows = []

    for _, row in df.iterrows():
        sector = row["direction_sector"]

        if pd.isna(sector):
            continue

        deviations = {}

        for sensor in sensors:
            residual_col = f"{sensor}_{variable}_minus_consensus"
            deviations[sensor] = abs(row[residual_col])

        if any(pd.isna(list(deviations.values()))):
            continue

        odd_sensor = max(deviations, key=deviations.get)

        rows.append({
            "direction_sector": sector,
            "odd_sensor": odd_sensor
        })

    odd_df = pd.DataFrame(rows)

    if odd_df.empty:
        return

    summary = (
        odd_df
        .groupby(["direction_sector", "odd_sensor"])
        .size()
        .unstack(fill_value=0)
    )

    summary_percent = summary.div(summary.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    summary_percent.plot(kind="bar", stacked=True)
    plt.title(f"Odd-one-out frequency by direction sector: {variable.upper()}")
    plt.xlabel("Direction sector, degrees")
    plt.ylabel("Percentage of records, %")
    plt.legend(title="Sensor")
    plt.xticks(rotation=45)
    save_plot(filename)


def plot_pairwise_scatter(df, variable, ylabel, filename_prefix):
    """
    Pairwise scatter plots with 1:1 line.
    """
    for sensor_x, sensor_y in pairwise_pairs():
        x_col = f"{sensor_x}_{variable}"
        y_col = f"{sensor_y}_{variable}"

        x = df[x_col]
        y = df[y_col]

        valid = x.notna() & y.notna()

        if valid.sum() == 0:
            continue

        min_value = min(x[valid].min(), y[valid].min())
        max_value = max(x[valid].max(), y[valid].max())

        plt.figure()
        plt.scatter(x[valid], y[valid], s=5, alpha=0.35)
        plt.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)

        plt.title(f"{sensor_y} vs {sensor_x}: {variable.upper()}")
        plt.xlabel(f"{sensor_x} {ylabel}")
        plt.ylabel(f"{sensor_y} {ylabel}")

        safe_name = f"{filename_prefix}_{sensor_y}_vs_{sensor_x}.png"
        safe_name = safe_name.replace("&", "and").replace(" ", "_")
        save_plot(safe_name)


def plot_pairwise_bland_altman(df, variable, ylabel, filename_prefix):
    """
    Pairwise Bland-Altman style plots:
        difference vs pair mean.
    """
    for sensor_a, sensor_b in pairwise_pairs():
        a_col = f"{sensor_a}_{variable}"
        b_col = f"{sensor_b}_{variable}"

        pair_mean = (df[a_col] + df[b_col]) / 2
        pair_diff = df[a_col] - df[b_col]

        valid = pair_mean.notna() & pair_diff.notna()

        if valid.sum() == 0:
            continue

        mean_bias = pair_diff[valid].mean()
        sd_diff = pair_diff[valid].std()

        upper_limit = mean_bias + 1.96 * sd_diff
        lower_limit = mean_bias - 1.96 * sd_diff

        plt.figure()
        plt.scatter(pair_mean[valid], pair_diff[valid], s=5, alpha=0.35)

        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(mean_bias, color="red", linewidth=1, label="Mean bias")
        plt.axhline(upper_limit, color="grey", linestyle="--", linewidth=1, label="+1.96 SD")
        plt.axhline(lower_limit, color="grey", linestyle="--", linewidth=1, label="-1.96 SD")

        plt.title(f"Bland-Altman: {sensor_a} minus {sensor_b}, {variable.upper()}")
        plt.xlabel(f"Pair mean {ylabel}")
        plt.ylabel(f"{sensor_a} - {sensor_b} {ylabel}")
        plt.legend()

        safe_name = f"{filename_prefix}_{sensor_a}_minus_{sensor_b}.png"
        safe_name = safe_name.replace("&", "and").replace(" ", "_")
        save_plot(safe_name)


def plot_pairwise_difference_by_direction(df, variable, ylabel, filename_prefix):
    """
    Pairwise difference boxplots by wind direction sector.
    """
    sector_values = sorted(df["direction_sector"].dropna().unique())

    for sensor_a, sensor_b in pairwise_pairs():
        diff = df[f"{sensor_a}_{variable}"] - df[f"{sensor_b}_{variable}"]

        data_to_plot = []

        for sector in sector_values:
            sector_data = diff[df["direction_sector"] == sector].dropna()
            data_to_plot.append(sector_data)

        plt.figure(figsize=(12, 6))
        plt.boxplot(data_to_plot, labels=[int(s) for s in sector_values], showfliers=False)
        plt.axhline(0, color="black", linewidth=1)

        plt.title(f"{sensor_a} minus {sensor_b} by direction sector: {variable.upper()}")
        plt.xlabel("Direction sector, degrees")
        plt.ylabel(ylabel)

        safe_name = f"{filename_prefix}_{sensor_a}_minus_{sensor_b}_by_direction.png"
        safe_name = safe_name.replace("&", "and").replace(" ", "_")
        save_plot(safe_name)


def plot_pairwise_direction_difference(df):
    """
    Pairwise circular direction difference plots.
    """
    for sensor_a, sensor_b in pairwise_pairs():
        diff = circular_difference_deg(
            df[f"{sensor_a}_direction"],
            df[f"{sensor_b}_direction"]
        )

        valid = (
            diff.notna()
            & df["avg_consensus"].notna()
            & (df["avg_consensus"] >= direction_speed_threshold)
        )

        if valid.sum() == 0:
            continue

        # Direction difference vs time
        plt.figure()
        plt.scatter(df.loc[valid, "timestamp"], diff[valid], s=5, alpha=0.35)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"Direction difference: {sensor_a} minus {sensor_b}")
        plt.xlabel("Time")
        plt.ylabel("Direction difference, degrees")
        save_plot(
            f"pairwise_direction_difference_time_{sensor_a}_minus_{sensor_b}.png"
            .replace("&", "and")
            .replace(" ", "_")
        )

        # Direction difference vs wind speed
        plt.figure()
        plt.scatter(df.loc[valid, "avg_consensus"], diff[valid], s=5, alpha=0.35)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"Direction difference vs wind speed: {sensor_a} minus {sensor_b}")
        plt.xlabel("Consensus average wind speed, m/s")
        plt.ylabel("Direction difference, degrees")
        save_plot(
            f"pairwise_direction_difference_vs_speed_{sensor_a}_minus_{sensor_b}.png"
            .replace("&", "and")
            .replace(" ", "_")
        )



def calculate_summary_statistics(df):
    """
    Create simple CSV summary statistics for each pair and variable.
    """
    rows = []

    for variable in ["avg", "max"]:
        for sensor_a, sensor_b in pairwise_pairs():
            a_col = f"{sensor_a}_{variable}"
            b_col = f"{sensor_b}_{variable}"

            diff = df[a_col] - df[b_col]

            valid = diff.notna() & df[a_col].notna() & df[b_col].notna()

            if valid.sum() == 0:
                continue

            bias = diff[valid].mean()
            median_bias = diff[valid].median()
            mae = diff[valid].abs().mean()
            rmse = np.sqrt((diff[valid] ** 2).mean())
            correlation = df.loc[valid, a_col].corr(df.loc[valid, b_col])

            rows.append({
                "variable": variable,
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "mean_bias_a_minus_b": bias,
                "median_bias_a_minus_b": median_bias,
                "MAE": mae,
                "RMSE": rmse,
                "correlation": correlation,
                "number_of_records": valid.sum()
            })

    # Direction summary
    for sensor_a, sensor_b in pairwise_pairs():
        diff = circular_difference_deg(
            df[f"{sensor_a}_direction"],
            df[f"{sensor_b}_direction"]
        )

        valid = (
            diff.notna()
            & df["avg_consensus"].notna()
            & (df["avg_consensus"] >= direction_speed_threshold)
        )

        if valid.sum() == 0:
            continue

        rows.append({
            "variable": "direction",
            "sensor_a": sensor_a,
            "sensor_b": sensor_b,
            "mean_bias_a_minus_b": diff[valid].mean(),
            "median_bias_a_minus_b": diff[valid].median(),
            "MAE": diff[valid].abs().mean(),
            "RMSE": np.sqrt((diff[valid] ** 2).mean()),
            "correlation": np.nan,
            "number_of_records": valid.sum()
        })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_folder, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)


# ============================================================
# READ DATA
# ============================================================

# Read CSV.
# header=0 assumes the first row contains column names.
# If your file has no header row, change header=0 to header=None.
raw = pd.read_csv(csv_path, header=0)

# Select columns by position.
# Python uses zero-based indexing:
# A=0, B=1, C=2, E=4, F=5, H=7, I=8, K=10, M=12, O=14.
df = pd.DataFrame()

df["timestamp"] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

df["THIES_avg"] = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
df["THIES_direction"] = pd.to_numeric(raw.iloc[:, 2], errors="coerce")

df["Ventus_avg"] = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
df["Ventus_direction"] = pd.to_numeric(raw.iloc[:, 5], errors="coerce")

df["A100&W200_avg"] = pd.to_numeric(raw.iloc[:, 7], errors="coerce")
df["A100&W200_direction"] = pd.to_numeric(raw.iloc[:, 8], errors="coerce")

df["A100&W200_max"] = pd.to_numeric(raw.iloc[:, 10], errors="coerce")
df["THIES_max"] = pd.to_numeric(raw.iloc[:, 12], errors="coerce")
df["Ventus_max"] = pd.to_numeric(raw.iloc[:, 14], errors="coerce")

# Sort by timestamp.
df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# REMOVE OUTLIER WIND SPEEDS ABOVE USER-DEFINED THRESHOLD
# ============================================================

# Set the maximum physically acceptable wind speed.
# Any average or maximum wind-speed value above this threshold
# will be replaced with NaN and excluded from later calculations.
#
# Example:
#   wind_speed_upper_limit = 75.0
#
# Set to None if you do not want to remove high-speed outliers.

wind_speed_upper_limit = 100  # m/s


if wind_speed_upper_limit is not None:

    wind_speed_columns = [
        "THIES_avg",
        "Ventus_avg",
        "A100&W200_avg",
        "THIES_max",
        "Ventus_max",
        "A100&W200_max",
    ]

    for col in wind_speed_columns:
        df.loc[df[col] > wind_speed_upper_limit, col] = np.nan

    print(f"Removed wind-speed readings above {wind_speed_upper_limit} m/s")


# ============================================================
# THREE-SENSOR CONSENSUS CALCULATIONS
# ============================================================

# Median consensus for average wind speed.
df["avg_consensus"] = df[
    ["THIES_avg", "Ventus_avg", "A100&W200_avg"]
].median(axis=1)

# Median consensus for maximum wind speed.
df["max_consensus"] = df[
    ["THIES_max", "Ventus_max", "A100&W200_max"]
].median(axis=1)

# Circular consensus for wind direction.
df["direction_consensus"] = circular_mean_deg(
    df[["THIES_direction", "Ventus_direction", "A100&W200_direction"]]
)

# Direction sector based on the three-sensor consensus direction.
df["direction_sector"] = add_direction_sector(
    df["direction_consensus"],
    direction_sector_width
)


# ============================================================
# SENSOR MINUS CONSENSUS
# ============================================================

for sensor in ["THIES", "Ventus", "A100&W200"]:
    df[f"{sensor}_avg_minus_consensus"] = df[f"{sensor}_avg"] - df["avg_consensus"]
    df[f"{sensor}_max_minus_consensus"] = df[f"{sensor}_max"] - df["max_consensus"]

    df[f"{sensor}_direction_minus_consensus"] = circular_difference_deg(
        df[f"{sensor}_direction"],
        df["direction_consensus"]
    )


# ============================================================
# THREE-SENSOR SPREAD
# ============================================================

df["avg_spread"] = (
    df[["THIES_avg", "Ventus_avg", "A100&W200_avg"]].max(axis=1)
    - df[["THIES_avg", "Ventus_avg", "A100&W200_avg"]].min(axis=1)
)

df["max_spread"] = (
    df[["THIES_max", "Ventus_max", "A100&W200_max"]].max(axis=1)
    - df[["THIES_max", "Ventus_max", "A100&W200_max"]].min(axis=1)
)

df["direction_spread"] = circular_spread_deg(
    df[["THIES_direction", "Ventus_direction", "A100&W200_direction"]]
)


# ============================================================
# ROLLING BIAS AND RMSE AGAINST CONSENSUS
# ============================================================

for variable in ["avg", "max"]:
    for sensor in ["THIES", "Ventus", "A100&W200"]:
        residual_col = f"{sensor}_{variable}_minus_consensus"

        df[f"{sensor}_{variable}_rolling_bias"] = (
            df[residual_col]
            .rolling(rolling_window_minutes, min_periods=10)
            .mean()
        )

        df[f"{sensor}_{variable}_rolling_rmse"] = np.sqrt(
            (df[residual_col] ** 2)
            .rolling(rolling_window_minutes, min_periods=10)
            .mean()
        )


# ============================================================
# TIER 1 PLOTS
# Three-sensor diagnostics
# ============================================================

# ------------------------------------------------------------
# 1. Time-series overlays
# ------------------------------------------------------------

plot_time_series(
    df,
    ["THIES_avg", "Ventus_avg", "A100&W200_avg"],
    "Average wind speed time series",
    "Average wind speed, m/s",
    "tier1_avg_wind_speed_time_series.png"
)

plot_time_series(
    df,
    ["THIES_max", "Ventus_max", "A100&W200_max"],
    "Maximum wind speed time series",
    "Maximum wind speed, m/s",
    "tier1_max_wind_speed_time_series.png"
)

# ------------------------------------------------------------
# 2. Sensor minus consensus vs time
# ------------------------------------------------------------

plot_sensor_minus_consensus(
    df,
    "avg",
    "Sensor minus consensus average wind speed, m/s",
    "tier1_avg_sensor_minus_consensus_time.png"
)

plot_sensor_minus_consensus(
    df,
    "max",
    "Sensor minus consensus maximum wind speed, m/s",
    "tier1_max_sensor_minus_consensus_time.png"
)

# ------------------------------------------------------------
# 3. Sensor minus consensus vs wind speed
# ------------------------------------------------------------

plot_residual_vs_speed(
    df,
    "avg",
    "Sensor minus consensus average wind speed, m/s",
    "tier1_avg_sensor_minus_consensus_vs_speed.png"
)

plot_residual_vs_speed(
    df,
    "max",
    "Sensor minus consensus maximum wind speed, m/s",
    "tier1_max_sensor_minus_consensus_vs_speed.png"
)



# ------------------------------------------------------------
# 5. Three-sensor spread
# ------------------------------------------------------------


plot_three_sensor_spread(
    df,
    "max",
    "Maximum wind speed spread, m/s",
    "tier1_max_three_sensor_spread_time.png"
)

# ------------------------------------------------------------
# 6. Three-sensor spread vs wind speed
# ------------------------------------------------------------

plot_spread_vs_speed(
    df,
    "avg",
    "Average wind speed spread, m/s",
    "tier1_avg_spread_vs_speed.png"
)

plot_spread_vs_speed(
    df,
    "max",
    "Maximum wind speed spread, m/s",
    "tier1_max_spread_vs_speed.png"
)

# ------------------------------------------------------------
# 7. Three-sensor spread by direction sector
# ------------------------------------------------------------

plot_spread_by_direction_sector(
    df,
    "avg",
    "Average wind speed spread, m/s",
    "tier1_avg_spread_by_direction_sector.png"
)

plot_spread_by_direction_sector(
    df,
    "max",
    "Maximum wind speed spread, m/s",
    "tier1_max_spread_by_direction_sector.png"
)

plot_spread_by_direction_sector(
    df,
    "direction",
    "Direction spread, degrees",
    "tier1_direction_spread_by_direction_sector.png"
)


# ------------------------------------------------------------
# 8. Rank frequency plots
# ------------------------------------------------------------

plot_rank_summary(
    df,
    "avg",
    "tier1_avg_rank_frequency.png"
)

plot_rank_summary(
    df,
    "max",
    "tier1_max_rank_frequency.png"
)


# ------------------------------------------------------------
# 9. Odd-one-out by direction sector
# ------------------------------------------------------------

plot_odd_one_out_by_direction(
    df,
    "avg",
    "tier1_avg_odd_one_out_by_direction.png"
)

plot_odd_one_out_by_direction(
    df,
    "max",
    "tier1_max_odd_one_out_by_direction.png"
)

plot_odd_one_out_by_direction(
    df,
    "direction",
    "tier1_direction_odd_one_out_by_direction.png"
)


# ------------------------------------------------------------
# 10. Rolling bias and rolling RMSE
# ------------------------------------------------------------

for variable in ["avg", "max"]:
    plt.figure()

    for sensor in ["THIES", "Ventus", "A100&W200"]:
        plt.plot(
            df["timestamp"],
            df[f"{sensor}_{variable}_rolling_bias"],
            label=sensor,
            linewidth=0.9
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Rolling bias against consensus: {variable.upper()}")
    plt.xlabel("Time")
    plt.ylabel("Rolling bias, m/s")
    plt.legend()
    save_plot(f"tier1_{variable}_rolling_bias.png")

    plt.figure()

    for sensor in ["THIES", "Ventus", "A100&W200"]:
        plt.plot(
            df["timestamp"],
            df[f"{sensor}_{variable}_rolling_rmse"],
            label=sensor,
            linewidth=0.9
        )

    plt.title(f"Rolling RMSE against consensus: {variable.upper()}")
    plt.xlabel("Time")
    plt.ylabel("Rolling RMSE, m/s")
    plt.legend()
    save_plot(f"tier1_{variable}_rolling_rmse.png")


# ============================================================
# TIER 2 PLOTS
# Pairwise diagnostics
# ============================================================

# ------------------------------------------------------------
# 1. Pairwise scatter plots with 1:1 line
# ------------------------------------------------------------

plot_pairwise_scatter(
    df,
    "avg",
    "average wind speed, m/s",
    "tier2_pairwise_scatter_avg"
)

plot_pairwise_scatter(
    df,
    "max",
    "maximum wind speed, m/s",
    "tier2_pairwise_scatter_max"
)


# ------------------------------------------------------------
# 2. Pairwise Bland-Altman plots
# ------------------------------------------------------------

plot_pairwise_bland_altman(
    df,
    "avg",
    "average wind speed, m/s",
    "tier2_bland_altman_avg"
)

plot_pairwise_bland_altman(
    df,
    "max",
    "maximum wind speed, m/s",
    "tier2_bland_altman_max"
)


# ------------------------------------------------------------
# 3. Pairwise difference by direction sector
# ------------------------------------------------------------

plot_pairwise_difference_by_direction(
    df,
    "avg",
    "Average wind speed difference, m/s",
    "tier2_avg_pairwise_difference"
)

plot_pairwise_difference_by_direction(
    df,
    "max",
    "Maximum wind speed difference, m/s",
    "tier2_max_pairwise_difference"
)


# ------------------------------------------------------------
# 5. Pairwise circular direction difference plots
# ------------------------------------------------------------

plot_pairwise_direction_difference(df)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

calculate_summary_statistics(df)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

processed_data_path = os.path.join(output_folder, "processed_wind_comparison_data.csv")
df.to_csv(processed_data_path, index=False)


# ============================================================
# FINISHED
# ============================================================

print("Analysis complete.")
print(f"Plots and output files saved in: {output_folder}")
print("Key output files:")
print("  - summary_statistics.csv")
print("  - processed_wind_comparison_data.csv")
