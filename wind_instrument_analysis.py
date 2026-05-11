# ============================================================
# 3-WIND-INSTRUMENT COMPARISON SCRIPT
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

    # Vectorized calculation of differences
    diffs = angles_deg.apply(lambda col: np.abs(circular_difference_deg(col, mean_dir)))

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
    # Filter valid data once
    valid_df = df[["direction_sector", f"{variable}_spread"]].dropna()

    if valid_df.empty:
        return

    sector_values = sorted(valid_df["direction_sector"].unique())

    # Use groupby for efficient filtering
    grouped = valid_df.groupby("direction_sector")[f"{variable}_spread"]
    data_to_plot = [grouped.get_group(sector).values for sector in sector_values]

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

    valid_df = df[cols].dropna()

    if valid_df.empty:
        return

    # Vectorized ranking: argsort gives indices, argsort again gives ranks
    ranks = valid_df.values.argsort(axis=1).argsort(axis=1)

    # Count occurrences of each rank for each sensor
    rank_counts = pd.DataFrame(index=sensors, columns=["Lowest", "Middle", "Highest"], dtype=int)
    rank_counts["Lowest"] = (ranks == 0).sum(axis=0)
    rank_counts["Middle"] = (ranks == 1).sum(axis=0)
    rank_counts["Highest"] = (ranks == 2).sum(axis=0)

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

    # Filter out rows with missing direction_sector
    valid_mask = df["direction_sector"].notna()

    # Create DataFrame with absolute deviations for each sensor
    deviation_cols = [f"{sensor}_{variable}_minus_consensus" for sensor in sensors]
    deviations = df.loc[valid_mask, deviation_cols].abs()

    # Check if all deviation columns are present
    if deviations.isna().all().all():
        return

    # Find which sensor has max deviation for each row (vectorized)
    odd_sensor_indices = deviations.values.argmax(axis=1)
    odd_sensors = [sensors[i] for i in odd_sensor_indices]

    # Create DataFrame with direction_sector and odd_sensor
    odd_df = pd.DataFrame({
        "direction_sector": df.loc[valid_mask, "direction_sector"].values,
        "odd_sensor": odd_sensors
    })

    # Remove rows where all deviations were NaN
    odd_df = odd_df[deviations.notna().any(axis=1).values]

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
    for sensor_a, sensor_b in pairwise_pairs():
        diff = df[f"{sensor_a}_{variable}"] - df[f"{sensor_b}_{variable}"]

        # Create temporary DataFrame with direction_sector and diff
        temp_df = pd.DataFrame({
            "direction_sector": df["direction_sector"],
            "diff": diff
        }).dropna()

        if temp_df.empty:
            continue

        sector_values = sorted(temp_df["direction_sector"].unique())

        # Use groupby for efficient filtering
        grouped = temp_df.groupby("direction_sector")["diff"]
        data_to_plot = [grouped.get_group(sector).values for sector in sector_values]

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



def calculate_data_availability(df):
    """
    Calculate data availability for each instrument and variable.

    Returns:
        DataFrame with availability statistics for each sensor and measurement type.
    """
    # Calculate time span
    first_timestamp = df["timestamp"].min()
    last_timestamp = df["timestamp"].max()

    time_span_minutes = (last_timestamp - first_timestamp).total_seconds() / 60
    expected_readings = int(time_span_minutes) + 1  # +1 to include both endpoints

    availability_data = []

    # Define all sensor-variable combinations
    sensor_vars = [
        ("THIES", "avg"),
        ("THIES", "max"),
        ("THIES", "direction"),
        ("Ventus", "avg"),
        ("Ventus", "max"),
        ("Ventus", "direction"),
        ("A100&W200", "avg"),
        ("A100&W200", "max"),
        ("A100&W200", "direction"),
    ]

    for sensor, variable in sensor_vars:
        col_name = f"{sensor}_{variable}"
        actual_readings = df[col_name].notna().sum()
        availability_percent = (actual_readings / expected_readings) * 100

        availability_data.append({
            "sensor": sensor,
            "variable": variable,
            "expected_readings": expected_readings,
            "actual_readings": actual_readings,
            "missing_readings": expected_readings - actual_readings,
            "availability_percent": availability_percent
        })

    availability_df = pd.DataFrame(availability_data)

    return availability_df, first_timestamp, last_timestamp, expected_readings


def plot_data_availability(availability_df, first_timestamp, last_timestamp, expected_readings):
    """
    Plot data availability for each instrument and variable.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare data for grouped bar chart
    sensors = ["THIES", "Ventus", "A100&W200"]
    variables = ["avg", "max", "direction"]

    x = np.arange(len(sensors))
    width = 0.25

    # Plot 1: Availability percentage
    for i, var in enumerate(variables):
        var_data = availability_df[availability_df["variable"] == var]
        percentages = var_data["availability_percent"].values
        ax1.bar(x + i * width, percentages, width, label=var.capitalize())

    ax1.set_xlabel("Sensor")
    ax1.set_ylabel("Data Availability (%)")
    ax1.set_title("Data Availability by Sensor and Variable")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(sensors)
    ax1.set_ylim([0, 105])
    ax1.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Actual vs Expected readings
    for i, var in enumerate(variables):
        var_data = availability_df[availability_df["variable"] == var]
        actual = var_data["actual_readings"].values
        ax2.bar(x + i * width, actual, width, label=var.capitalize())

    ax2.axhline(expected_readings, color='red', linestyle='--', linewidth=1.5,
                label=f'Expected ({expected_readings:,})')
    ax2.set_xlabel("Sensor")
    ax2.set_ylabel("Number of Readings")
    ax2.set_title("Actual vs Expected Readings")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(sensors)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Format y-axis with comma separators
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add timeframe text
    time_span_days = (last_timestamp - first_timestamp).total_seconds() / 86400
    fig.suptitle(f'Data Availability Analysis\n'
                 f'Time Period: {first_timestamp.strftime("%Y-%m-%d %H:%M")} to '
                 f'{last_timestamp.strftime("%Y-%m-%d %H:%M")} '
                 f'({time_span_days:.1f} days)',
                 fontsize=12, y=1.02)

    save_plot("data_availability_summary.png")


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
# DATA AVAILABILITY ANALYSIS
# ============================================================

print("\nCalculating data availability...")
availability_df, first_ts, last_ts, expected_count = calculate_data_availability(df)

# Display availability summary
print("\nData Availability Summary:")
print("=" * 80)
print(f"Time Period: {first_ts.strftime('%Y-%m-%d %H:%M')} to {last_ts.strftime('%Y-%m-%d %H:%M')}")
print(f"Expected readings per instrument: {expected_count:,}")
print(f"\n{'Sensor':<15} {'Variable':<12} {'Actual':<12} {'Missing':<12} {'Availability':<12}")
print("-" * 80)
for _, row in availability_df.iterrows():
    print(f"{row['sensor']:<15} {row['variable']:<12} {int(row['actual_readings']):<12,} "
          f"{int(row['missing_readings']):<12,} {row['availability_percent']:<11.2f}%")
print("=" * 80)

# Plot data availability
plot_data_availability(availability_df, first_ts, last_ts, expected_count)

# Save availability data to CSV
availability_path = os.path.join(output_folder, "data_availability.csv")
availability_df.to_csv(availability_path, index=False)


# ============================================================
# Three-sensor diagnostics
# ============================================================

# ------------------------------------------------------------
# 1. Sensor minus consensus vs time
# ------------------------------------------------------------

plot_sensor_minus_consensus(
    df,
    "max",
    "Sensor minus consensus maximum wind speed, m/s",
    "max_sensor_minus_consensus_time.png"
)

# ------------------------------------------------------------
# 2. Sensor minus consensus vs wind speed
# ------------------------------------------------------------

plot_residual_vs_speed(
    df,
    "avg",
    "Sensor minus consensus average wind speed, m/s",
    "avg_sensor_minus_consensus_vs_speed.png"
)

plot_residual_vs_speed(
    df,
    "max",
    "Sensor minus consensus maximum wind speed, m/s",
    "max_sensor_minus_consensus_vs_speed.png"
)

# ------------------------------------------------------------
# 3. Three-sensor spread vs wind speed
# ------------------------------------------------------------

plot_spread_vs_speed(
    df,
    "avg",
    "Average wind speed spread, m/s",
    "avg_spread_vs_speed.png"
)

plot_spread_vs_speed(
    df,
    "max",
    "Maximum wind speed spread, m/s",
    "max_spread_vs_speed.png"
)

# ------------------------------------------------------------
# 4. Three-sensor spread by direction sector
# ------------------------------------------------------------

plot_spread_by_direction_sector(
    df,
    "avg",
    "Average wind speed spread, m/s",
    "avg_spread_by_direction_sector.png"
)

plot_spread_by_direction_sector(
    df,
    "max",
    "Maximum wind speed spread, m/s",
    "max_spread_by_direction_sector.png"
)

plot_spread_by_direction_sector(
    df,
    "direction",
    "Direction spread, degrees",
    "direction_spread_by_direction_sector.png"
)

#-------------------------------------------------------------
# 5. Rank frequency plots
# ------------------------------------------------------------

plot_rank_summary(
    df,
    "avg",
    "avg_rank_frequency.png"
)

plot_rank_summary(
    df,
    "max",
    "max_rank_frequency.png"
)

#-------------------------------------------------------------
# 6. Odd-one-out by direction sector
# ------------------------------------------------------------

plot_odd_one_out_by_direction(
    df,
    "avg",
    "avg_odd_one_out_by_direction.png"
)

plot_odd_one_out_by_direction(
    df,
    "max",
    "max_odd_one_out_by_direction.png"
)

plot_odd_one_out_by_direction(
    df,
    "direction",
    "direction_odd_one_out_by_direction.png"
)

#-------------------------------------------------------------
# 7. Rolling bias and rolling RMSE
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
    save_plot(f"{variable}_rolling_bias.png")

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
    save_plot(f"{variable}_rolling_rmse.png")


# ============================================================
# Pairwise diagnostics
# ============================================================

# ------------------------------------------------------------
# 1. Pairwise scatter plots with 1:1 line
# ------------------------------------------------------------

plot_pairwise_scatter(
    df,
    "avg",
    "average wind speed, m/s",
    "pairwise_scatter_avg"
)

plot_pairwise_scatter(
    df,
    "max",
    "maximum wind speed, m/s",
    "pairwise_scatter_max"
)


# ------------------------------------------------------------
# 2. Pairwise Bland-Altman plots
# ------------------------------------------------------------

plot_pairwise_bland_altman(
    df,
    "avg",
    "average wind speed, m/s",
    "bland_altman_avg"
)

plot_pairwise_bland_altman(
    df,
    "max",
    "maximum wind speed, m/s",
    "bland_altman_max"
)


# ------------------------------------------------------------
# 3. Pairwise difference by direction sector
# ------------------------------------------------------------

plot_pairwise_difference_by_direction(
    df,
    "avg",
    "Average wind speed difference, m/s",
    "avg_pairwise_difference"
)

plot_pairwise_difference_by_direction(
    df,
    "max",
    "Maximum wind speed difference, m/s",
    "max_pairwise_difference"
)


# ------------------------------------------------------------
# 4. Pairwise circular direction difference plots
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
print("  - data_availability.csv")
print("  - summary_statistics.csv")
print("  - processed_wind_comparison_data.csv")
