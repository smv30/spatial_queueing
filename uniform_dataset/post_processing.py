import numpy as np
import pandas as pd
import os
import argparse
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import linregress


def consolidate_the_kpis(root_dir, old_kpi=False):
    """
    Walks root_dir (and all subdirectories), finds every 'kpi.csv',
    concatenates them, writes 'kpi_consolidated.csv' into root_dir,
    and returns the consolidated DataFrame.
    """
    dfs = []
    for filepath in Path(root_dir).rglob("kpi.csv"):
        try:
            df = pd.read_csv(filepath)
            df["results_folder"] = Path(filepath)
            dfs.append(df)
        except FileNotFoundError:
            # shouldn't happen since rglob found it, but just in case
            continue

    if not dfs:
        raise FileNotFoundError(f"No 'kpi.csv' files found under {root_dir!r}")

    kpi_consolidated = pd.concat(dfs, ignore_index=True)
    group_cols = [
        "fleet_size", "n_chargers", "n_posts", "arrival_rate_pmin", "consumption_kwhpmi",
          "charge_rate_kw", "avg_vel_mph", "pack_size_kwh", "matching_algorithm"
    ]

    kpi_consolidated = (
        kpi_consolidated
        .groupby(group_cols, as_index=False)
        .agg({
            "service_level_percentage_second_half": "mean",
            "avg_pickup_time_min_second_half": "mean",
            "avg_drive_time_to_charger": "mean",
            "avg_trip_time_fulfilled_min_second_half": "mean",
            "total_n_trips": "mean",
            "avg_trip_time_min": "mean",
            "total_sim_duration_min": "count" # n_posts can be any other column in dfs
        })
        .rename(columns={"total_sim_duration_min": "n_repeat"})
        .round({
        "service_level_percentage_second_half": 3,
        "avg_pickup_time_min_second_half": 2,
        "avg_drive_time_to_charger": 2,
        "avg_trip_time_fulfilled_min_second_half": 2,
        "total_n_trips": 0,
        "avg_trip_time_min": 2
        })
        )
    kpi_consolidated["r"] = kpi_consolidated["consumption_kwhpmi"] * kpi_consolidated["avg_vel_mph"] / kpi_consolidated["charge_rate_kw"]
    kpi_consolidated["first_order_fleet_size"] = (1 + kpi_consolidated["r"]) * kpi_consolidated["arrival_rate_pmin"] * 0.9 * kpi_consolidated["avg_trip_time_fulfilled_min_second_half"]
    kpi_consolidated["first_order_n_chargers"] = kpi_consolidated["r"] * kpi_consolidated["arrival_rate_pmin"] * 0.9 * kpi_consolidated["avg_trip_time_fulfilled_min_second_half"]
    kpi_consolidated["buffer_fleet_size"] = kpi_consolidated["fleet_size"] - kpi_consolidated["first_order_fleet_size"]
    kpi_consolidated["buffer_n_chargers"] = kpi_consolidated["n_chargers"] * kpi_consolidated["n_posts"] - kpi_consolidated["first_order_n_chargers"]

    output_path = Path(root_dir) / "kpi_consolidated.csv"
    kpi_consolidated.to_csv(output_path, index=False)
    return kpi_consolidated

def asymptotic_plots(kpi_consolidated, plot_name, root_dir):
    rank_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    kpi_consolidated["series"] = (
        kpi_consolidated.groupby("arrival_rate_pmin")["n_chargers"]
        .rank(method="first", ascending=False)  # rank within group
        .astype(int)
        .map(rank_map)  # map rank to letter
    )
    if plot_name == "buffer_fleet_size":
        y_axis_title = r"$n-(1+r)\tilde{T}_R\alpha \lambda$"
        column_name = "buffer_fleet_size"
    elif plot_name == "buffer_n_chargers":
        y_axis_title = r"$m-r\tilde{T}_R \alpha \lambda$"
        column_name = "buffer_n_chargers"
    elif plot_name == "pickup_min":
        y_axis_title = "Average Pickup Time (min)"
        column_name = "avg_pickup_time_min_second_half"
    elif plot_name == "drive_to_charger_min":
        y_axis_title = "Average Drive to Charger Time (min)"
        column_name = "avg_drive_time_to_charger"
    fig, ax = plt.subplots()
    ax.set_xlabel("Arrival Rate (per min)", fontsize=15)
    ax.set_ylabel(y_axis_title, fontsize=15)
    ax.tick_params(axis='both', labelsize=14)

    list_color = ['#1F77B4', '#FC8D62', '#9467BD', '#E6AB02']
    list_linestyle = ["solid", "dashed", "dashdot", "dotted"]
    list_marker = ["o", "s", "*", "D"]
    linewidth = 3
    list_markersize = [8, 8, 12, 8]
    count = 0
    for series in ["A", "B", "C", "D"]:
        df_filtered = kpi_consolidated[kpi_consolidated["series"] == series].sort_values(["arrival_rate_pmin"], ascending=True)
        x = np.log(df_filtered["arrival_rate_pmin"])
        y = np.log(df_filtered[column_name])
        result = linregress(x, y)
        slope = np.round(result.slope, 3)
        ax.plot(df_filtered["arrival_rate_pmin"], df_filtered[column_name],
                  label=f"{series}, fit={slope}", color=list_color[count],
                    linestyle=list_linestyle[count], linewidth=linewidth,
                      marker=list_marker[count], markersize=list_markersize[count])
        count += 1
    ax.legend(fontsize=13, handlelength=4)

    plot_filename = os.path.join(root_dir, f"asymptotic_{plot_name}.png")
    plot_filename_pgf = os.path.join(root_dir, f"asymptotic_{plot_name}.pgf")
    # plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,    # use inline math for ticks
            "pgf.rcfonts": False,   # don't setup fonts from rc parameters
            "pgf.texsystem": "pdflatex" # or "xelatex" or "lualatex"
        })
    plt.savefig(plot_filename_pgf, format='pgf')

def contour_plots(kpi_consolidated, plot_name, root_dir):
    kpi_consolidated = kpi_consolidated.sort_values(by=["n_chargers", "pack_size_kwh", "fleet_size"])
    # Define the data
    fleet_size = kpi_consolidated["fleet_size"].unique()
    n_chargers = kpi_consolidated["n_chargers"].unique()  
    packsize_kwh = kpi_consolidated["pack_size_kwh"].unique()
    service_level_1d = kpi_consolidated["service_level_percentage_second_half"].to_numpy()

    if plot_name == "charger_contour":
        # Reshape the 1D array to 2D array
        service_level = service_level_1d.reshape(len(n_chargers), len(fleet_size))

        # Create a meshgrid
        X, Y = np.meshgrid(fleet_size, n_chargers)
        ylabel = "Number of Chargers"
        yticks = n_chargers
        levels = np.linspace(30, 100, num=8)
    elif plot_name == "packsize_contour":
        # Reshape the 1D array to 2D array
        service_level = service_level_1d.reshape(len(packsize_kwh), len(fleet_size))

        # Create a meshgrid
        X, Y = np.meshgrid(fleet_size, packsize_kwh)
        ylabel = "Battery Pack Size (kWh)"
        yticks = packsize_kwh
        levels = np.linspace(68, 100, num=9)

    # Plot the contour
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    contour = plt.contourf(X, Y, service_level, levels=levels, cmap='viridis')
    cbar = plt.colorbar(contour)
    plt.title("Service Level Percentage", fontsize=20)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel("Fleet Size", fontsize=20)
    plt.xticks(fleet_size, fontsize=18)
    plt.yticks(yticks, fontsize=18)
    cbar.ax.tick_params(labelsize=20)
    plot_filename = os.path.join(root_dir, f"{plot_name}.png")
    plt.savefig(plot_filename, bbox_inches='tight')

def policy_comparison_plots(kpi_consolidated, plot_name, root_dir):
    rank_map = {1: 'A', 2: 'B'}
    kpi_consolidated["series"] = (
        kpi_consolidated.groupby(["arrival_rate_pmin", "matching_algorithm"])["n_chargers"]
        .rank(method="first", ascending=False)  # rank within group
        .astype(int)
        .map(rank_map)  # map rank to letter
    )
    kpi_consolidated = kpi_consolidated.sort_values(by=["arrival_rate_pmin"])
    arrival_rates = kpi_consolidated["arrival_rate_pmin"].unique()
    kpi_consolidated["workload_served"] = kpi_consolidated["service_level_percentage_second_half"] * kpi_consolidated["avg_trip_time_fulfilled_min_second_half"] / kpi_consolidated["avg_trip_time_min"]
    if plot_name == "service_level":
        y_column = "service_level_percentage_second_half"
        y_label = "Frac of Fulfilled Trips"
    elif plot_name == "pickup_min":
        y_column = "avg_pickup_time_min_second_half"
        y_label = "Pickup Time (min)"
    elif plot_name == "drive_to_charger_min":
        y_column = "avg_drive_time_to_charger"
        y_label = "Drive to Charger Time (min)"
    elif plot_name == "workload":
        y_column = "workload_served"
        y_label = "Percentage Workload Served"
    else:
        raise ValueError(f"Plot type {plot_name} is not defined")
    list_linestyle = ["solid", "solid", "dashed", "dashed", "dashdot", "dashdot"]
    linewidth = 3
    count = 0
    list_markers = ["o", "*", "p", "s", "o", "D"]
    list_markersize = [8, 12, 12, 8, 8, 8]
    list_color = ["#7B3F00", "#9467BD", "#2CA02C", "#ff7f00", "#E6AB02", "#1F77B4"]
    for policy in ["Power-of-2", "Closest Dispatch", "Closest Available Dispatch"]:
        for series in ["B", "A"]:
            y = kpi_consolidated[(kpi_consolidated["matching_algorithm"] == policy) & (kpi_consolidated["series"] == series)][y_column].to_numpy()
            plt.plot(arrival_rates, y, 
                     linestyle=list_linestyle[count], linewidth=linewidth,
                     markersize=list_markersize[count], marker=list_markers[count],
                     color=list_color[count]
                     )
            count += 1
    plt.xlabel("Arrival Rate (per min)", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plot_filename = os.path.join(root_dir, f"{plot_name}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.clf()
    

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--root_dir', type=str, default=root_dir)
    parser.add_argument('-output', '--output_type', type=str, default="fleet_plot")
    args = parser.parse_args()
    if args.output_type == "asymptotic_sim":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        asymptotic_plots(kpi_consolidated, "buffer_fleet_size", args.root_dir)
        asymptotic_plots(kpi_consolidated, "buffer_n_chargers", args.root_dir)
        asymptotic_plots(kpi_consolidated, "pickup_min", args.root_dir)
        asymptotic_plots(kpi_consolidated, "drive_to_charger_min", args.root_dir)
    elif args.output_type == "charger_contour":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        contour_plots(kpi_consolidated, "charger_contour", args.root_dir)
    elif args.output_type == "packsize_contour":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        contour_plots(kpi_consolidated, "packsize_contour", args.root_dir)
    elif args.output_type == "policy_comparison":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        policy_comparison_plots(kpi_consolidated, "service_level", args.root_dir)
        policy_comparison_plots(kpi_consolidated, "pickup_min", args.root_dir)
        policy_comparison_plots(kpi_consolidated, "drive_to_charger_min", args.root_dir)
        policy_comparison_plots(kpi_consolidated, "workload", args.root_dir)
