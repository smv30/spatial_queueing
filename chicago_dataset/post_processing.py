import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from datetime import timedelta, datetime


def consolidate_the_kpis(root_dir):
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
            if df["d"].iloc[0] == 2:
                if df["algo"].iloc[0] == "CAN_POD":
                    df["algo"] = "CAN_PO2"
                elif df["algo"].iloc[0] == "POD":
                    df["algo"] = "PO2"
            dfs.append(df)
        except FileNotFoundError:
            # shouldn't happen since rglob found it, but just in case
            continue

    if not dfs:
        raise FileNotFoundError(f"No 'kpi.csv' files found under {root_dir!r}")

    kpi_consolidated = pd.concat(dfs, ignore_index=True)
    group_cols = [
        "ev_type", "charge_rate_kw", "algo",
        "fleet_size", "n_chargers", "d",
        "pickup_threshold_min", "perc_trip_filter"
    ]

    kpi_consolidated = (
        kpi_consolidated
        .groupby(group_cols, as_index=False)
        .agg({
            "percentage_workload_served": "mean",
            "service_level_percentage": "mean",
            "avg_pickup_time_min": "mean",
            "avg_drive_time_to_charger": "mean",
            "n_posts": "count" # n_posts can be any other column in dfs
        })
        .rename(columns={"n_posts": "n_repeat"})
        .round({
        "percentage_workload_served": 3,
        "service_level_percentage":   1,
        "avg_pickup_time_min": 2,
        "avg_drive_time_to_charger": 2
        })
        )

    output_path = Path(root_dir) / "kpi_consolidated.csv"
    kpi_consolidated.to_csv(output_path, index=False)
    return kpi_consolidated

def compute_90_percent_fleet(kpi_consolidated, root_dir):
    # 1) sort by fleet_size
    df = kpi_consolidated.sort_values("fleet_size")

    group_cols = [
        "ev_type", "charge_rate_kw", "algo", "d",
        "pickup_threshold_min", "perc_trip_filter", "n_chargers"
    ]

    records = []
    # 2) regress fleet_size → percentage_workload_served per group
    for vals, grp in df.groupby(group_cols):
        X = grp[["fleet_size"]].values.reshape(-1,1)
        y = grp["percentage_workload_served"].values

        # skip if not enough points
        if len(grp) < 2:
            continue
        else:
            model = LinearRegression().fit(X, y)
            a, b = model.coef_[0], model.intercept_
            r2 = model.score(X, y)
            # 3) solve 0.9 = a*x + b  ⇒  x = (0.9 - b)/a
            x90 = (0.9 - b) / a if a != 0 else np.nan

        rec = dict(zip(group_cols, vals))
        rec["90_percent_fleet_size"] = int(x90)
        rec["r_squared"] = r2
        records.append(rec)
    
    # 4) build final DataFrame
    df_90_percent_fleet_size = pd.DataFrame.from_records(records)
    output_path = Path(root_dir) / "90_percent_fleet_size.csv"
    df_90_percent_fleet_size.to_csv(output_path)

    return df_90_percent_fleet_size

def optimize_fleet(df_90, root_dir):
    """
    df_90 must have columns:
      ['ev_type','charge_rate_kw','algo','d','pickup_threshold_min',
       'perc_trip_filter','n_chargers','90_percent_fleet_size']
    """

    # 1) For each (ev_type,charge_rate_kw,algo,n_chargers,d,pickup_threshold_min),
    #    sum 90_percent_fleet_size across all perc_trip_filter → total_fleet_size_across_arrival_rates
    group_tot = [
        "ev_type", "charge_rate_kw", "algo", "d", "pickup_threshold_min"
    ]
    df_tot = (
        df_90
        .groupby(group_tot)["90_percent_fleet_size"]
        .sum()
        .reset_index(name="total_fleet_size_across_arrival_rates")
    )

    # 2) For each (ev_type,charge_rate_kw,algo,n_chargers), find the (d,pickup_threshold_min)
    #    that minimizes total_fleet_size_across_arrival_rates
    group_opt = ["ev_type", "charge_rate_kw", "algo"]
    idx = (
        df_tot
        .groupby(group_opt)["total_fleet_size_across_arrival_rates"]
        .idxmin()
        .dropna()
        .astype(int)
    )
    df_opt = df_tot.loc[idx, group_opt + ["d", "pickup_threshold_min"]]
    df_opt = df_opt.rename(columns={
        "d": "optimal_d",
        "pickup_threshold_min": "optimal_pickup_threshold_min"
    }).reset_index(drop=True)

    # 3) Merge "optimal_d" and "optimal_pickup_threshold_min" back into df_90
    df_merged = df_90.merge(
        df_opt,
        on=group_opt,
        how="left"
    )

    # 4) For each (ev_type,charge_rate_kw,algo,n_chargers,perc_trip_filter),
    #    keep only rows where (d, pickup_threshold_min) == (optimal_d, optimal_pickup_threshold_min)
    mask = (
        (df_merged["d"] == df_merged["optimal_d"]) &
        (df_merged["pickup_threshold_min"] == df_merged["optimal_pickup_threshold_min"])
    )
    df_filtered = df_merged[mask].reset_index(drop=True).copy()
    df_filtered = df_filtered.drop(columns=["d", "pickup_threshold_min"])

    output_path = Path(root_dir) / "optimized_90_percent_fleet_size.csv"
    df_filtered.to_csv(output_path)

    return df_filtered 

def optimize_fleet_ode(df_90, root_dir):
    """
    df_90 must have columns:
      ['ev_type','charge_rate_kw','algo','d','pickup_threshold_min',
       'perc_trip_filter','n_chargers','90_percent_fleet_size']
    """

    # 2) For each (ev_type,charge_rate_kw,algo,n_chargers), find the (d,pickup_threshold_min)
    #    that minimizes total_fleet_size_across_arrival_rates
    group_opt = ["n_chargers"]
    idx = (
        df_90
        .groupby(group_opt)["90_percent_fleet_size"]
        .idxmin()
        .dropna()
        .astype(int)
    )
    df_opt = df_90.loc[idx, group_opt + ["d", "pickup_threshold_min"]]
    df_opt = df_opt.rename(columns={
        "d": "optimal_d",
        "pickup_threshold_min": "optimal_pickup_threshold_min"
    }).reset_index(drop=True)

    # 3) Merge "optimal_d" and "optimal_pickup_threshold_min" back into df_90
    df_merged = df_90.merge(
        df_opt,
        on=group_opt,
        how="left"
    )

    # 4) For each (ev_type,charge_rate_kw,algo,n_chargers,perc_trip_filter),
    #    keep only rows where (d, pickup_threshold_min) == (optimal_d, optimal_pickup_threshold_min)
    mask = (
        (df_merged["d"] == df_merged["optimal_d"]) &
        (df_merged["pickup_threshold_min"] == df_merged["optimal_pickup_threshold_min"])
    )
    df_filtered = df_merged[mask].reset_index(drop=True).copy()
    df_filtered = df_filtered.drop(columns=["d", "pickup_threshold_min"])

    output_path = Path(root_dir) / "optimized_90_percent_fleet_size.csv"
    df_filtered.to_csv(output_path)

    return df_filtered 

def fleet_size_plots(df_results, plot_name, root_dir):
    # Add customer per minute column
    df_results['cust_per_min'] = df_results['perc_trip_filter'] * 106 # The 106 is the arrival rate of the full dataset
    dashed_style = "--"
    if plot_name == "PO2_vs_CAN":
        # Plot 1 configuration
        plot_settings = [
            {"algo": "PO2",     "ev_type": "Nissan_Leaf",           "charge_rate_kw": 20, "label": "Nissan Po2",      "linestyle": "solid"},
            {"algo": "PO2",     "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20, "label": "Tesla Po2",       "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20, "label": "Tesla CaN+Po2",   "linestyle": dashed_style},
            {"algo": "PO2",     "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 20, "label": "Mustang Po2",     "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 20, "label": "Mustang CaN+Po2", "linestyle": dashed_style},
            {"algo": "PO2",     "ev_type": "Waymo",                 "charge_rate_kw": 20, "label": "Hyundai Po2",     "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Waymo",                 "charge_rate_kw": 20, "label": "Hyundai CaN+Po2", "linestyle": dashed_style},
        ]
        title = "Po2 vs CaN+Po2 for 20kW Charge Rate"
        y_axis_title = "90% Fleet Size"
    elif plot_name == "CAN_PO2_charge_rate":
        # Plot 2 configuration
        plot_settings = [
            {"algo": "CAN_PO2", "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20, "label": "Tesla: 20kW",     "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Tesla_Model_3",         "charge_rate_kw": 100, "label": "Tesla: 100kW",   "linestyle": dashed_style},
            {"algo": "CAN_PO2", "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 20, "label": "Mustang: 20kW",   "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 100, "label": "Mustang: 100kW", "linestyle": dashed_style},
            {"algo": "CAN_PO2", "ev_type": "Waymo",                 "charge_rate_kw": 20, "label": "Hyundai: 20kW",   "linestyle": "solid"},
            {"algo": "CAN_PO2", "ev_type": "Waymo",                 "charge_rate_kw": 100, "label": "Hyundai: 100kW", "linestyle": dashed_style},
        ]
        title = "CaN+Po2 for 20kW vs 100kW Charge Rate"
        y_axis_title = "90% Fleet Size"
    elif plot_name == "POD_vs_CAN_R_20kW":
        # Plot 3 configuration
        list_label = []
        pod_filtered_results = df_results[(df_results["charge_rate_kw"] == 20) & (df_results["algo"] == "POD")]
        for ev_type in ["Nissan_Leaf", "Tesla_Model_3"]:
            if pod_filtered_results[pod_filtered_results["ev_type"] == ev_type]["optimal_d"].nunique() == 1:
                unique_d = pod_filtered_results[pod_filtered_results["ev_type"] == ev_type]["optimal_d"].iloc[0]
                if unique_d.is_integer():
                    unique_d = int(unique_d)
                if unique_d == 1:
                    list_label.append("CD")
                else:
                    list_label.append(f"Po{unique_d}")
            else:
                raise ValueError("A single value of d must be used")
        can_filtered_results = df_results[(df_results["charge_rate_kw"] == 20) & (df_results["algo"] == "CAN_R_POD")]
        for ev_type in ["Nissan_Leaf", "Tesla_Model_3", "Mustang_Mach_E_ER_AWD", "Waymo"]:
            if can_filtered_results[can_filtered_results["ev_type"] == ev_type]["optimal_d"].nunique() == 1:
                unique_d = can_filtered_results[can_filtered_results["ev_type"] == ev_type]["optimal_d"].iloc[0]
                if unique_d.is_integer():
                    unique_d = int(unique_d)
                if unique_d == 1:
                    list_label.append("CaN-R+CD")
                else:
                    list_label.append(f"CaN-R+Po{unique_d}")
            else:
                raise ValueError("A single value of d must be used")
        plot_settings = [
            {"algo": "POD",       "ev_type": "Nissan_Leaf",           "charge_rate_kw": 20, "label": f"Nissan: {list_label[0]}",  "linestyle": "solid"},
            {"algo": "CAN_R_POD", "ev_type": "Nissan_Leaf",           "charge_rate_kw": 20, "label": f"Nissan: {list_label[2]}",  "linestyle": dashed_style},
            {"algo": "POD",       "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20, "label": f"Tesla: {list_label[1]}",   "linestyle": "solid"},
            {"algo": "CAN_R_POD", "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20, "label": f"Tesla: {list_label[3]}",   "linestyle": dashed_style},
            {"algo": "CAN_R_POD", "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 20, "label": f"Mustang: {list_label[4]}", "linestyle": dashed_style},
            {"algo": "CAN_R_POD", "ev_type": "Waymo",                 "charge_rate_kw": 20, "label": f"Hyundai: {list_label[5]}", "linestyle": dashed_style},
        ]
        title = "Pod vs CaN-R+Pod for 20kW Charge Rate"
        y_axis_title = "90% Fleet Size"
    elif plot_name == "POD_vs_CAN_R_100kW":
        # Plot 4 configuration
        list_label = []
        pod_filtered_results = df_results[(df_results["charge_rate_kw"] == 20) & (df_results["algo"] == "POD")]
        for ev_type in ["Nissan_Leaf", "Tesla_Model_3"]:
            if pod_filtered_results[pod_filtered_results["ev_type"] == ev_type]["optimal_d"].nunique() == 1:
                unique_d = pod_filtered_results[pod_filtered_results["ev_type"] == ev_type]["optimal_d"].iloc[0]
                if unique_d.is_integer():
                    unique_d = int(unique_d)
                if unique_d == 1:
                    list_label.append("CD")
                else:
                    list_label.append(f"Po{unique_d}")
            else:
                raise ValueError("A single value of d must be used")
        can_filtered_results = df_results[(df_results["charge_rate_kw"] == 100) & (df_results["algo"] == "CAN_R_POD")]
        for ev_type in ["Nissan_Leaf", "Tesla_Model_3", "Mustang_Mach_E_ER_AWD", "Waymo"]:
            if can_filtered_results[can_filtered_results["ev_type"] == ev_type]["optimal_d"].nunique() == 1:
                unique_d = can_filtered_results[can_filtered_results["ev_type"] == ev_type]["optimal_d"].iloc[0]
                if unique_d.is_integer():
                    unique_d = int(unique_d)
                if unique_d == 1:
                    list_label.append("CaN-R+CD")
                else:
                    list_label.append(f"CaN-R+Po{unique_d}")            
            else:
                raise ValueError("A single value of d must be used")
        plot_settings = [
            {"algo": "POD",       "ev_type": "Nissan_Leaf",           "charge_rate_kw": 20,  "label": f"Nissan: {list_label[0]}, 20kW",   "linestyle": "solid"},
            {"algo": "CAN_R_POD", "ev_type": "Nissan_Leaf",           "charge_rate_kw": 100, "label": f"Nissan: {list_label[2]}, 100kW",  "linestyle": dashed_style},
            {"algo": "POD",       "ev_type": "Tesla_Model_3",         "charge_rate_kw": 20,  "label": f"Tesla: {list_label[1]}, 20kW",    "linestyle": "solid"},
            {"algo": "CAN_R_POD", "ev_type": "Tesla_Model_3",         "charge_rate_kw": 100, "label": f"Tesla: {list_label[3]}, 100kW",   "linestyle": dashed_style},
            {"algo": "CAN_R_POD", "ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 100, "label": f"Mustang: {list_label[4]}, 100kW", "linestyle": dashed_style},
            {"algo": "CAN_R_POD", "ev_type": "Waymo",                 "charge_rate_kw": 100, "label": f"Hyundai: {list_label[5]}, 100kW", "linestyle": dashed_style},
        ]
        title = "Pod (20kW) vs CaN-R+Pod (100kW)"
        y_axis_title = "90% Fleet Size"
    elif plot_name == "CAN_R_POD_N":
        # Plot 5 configuration
        # 1) pivot so each algo is its own column
        df_results_filtered = df_results[((df_results["algo"].isin(["CAN_R_POD", "CAN_R_POD_N"])) &
                                        (df_results["ev_type"].isin(["Mustang_Mach_E_ER_AWD", "Waymo"])))]
        df_pivot = df_results_filtered.pivot(
            index=["ev_type", "charge_rate_kw", "cust_per_min"],
            columns="algo",
            values="90_percent_fleet_size"
        )

        # 2) compute difference
        df_pivot["90_percent_fleet_size"] = (
            df_pivot["CAN_R_POD"] - df_pivot["CAN_R_POD_N"]
        )
        df_pivot = df_pivot.reset_index()
        plot_settings = [
            {"ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 20,  "label": f"Mustang: 20kW",  "linestyle": "solid"},
            {"ev_type": "Mustang_Mach_E_ER_AWD", "charge_rate_kw": 100, "label": f"Mustang: 100kW", "linestyle": dashed_style},
            {"ev_type": "Waymo",                 "charge_rate_kw": 20,  "label": f"Hyundai: 20kW",  "linestyle": "solid"},
            {"ev_type": "Waymo",                 "charge_rate_kw": 100, "label": f"Hyundai: 100kW", "linestyle": dashed_style},
        ]
        title = "90% Fleet Size Diff: CaN-R+Pod vs CaN-R+Pod-N"
        y_axis_title = "Improvement: 90% Fleet Size"
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Customers per Minute", fontsize=18)
    ax.set_ylabel(y_axis_title, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if plot_name != "CAN_R_POD_N":
        ax.set_ylim([df_results["90_percent_fleet_size"].min() - 50, df_results["90_percent_fleet_size"].max() + 50])
    ax.set_title(title, fontsize=18)
    inset_ylim_min = df_results["90_percent_fleet_size"].max() + 100
    inset_ylim_max = 0
    if plot_name in ["CAN_PO2_charge_rate", "POD_vs_CAN_R_100kW"]:
        # Add zoomed inset
        axins = inset_axes(ax, width="25%", height="30%", loc='lower right', borderpad=3)
    for setting in plot_settings:
        if plot_name != "CAN_R_POD_N":
            sel = df_results[
                (df_results['algo'] == setting['algo']) &
                (df_results['ev_type'] == setting['ev_type']) &
                (df_results['charge_rate_kw'] == setting['charge_rate_kw'])
            ].sort_values('cust_per_min')
        else:
            sel = df_pivot[
                (df_pivot['ev_type'] == setting['ev_type']) &
                (df_pivot['charge_rate_kw'] == setting['charge_rate_kw'])
            ].sort_values('cust_per_min')
        if setting["ev_type"] == "Nissan_Leaf":
            marker = "s"
            markersize = 8
            color = "#E6AB02"
        elif setting["ev_type"] == "Tesla_Model_3":
            marker = "o"
            markersize = 8
            color = "#377eb8"
        elif setting["ev_type"] == "Mustang_Mach_E_ER_AWD":
            marker = "*"
            markersize = 12
            color = "#ff7f00"
        elif setting["ev_type"] == "Waymo":
            marker = "p"
            markersize = 8
            color = "#9467BD"
        else:
            raise ValueError("No such EV type exists")
        if setting["linestyle"] == "solid":
            linewidth = 3
        elif setting["linestyle"] == dashed_style:
            linewidth = 4
        else:
            raise ValueError("linestyle should be either solid or dashed")
        ax.plot(sel['cust_per_min'], sel['90_percent_fleet_size'],
                  label=setting["label"], color=color,
                    linestyle=setting["linestyle"], linewidth=linewidth,
                      marker=marker, markersize=markersize)
        if plot_name in ["CAN_PO2_charge_rate", "POD_vs_CAN_R_100kW"]:
            axins.plot(sel['cust_per_min'], sel['90_percent_fleet_size'],
                        color=color, linewidth=linewidth,
                          linestyle=setting["linestyle"],
                            marker=marker, markersize=markersize)
        inset_ylim_min = min(inset_ylim_min, sel[sel["cust_per_min"] > 60]["90_percent_fleet_size"].min() - 50)
        inset_ylim_max = max(inset_ylim_max, sel[sel["cust_per_min"] > 60]["90_percent_fleet_size"].max() + 50)
    for line in plt.gca().get_lines():
        if line.get_linestyle() == "--":
            line.set_dashes((2, 2))  # now apply the denser dashes
    if plot_name in ["CAN_PO2_charge_rate", "POD_vs_CAN_R_100kW"]:
        # Set zoom range
        axins.set_xlim(60, 64.5)
        axins.set_ylim(inset_ylim_min, inset_ylim_max)

        # Hide tick labels 
        axins.tick_params(labelbottom=False)

        # Connect inset with main plot
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black")
    ax.legend(fontsize=11, handlelength=4)

    plot_filename = os.path.join(root_dir, f"fleet_{plot_name}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

def pod_varying_d_plots(df_results, plot_name, root_dir):
    list_color = ['#1F77B4', '#FC8D62', '#9467BD', '#E6AB02']
    list_marker = ["o", "s", "*", "D"]
    linewidth = 3
    list_markersize = [8, 8, 12, 8]
    list_linestyle = ["solid", "dashed", "dashdot", (0, (10, 5, 2, 5))]
    
    if plot_name in ["Nissan_pod_workload", "Tesla_pod_workload", "Nissan_pod_pickup", "Tesla_pod_pickup"]:
        if plot_name in ["Nissan_pod_workload", "Nissan_pod_pickup"]:
            df_results_filtered = df_results[
                    (df_results["algo"].isin(["POD", "PO2"])) & (df_results["ev_type"] == "Nissan_Leaf")
                    ].sort_values(by="d", ascending=True)
            plot_title = "Nissan Leaf"
        else:
            df_results_filtered = df_results[
                    (df_results["algo"].isin(["POD", "PO2"])) & (df_results["ev_type"] == "Tesla_Model_3")
                    ].sort_values(by="d", ascending=True)
            plot_title = "Tesla Model 3"
        list_label = ["30 min", "45 min", "60 min", r"$\infty$"]
        count = 0
        if plot_name in ["Nissan_pod_pickup", "Tesla_pod_pickup"]:
            column_name = "avg_pickup_time_min"
        else:
            column_name = "percentage_workload_served"
        for tp in [30, 45, 60, 0]:
            df_results_filtered2 = df_results_filtered[df_results_filtered["pickup_threshold_min"] == tp]
            plt.plot(df_results_filtered2["d"], df_results_filtered2[column_name],
                        label=list_label[count], color=list_color[count],
                        linewidth=linewidth, linestyle=list_linestyle[count],
                        marker=list_marker[count], markersize=list_markersize[count])
            count += 1
    elif plot_name == "POTP_workload":
        df_results_filtered = df_results[(df_results["algo"] == "POTP")].sort_values(by="pickup_threshold_min", ascending=True)
        list_label = ["Nissan", "Tesla", "Mustang"]
        count = 0
        plot_title = ""
        for ev in ["Nissan_Leaf", "Tesla_Model_3", "Mustang_Mach_E_ER_AWD"]:
            df_results_filtered2 = df_results_filtered[df_results_filtered["ev_type"] == ev]
            plt.plot(df_results_filtered2["pickup_threshold_min"], df_results_filtered2["percentage_workload_served"],
                        label=list_label[count], color=list_color[count],
                        linewidth=linewidth, linestyle=list_linestyle[count],
                        marker=list_marker[count], markersize=list_markersize[count])
            count += 1
    else:
        raise ValueError("No such plot type exists: try Nissan_pod Tesla_pod or POTP")

    plt.xlabel(r"$d$ in Power-of-$d$", fontsize=18)
    if plot_name in ["Nissan_pod_pickup", "Tesla_pod_pickup"]:
        plt.ylabel("Pickup Time (min)", fontsize=18)
    else:
        plt.ylabel("Workload %", fontsize=18)
    plt.title(plot_title, fontsize=18)
    if plot_name in ["Nissan_pod_workload", "Tesla_pod_workload"]:
        plt.legend(title = r"$T_{P, \max}$", fontsize=16, title_fontsize=18)
    elif plot_name == "POTP_workload":
        plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plot_filename = os.path.join(root_dir, f"pod_as_a_function_of_d_{plot_name}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.clf()

def contour_plots(df_results, plot_name, root_dir):
    df_results["n_chargers"] = df_results["n_chargers"] * 4 # as there are 4 posts
    # Define the data
    fleet_size = df_results["fleet_size"].unique()
    n_chargers = df_results["n_chargers"].unique()  
    if plot_name == "Nissan":
        df_results_filtered = df_results[
            df_results["ev_type"] == "Nissan_Leaf"
            ].sort_values(by=["n_chargers", "fleet_size"])
        plot_title = "Percentage Workload Served: Nissan Leaf"
    elif plot_name == "Tesla":
        df_results_filtered = df_results[
            df_results["ev_type"] == "Tesla_Model_3"
            ].sort_values(by=["n_chargers", "fleet_size"])
        plot_title = "Percentage Workload Served: Tesla Model 3"
    percentage_workload_served_1d = df_results_filtered["percentage_workload_served"].to_numpy()

    # Reshape the 1D array to 2D array
    percentage_workload_served = percentage_workload_served_1d.reshape(len(n_chargers), len(fleet_size))

    # Create a meshgrid
    X, Y = np.meshgrid(fleet_size, n_chargers)

    # Plot the contour
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    contour = plt.contourf(X, Y, percentage_workload_served, cmap='viridis')
    cbar = plt.colorbar(contour)
    plt.title(plot_title, fontsize=20)
    plt.ylabel("Number of Chargers", fontsize=18)
    plt.xlabel("Fleet Size", fontsize=20)
    plt.xticks(fleet_size, fontsize=18)
    plt.yticks(n_chargers, fontsize=18)
    cbar.ax.tick_params(labelsize=20)
    plt.grid(True)
    plot_filename = os.path.join(root_dir, f"contour_{plot_name}.png")
    plt.savefig(plot_filename, bbox_inches='tight')

def stackplot(root_dir, relocation_bool=None, plot_title=None, plot_dir=None, start_datetime=datetime(2022, 6, 14, 0, 0, 0)):
    if plot_dir == None:
        plot_dir = os.path.join(root_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    demand_curve_path = os.path.join(root_dir, "demand_curve")
    demand_curve_csv = os.path.join(demand_curve_path, "fleet_demand_curve.csv")
    df_demand_curve_data = pd.read_csv(demand_curve_csv)
    trips_in_progress_csv = os.path.join(root_dir, "trips_in_progress.csv")
    df_trips_in_progress = pd.read_csv(trips_in_progress_csv)
    if relocation_bool is None:
        if "n_cars_fake_charging" in df_demand_curve_data.columns:
            if df_demand_curve_data["n_cars_fake_charging"].max() > 0:
                relocation_bool = True
            else:
                relocation_bool = False
        else:
            relocation_bool = False
            df_demand_curve_data["relocating"] = 0
    # Stackplot of the state of the EVs with SoC overlaid
    if relocation_bool is True:
        df_demand_curve_data["actual_charging"] = df_demand_curve_data["charging"] - df_demand_curve_data["n_cars_fake_charging"]

        df_demand_curve_data["actual_driving_to_charger"] = df_demand_curve_data["driving_to_charger"] - df_demand_curve_data["n_cars_fake_driving_to_charger"]

        df_demand_curve_data["actual_idle"] = df_demand_curve_data["idle"] + df_demand_curve_data["n_cars_fake_charging"] + df_demand_curve_data["n_cars_fake_waiting_for_charger"]

        df_demand_curve_data["actual_waiting_for_charger"] = df_demand_curve_data["waiting_for_charger"] - df_demand_curve_data["n_cars_fake_waiting_for_charger"]

        filtered_demand_curve_data = df_demand_curve_data[[
        "driving_with_passenger", "driving_without_passenger", "actual_idle", "actual_driving_to_charger", 
        "n_cars_fake_driving_to_charger", "charging", "waiting_for_charger"]] # n_cars_fake_dirivng_to_charger are relocating
    else:
        filtered_demand_curve_data = df_demand_curve_data[[
        "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "relocating",
            "charging", "waiting_for_charger"]]
    # Approximately 12 ticks on the x-axis rounded off to the nearest 30 min
    sim_duration_min = np.ceil(df_demand_curve_data["time"].max())
    time_delta_for_ticks = int(sim_duration_min / 12) 
    time_delta_for_ticks = int(np.ceil(time_delta_for_ticks / 30) * 30) # to the nearest 30 min
    time_in_minutes = np.arange(0, sim_duration_min + time_delta_for_ticks, time_delta_for_ticks)
    time_of_day = [start_datetime + timedelta(minutes=int(minutes)) for minutes in time_in_minutes]
    time_of_day_str = [time.strftime('%H:%M') for time in time_of_day]
    n_cars = sum(filtered_demand_curve_data.iloc[1])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = df_demand_curve_data["time"].to_numpy()
    soc = df_demand_curve_data["avg_soc"].to_numpy()
    ax1.stackplot(x, np.transpose(filtered_demand_curve_data.to_numpy()), 
        colors=['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#8C564B', '#E6AB02', '#036c5f']
        )
    ax2.plot(x, soc, 'k', linewidth=2)
    ax1.set_xlabel("Time of the day", fontsize=20)
    ax1.set_ylabel("Number of Cars", fontsize=20)
    ax1.set_xticks(time_in_minutes, time_of_day_str, rotation=45, fontsize=14)
    ax1.set_ylim([0, n_cars])
    ax2.set_ylabel("SOC", fontsize=20)
    ax2.set_ylim([0, 1])
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.plot(df_trips_in_progress["time (min)"], df_trips_in_progress["number of trips"], color="#7B3F00", linewidth=2)
    if plot_title == None:
        plot_title = "Demand Stackplot with SOC overlaid"
    plt.title(plot_title, fontsize=20)
    demand_curve_plot_file = os.path.join(plot_dir, "demand_curve_stackplot.png")
    plt.savefig(demand_curve_plot_file, dpi=300, bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--root_dir', type=str, default=root_dir)
    parser.add_argument('-output', '--output_type', type=str, default="fleet_plot")
    parser.add_argument('-pt', '--plot_title', type=str, default="")
    args = parser.parse_args()
    if args.output_type == "fleet_plot":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        df_90_percent_fleet = compute_90_percent_fleet(kpi_consolidated, args.root_dir)
        df_results = optimize_fleet(df_90_percent_fleet, args.root_dir)
        fleet_size_plots(df_results, "PO2_vs_CAN", args.root_dir)
        fleet_size_plots(df_results, "CAN_PO2_charge_rate", args.root_dir)
        fleet_size_plots(df_results, "POD_vs_CAN_R_20kW", args.root_dir)
        fleet_size_plots(df_results, "POD_vs_CAN_R_100kW", args.root_dir)
        fleet_size_plots(df_results, "CAN_R_POD_N", args.root_dir)
    elif args.output_type == "policy_comparison_table":
        consolidate_the_kpis(args.root_dir)
    elif args.output_type == "pod_as_a_function_of_d_plot":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        pod_varying_d_plots(kpi_consolidated, "Nissan_pod_workload", args.root_dir)
        pod_varying_d_plots(kpi_consolidated, "Tesla_pod_workload", args.root_dir)
        pod_varying_d_plots(kpi_consolidated, "Nissan_pod_pickup", args.root_dir)
        pod_varying_d_plots(kpi_consolidated, "Tesla_pod_pickup", args.root_dir)
        pod_varying_d_plots(kpi_consolidated, "POTP_workload", args.root_dir)
    elif args.output_type == "contour_plot":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        contour_plots(kpi_consolidated, "Nissan", args.root_dir)
        contour_plots(kpi_consolidated, "Tesla", args.root_dir)
    elif args.output_type == "pod_ode_detailed_sims":
        kpi_consolidated = consolidate_the_kpis(args.root_dir)
        df_90_percent_fleet = compute_90_percent_fleet(kpi_consolidated, args.root_dir)
        df_results = optimize_fleet_ode(df_90_percent_fleet, args.root_dir)
    elif args.output_type == "stackplot":
        stackplot(args.root_dir, plot_title=args.plot_title)

