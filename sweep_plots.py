import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import os
from sim_metadata import SimMetaData
from sklearn.linear_model import LinearRegression


def contour_plot():
    pd_kpi = pd.read_csv("msom_sim_results/consolidated_kpi_pack_size_contour.csv")

    pd_kpi["total_n_chargers"] = pd_kpi["n_chargers"] * pd_kpi["n_posts"]

    what = "service_level_percentage_second_half"

    x = pd_kpi["fleet_size"]
    y = pd_kpi["pack_size_kwh"]
    z = pd_kpi[what]

    # Set up a regular grid of interpolation points
    x_new = pd_kpi["fleet_size"].drop_duplicates().sort_values()
    y_new = [4, 6, 8, 10, 15, 20]
    xi, yi = np.meshgrid(x_new, y_new)

    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)

    plt.tricontourf(x, y, z)
    plt.colorbar()
    #plt.tricontour(x, y, z, [90])
    plt.xlabel("Fleet Size")
    plt.ylabel("Battery Pack Size (kWh)")
    plt.title("Service Level Percentage")
    plt.savefig(f"msom_sim_results/contour_pack_size_{what}.eps")


def comparing_algos(plot):
    pd_kpi = pd.read_csv("msom_sim_results/consolidated_kpi_comparing_algos.csv")
    pd_kpi_filtered = pd_kpi.filter(items=["arrival_rate_pmin", "pack_size_kwh", "avg_pickup_time_min",
                                           "avg_drive_time_to_charger", "service_level_percentage_second_half",
                                           "matching_algorithm", "gamma", "beta"])
    x = pd_kpi_filtered["arrival_rate_pmin"].drop_duplicates().sort_values()
    gamma_0_6_mask = (pd_kpi_filtered["gamma"] == 0.6)
    gamma_0_65_mask = (pd_kpi_filtered["gamma"] == 0.65)
    beta_1_mask = (pd_kpi_filtered["beta"] == 1.0)
    pack_size_10_mask = (pd_kpi_filtered["pack_size_kwh"] == 10)
    pack_size_20_mask = (pd_kpi_filtered["pack_size_kwh"] == 20)
    pack_size_40_mask = (pd_kpi_filtered["pack_size_kwh"] == 40)
    closest_dispatch_mask = (pd_kpi_filtered["matching_algorithm"] == "Closest Dispatch")
    closest_available_dispatch_mask = (pd_kpi_filtered["matching_algorithm"] == "Closest Available Dispatch")
    power_of_2_mask = (pd_kpi_filtered["matching_algorithm"] == "Power-of-2")
    arrrival_rate_filter = (pd_kpi_filtered["arrival_rate_pmin"] >= 10)
    pd_service_level_10 = pd.pivot_table(pd_kpi_filtered[gamma_0_6_mask & beta_1_mask & pack_size_10_mask & arrrival_rate_filter],
                                         values="service_level_percentage_second_half",
                                         index="matching_algorithm",
                                         columns=["arrival_rate_pmin"]).T
    pd_service_level_20 = pd.pivot_table(pd_kpi_filtered[gamma_0_6_mask & beta_1_mask & pack_size_20_mask & arrrival_rate_filter],
                                         values="service_level_percentage_second_half",
                                         index="matching_algorithm",
                                         columns=["arrival_rate_pmin"]).T
    pickup_time_0_6_cd = pd_kpi_filtered[gamma_0_6_mask
                                         & beta_1_mask
                                         & pack_size_40_mask
                                         & closest_dispatch_mask
                                         ].sort_values(by="arrival_rate_pmin")[plot]
    slope_pickup_time_0_6_cd = LinearRegression().fit(np.log(x).values[Ellipsis, None],
                                                       np.log(pickup_time_0_6_cd)).coef_

    pickup_time_0_65_cd = pd_kpi_filtered[gamma_0_65_mask
                                          & pack_size_40_mask
                                          & closest_dispatch_mask
                                          ].sort_values(by="arrival_rate_pmin")[plot]

    reg = LinearRegression().fit(np.log(x).values[Ellipsis, None], np.log(pickup_time_0_65_cd))
    slope_pickup_time_0_65_cd = reg.coef_

    pickup_time_0_6_cad = pd_kpi_filtered[gamma_0_6_mask
                                          & beta_1_mask
                                          & pack_size_40_mask
                                          & closest_available_dispatch_mask
                                          ].sort_values(by="arrival_rate_pmin")[plot]

    slope_pickup_time_0_6_cad = LinearRegression().fit(np.log(x).values[Ellipsis, None], np.log(pickup_time_0_6_cad)).coef_

    pickup_time_0_65_cad = pd_kpi_filtered[gamma_0_65_mask
                                           & pack_size_40_mask
                                           & closest_available_dispatch_mask
                                           ].sort_values(by="arrival_rate_pmin")[plot]

    slope_pickup_time_0_65_cad = LinearRegression().fit(np.log(x).values[Ellipsis, None], np.log(pickup_time_0_65_cad)).coef_

    pickup_time_0_6_po2 = pd_kpi_filtered[gamma_0_6_mask
                                          & beta_1_mask
                                          & pack_size_40_mask
                                          & power_of_2_mask
                                          ].sort_values(by="arrival_rate_pmin")[plot]

    slope_pickup_time_0_6_po2 = LinearRegression().fit(np.log(x).values[Ellipsis, None], np.log(pickup_time_0_6_po2)).coef_

    pickup_time_0_65_po2 = pd_kpi_filtered[gamma_0_65_mask
                                           & pack_size_40_mask
                                           & power_of_2_mask
                                           ].sort_values(by="arrival_rate_pmin")[plot]

    slope_pickup_time_0_65_po2 = LinearRegression().fit(np.log(x).values[Ellipsis, None], np.log(pickup_time_0_65_po2)).coef_

    plt.plot(x,
             pickup_time_0_6_cad,
             "#377eb8",
             linewidth=2.5,
             label=rf'CAD, $\gamma=0.6, \beta=1$, fit={np.round(abs(slope_pickup_time_0_6_cad[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'CAD, $\gamma=0.6, \beta=1$',
             )
    plt.plot(x,
             pickup_time_0_65_cad,
             "#ff7f00",
             linestyle=(0, (5, 10)),
             linewidth=2.5,
             label=rf'CAD, $\gamma=0.65, \beta=0.8$, fit={np.round(abs(slope_pickup_time_0_65_cad[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'CAD, $\gamma=0.65, \beta=0.8$',
             )
    plt.plot(x,
             pickup_time_0_6_po2,
             "#4daf4a",
             linestyle="dashed",
             linewidth=2.5,
             label=rf'Po2, $\gamma=0.6, \beta=1$, fit={np.round(abs(slope_pickup_time_0_6_po2[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'PO2, $\gamma=0.6, \beta=1$',
             )
    plt.plot(x,
             pickup_time_0_65_po2,
             "#f781bf",
             linestyle="dashdot",
             linewidth=2.5,
             label=rf'Po2, $\gamma=0.65, \beta=0.8$, fit={np.round(abs(slope_pickup_time_0_65_po2[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'PO2, $\gamma=0.65, \beta=0.8$',
             )
    plt.plot(x,
             pickup_time_0_6_cd,
             "#984ea3",
             linestyle=(0, (3, 1, 1, 1)),
             linewidth=2.5,
             label=rf'CD, $\gamma=0.6, \beta=1$, fit={np.round(abs(slope_pickup_time_0_6_cd[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'CD, $\gamma=0.65, \beta=0.8$',
             )
    plt.plot(x,
             pickup_time_0_65_cd,
             "#a65628",
             linestyle=(5, (10, 3)),
             linewidth=2.5,
             label=rf'CD, $\gamma=0.65, \beta=0.8$, fit={np.round(abs(slope_pickup_time_0_65_cd[0]), 2)}'
             if plot != "service_level_percentage_second_half" else rf'CD, $\gamma=0.65, \beta=0.8$',
             )

    if plot == "avg_pickup_time_min":
        y_label = "Pickup Time (min)"
    elif plot == "avg_drive_time_to_charger":
        y_label = "Average Drive Time to Charger (min)"
    elif plot == "service_level_percentage_second_half":
        y_label = "Service Level (%)"
    plt.xlabel("Arrival Rate (per min)")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"msom_sim_results/{plot}.eps")
    plt.clf()


def stackplot(algorithm, csv_path, save_fig_name):
    if algorithm == "CAD":
        title = "Closest Available Dispatch"
    elif algorithm == "CD":
        title = "Closest Dispatch"
    elif algorithm == "PO2":
        title = "Power of 2"
    elif algorithm == "PO7":
        title = "Power of 7"
    else:
        raise ValueError("Wait, what algorithm?")
    workload = 160 * 0.5214 * 10 / 20 * 60
    plot_dir = "msom_sim_results/comparing_algos"
    df_demand_curve_data = pd.read_csv(csv_path)
    n_cars = int(df_demand_curve_data["idle"][0]
                 + df_demand_curve_data["driving_to_charger"][0]
                 + df_demand_curve_data["charging"][0]
                 + df_demand_curve_data["driving_with_passenger"][0]
                 + df_demand_curve_data["driving_without_passenger"][0]
                 + df_demand_curve_data["waiting_for_charger"][0]
                 )
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = df_demand_curve_data["time"].to_numpy()
    soc = df_demand_curve_data["avg_soc"].to_numpy()
    ax1.stackplot(x, np.transpose(df_demand_curve_data[[
        "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging",
        "waiting_for_charger"]].to_numpy()), colors=['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#E6AB02', '#036c5f'])
    ax2.plot(x, soc, 'k', linewidth=3)
    ax1.plot(x, np.ones(len(x)) * workload, "#7B3F00", linestyle="dashed", linewidth=3)
    ax1.set_xlabel("Time (min)", fontsize=18)
    ax1.set_ylabel("Number of Cars", fontsize=18)
    ax1.set_ylim([0, n_cars])
    ax1.set_xlim([0,1000])
    ax2.set_ylabel("SOC", fontsize=18)
    ax2.set_ylim([0, 1])
    plt.title(title, fontsize=22)
    demand_curve_plot_file = os.path.join(plot_dir, save_fig_name)
    plt.savefig(demand_curve_plot_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()


def soc_plot():
    csv_path_cad = "msom_sim_results/CAD_160_0.65_0.8_fleet_demand_curve.csv"
    csv_path_cd = "msom_sim_results/CD_160_0.65_0.8_fleet_demand_curve.csv"
    csv_path_po2 = "msom_sim_results/Po2_160_0.65_0.8_fleet_demand_curve.csv"
    plot_dir = "msom_sim_results"
    df_demand_curve_data_cad = pd.read_csv(csv_path_cad)
    df_demand_curve_data_cd = pd.read_csv(csv_path_cd)
    df_demand_curve_data_po2 = pd.read_csv(csv_path_po2)

    fig, ax = plt.subplots()
    x_cad = df_demand_curve_data_cad["time"].to_numpy()
    soc_cad = df_demand_curve_data_cad["avg_soc"].to_numpy()
    stdev_soc_cad = df_demand_curve_data_cad["stdev_soc"].to_numpy()

    x_cd = df_demand_curve_data_cd["time"].to_numpy()
    soc_cd = df_demand_curve_data_cd["avg_soc"].to_numpy()
    stdev_soc_cd = df_demand_curve_data_cd["stdev_soc"].to_numpy()

    x_po2 = df_demand_curve_data_cad["time"].to_numpy()
    soc_po2 = df_demand_curve_data_cd["avg_soc"].to_numpy()
    stdev_soc_po2 = df_demand_curve_data_po2["stdev_soc"].to_numpy()

    ax.plot(x_cad, soc_cad, '#1F77B4', linewidth=3)
    ax.plot(x_cd, soc_cd, '#FC8D62', linewidth=3)
    ax.plot(x_po2, soc_po2, '#98DF8A', linewidth=3)

    ax.fill_between(x_cad, (soc_cad - stdev_soc_cad), (soc_cad + stdev_soc_cad), color='#1F77B4', alpha=0.2)
    ax.fill_between(x_cd, (soc_cd - stdev_soc_cd), (soc_cd + stdev_soc_cd), color='#FC8D62', alpha=0.2)
    ax.fill_between(x_po2, (soc_po2 - stdev_soc_po2), (soc_po2 + stdev_soc_po2), color='#98DF8A', alpha=0.2)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("State of Charge")
    ax.set_ylim([0, 1])
    plt.title("Evolution of State of Charge with Standard Deviation Highlighted")
    demand_curve_plot_file = os.path.join(plot_dir, f"soc_evolution.png")
    plt.savefig(demand_curve_plot_file)
    plt.clf()


def asymptotic_plots(plot):
    list_arrival_rate = np.array([5, 10, 20, 40, 80, 160, 320])
    # list_buffer_beta_0_8_c_1 = np.array([89, 124, 181, 279, 383, 565, 884])
    # list_pickup_time_beta_0_8_c_1 = np.array([4.578, 3.386, 2.547, 1.848, 1.360, 1.001])
    # list_drive_to_charger_time_beta_0_8_c_1 = np.array([5.728, 4.338, 3.007, 2.096, 1.537, 1.172])
    #
    # list_buffer_beta_0_9_c_1 = np.array([64, 99, 142, 210, 294, 443, 627])
    # list_pickup_time_beta_0_9_c_1 = np.array([5.569, 4.241, 3.101, 2.262, 1.631, 1.207, 0.865])
    # list_drive_to_charger_time_beta_0_9_c_1 = np.array([5.694, 4.476, 3.047, 2.128, 1.524, 1.142, 0.813])
    # list_charger_buffer_0_9_c_1 = np.array([47, 86, 172, 320, 602, 1130, 2108])
    #
    # list_buffer_beta_1_c_1 = np.array([57, 79, 117, 170, 243, 340, 491])
    # list_pickup_time_beta_1_c_1 = np.array([5.352, 3.813, 2.833, 2.034, 1.487, 1.052, 0.760])
    # list_drive_to_charger_time_beta_1_c_1 = np.array([4.798, 3.213, 2.323, 1.580, 1.156, 0.803, 0.576])
    # list_charger_buffer_1_c_1 = np.array([71, 150, 299, 609, 1218, 2442, 4886])

    list_buffer_beta_1_c_4 = np.array([39, 59, 85, 128, 179, 248, 358])
    list_pickup_time_beta_1_c_4 = np.array([4.654, 3.446, 2.511, 1.811, 1.295, 0.929, 0.663])
    list_drive_to_charger_time_beta_1_c_4 = np.array([2.350, 1.688, 1.221, 0.860, 0.605, 0.425, 0.299])
    list_charger_buffer_1_c_4 = np.array([303, 606, 1212, 2425, 4850, 9707, 19422])

    slope_buffer_beta_1_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_buffer_beta_1_c_4)
    ).coef_[0], 3)
    slope_pickup_time_beta_1_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_pickup_time_beta_1_c_4)
    ).coef_[0], 3)
    slope_drive_to_charger_time_beta_1_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_drive_to_charger_time_beta_1_c_4)
    ).coef_[0], 3)
    slope_charger_buffer_1_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_charger_buffer_1_c_4)
    ).coef_[0], 3)

    list_buffer_beta_0_9_c_4 = np.array([44, 64, 94, 145, 205, 291, 429])
    list_pickup_time_beta_0_9_c_4 = np.array([4.813, 3.526, 2.584, 1.889, 1.362, 0.982, 0.710])
    list_drive_to_charger_time_beta_0_9_c_4 = np.array([3.032, 2.155, 1.559, 1.169, 0.834, 0.613, 0.450])
    list_charger_buffer_0_9_c_4 = np.array([191, 366, 683, 1281, 2384, 4466, 8333])

    slope_buffer_beta_0_9_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_buffer_beta_0_9_c_4)
    ).coef_[0], 3)
    slope_pickup_time_beta_0_9_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_pickup_time_beta_0_9_c_4)
    ).coef_[0], 3)
    slope_drive_to_charger_time_beta_0_9_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_drive_to_charger_time_beta_0_9_c_4)
    ).coef_[0], 3)
    slope_charger_buffer_0_9_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_charger_buffer_0_9_c_4)
    ).coef_[0], 3)

    list_buffer_beta_0_8_c_4 = np.array([47, 74, 108, 170, 246, 357, 533])
    list_pickup_time_beta_0_8_c_4 = np.array([5.022, 3.729, 2.729, 2.025, 1.497, 1.083, 0.792])
    list_drive_to_charger_time_beta_0_8_c_4 = np.array([3.541, 2.806, 2.045, 1.555, 1.178, 0.866, 0.653])
    list_charger_buffer_beta_0_8_c_4 = np.array([127, 222, 388, 672, 1177, 2057, 3586])

    slope_buffer_beta_0_8_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_buffer_beta_0_8_c_4)
    ).coef_[0], 3)
    slope_pickup_time_beta_0_8_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_pickup_time_beta_0_8_c_4)
    ).coef_[0], 3)
    slope_drive_to_charger_time_beta_0_8_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_drive_to_charger_time_beta_0_8_c_4)
    ).coef_[0], 3)
    slope_charger_buffer_0_8_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_charger_buffer_beta_0_8_c_4)
    ).coef_[0], 3)

    list_buffer_beta_0_7_c_4 = np.array([55, 83, 127, 205, 296, 473, 698])
    list_pickup_time_beta_0_7_c_4 = np.array([5.337, 3.832, 2.933, 2.248, 1.641, 1.249, 0.909])
    list_drive_to_charger_time_beta_0_7_c_4 = np.array([4.629, 3.418, 2.620, 2.072, 1.542, 1.229, 0.917])
    list_charger_buffer_beta_0_7_c_4 = np.array([83, 142, 235, 383, 645, 1074, 1653])

    slope_buffer_beta_0_7_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate[1:]).reshape(-1, 1),
        np.log(list_buffer_beta_0_7_c_4[1:])
    ).coef_[0], 3)
    slope_pickup_time_beta_0_7_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_pickup_time_beta_0_7_c_4)
    ).coef_[0], 3)
    slope_drive_to_charger_time_beta_0_7_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_drive_to_charger_time_beta_0_7_c_4)
    ).coef_[0], 3)
    slope_charger_buffer_0_7_c_4 = np.round(LinearRegression().fit(
        np.log(list_arrival_rate).reshape(-1, 1),
        np.log(list_charger_buffer_beta_0_7_c_4)
    ).coef_[0], 3)

    # list_charger_buffer_0_8 = np.array([30, 53, 99, 166, 295, 529, 922])
    # slope_charger_buffer_0_8 = LinearRegression().fit(
    #     np.log(list_arrival_rate[1:]).reshape(-1, 1), np.log(list_charger_buffer_0_8[1:])
    #                                             ).coef_
    if plot == "fleet_size":
        y1 = list_buffer_beta_1_c_4
        y2 = list_buffer_beta_0_9_c_4
        y3 = list_buffer_beta_0_8_c_4
        y4 = list_buffer_beta_0_7_c_4

        s1 = slope_buffer_beta_1_c_4
        s2 = slope_buffer_beta_0_9_c_4
        s3 = slope_buffer_beta_0_8_c_4
        s4 = slope_buffer_beta_0_7_c_4

        y_label = r"Fleet Size Buffer"
    elif plot == "num_chargers":
        y1 = list_charger_buffer_1_c_4
        y2 = list_charger_buffer_0_9_c_4
        y3 = list_charger_buffer_beta_0_8_c_4
        y4 = list_charger_buffer_beta_0_7_c_4

        s1 = slope_charger_buffer_1_c_4
        s2 = slope_charger_buffer_0_9_c_4
        s3 = slope_charger_buffer_0_8_c_4
        s4 = slope_charger_buffer_0_7_c_4

        y_label = r"$m - r\tilde{T}_R \alpha \lambda$"
    elif plot == "pickup_time":
        y1 = list_pickup_time_beta_1_c_4
        y2 = list_pickup_time_beta_0_9_c_4
        y3 = list_pickup_time_beta_0_8_c_4
        y4 = list_pickup_time_beta_0_7_c_4

        s1 = slope_pickup_time_beta_1_c_4
        s2 = slope_pickup_time_beta_0_9_c_4
        s3 = slope_pickup_time_beta_0_8_c_4
        s4 = slope_pickup_time_beta_0_7_c_4

        y_label = "Average Pickup Time (min)"
    elif plot == "drive_to_charger_time":
        y1 = list_drive_to_charger_time_beta_1_c_4
        y2 = list_drive_to_charger_time_beta_0_9_c_4
        y3 = list_drive_to_charger_time_beta_0_8_c_4
        y4 = list_drive_to_charger_time_beta_0_7_c_4

        s1 = slope_drive_to_charger_time_beta_1_c_4
        s2 = slope_drive_to_charger_time_beta_0_9_c_4
        s3 = slope_drive_to_charger_time_beta_0_8_c_4
        s4 = slope_drive_to_charger_time_beta_0_7_c_4

        y_label = "Average Drive to Charger Time (min)"
    else:
        raise ValueError("What plot dude?")
    fig, ax = plt.subplots()
    ax.plot(list_arrival_rate, y1,
            "#377eb8",
            linewidth=3,
            markersize=8,
            label=f'A, fit={s1}',
            marker="o",
            )
    ax.plot(list_arrival_rate, y2,
            "#ff7f00",
            linewidth=3,
            markersize=8,
            label=f'B, fit={s2}',
            marker="s",
            linestyle="dashed"
            )
    ax.plot(list_arrival_rate, y3,
            "#9467BD",
            linewidth=3,
            markersize=12,
            label=f'C, fit={s3}',
            marker="*",
            linestyle="dashdot"
            )
    ax.plot(list_arrival_rate, y4,
            "#E6AB02",
            linewidth=3,
            markersize=8,
            label=f'D, fit={s4}',
            marker="D",
            linestyle="dotted"
            )
    ax.legend(fontsize=16, loc=2)
    plt.xlabel("Arrival Rate (per min)", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.ylim([0, 715])
    # plt.show()
    plt.savefig(f"job_talk_plots/scaling_result_{plot}.png")
    plt.clf()


def comparing_algos_data_based(plot):
    x = np.array([5, 10, 20, 40, 80, 160, 320])

    cad_service_level_beta_1 = np.array([88.12, 88.60, 89.17, 89.63, 90.13, 90.65, 91.05])
    cd_service_level_beta_1 = np.array([89.75, 89.59, 89.49, 89.38, 89.16, 89.14, 89.07])
    po2_service_level_beta_1 = np.array([89.78, 89.95, 90.02, 90.03, 90.07, 90.13, 90.03])

    cad_pickup_time_beta_1 = np.array([5.723, 4.417, 3.459, 2.631, 1.995, 1.510, 1.147])
    cd_pickup_time_beta_1 = np.array([4.076, 2.985, 2.227, 1.606, 1.160, 0.825, 0.590])
    po2_pickup_time_beta_1 = np.array([4.654, 3.446, 2.511, 1.811, 1.295, 0.929, 0.663])

    cad_drive_to_charger_time_beta_1 = np.array([2.343, 1.687, 1.217, 0.858, 0.604, 0.424, 0.299])
    cd_drive_to_charger_time_beta_1 = np.array([2.350, 1.693, 1.223, 0.860, 0.606, 0.425, 0.299])
    po2_drive_to_charger_time_beta_1 = np.array([2.351, 1.692, 1.223, 0.859, 0.605, 0.425, 0.299])

    cad_profit_beta_1 = np.array([-1337.46, -2675.15, -5072.25, -9330.04, -15699.77, -27286.52, -44143.59])
    cd_profit_beta_1 = np.array([243.63, 304.51, 321.90, 50.22, -937.06, -3174.26, -6446.88])
    # po2_profit_beta_1 = np.array([0.18, 0.36, 0.74, 1.50, 3.00, 6.04, 12.11])

    cad_service_level_beta_0_8 = np.array([88.21, 88.70, 89.41, 90.09, 90.17, 90.83, 91.22])
    cd_service_level_beta_0_8 = np.array([88.71, 88.49, 88.89, 88.93, 88.74, 88.83, 88.88])
    po2_service_level_beta_0_8 = np.array([90.07, 89.99, 90.24, 90.07, 89.92, 90.06, 90.01])

    cad_pickup_time_beta_0_8 = np.array([6.001, 4.799, 3.657, 2.812, 2.175, 1.641, 1.252])
    cd_pickup_time_beta_0_8 = np.array([4.343, 3.400, 2.538, 1.906, 1.412, 1.023, 0.755])
    po2_pickup_time_beta_0_8 = np.array([5.022, 3.729, 2.729, 2.025, 1.497, 1.083, 0.792])

    cad_drive_to_charger_time_beta_0_8 = np.array([3.485, 2.788, 2.027, 1.532, 1.163, 0.856, 0.644])
    cd_drive_to_charger_time_beta_0_8 = np.array([3.537, 2.818, 2.050, 1.561, 1.184, 0.870, 0.655])
    po2_drive_to_charger_time_beta_0_8 = np.array([3.528, 2.806, 2.045, 1.555, 1.178, 0.866, 0.652])

    cad_profit_beta_0_8 = np.array([-1355.63, -2806.73, -5074.25, -8352.58, -15440.98, -25679.65, -41720.97])
    cd_profit_beta_0_8 = np.array([-9.43, -260.27, -640.59, -858.15, -2359.71, -5752.13, -10794.15])
    # po2_profit_beta_0_8 = np.array([0.17, 0.36, 0.73, 1.48, 2.97, 6.00, 12.06])

    cad_workload_beta_1 = np.array([82.32, 81.67, 82.58, 82.31, 82.81, 83.43, 83.89])
    cd_workload_beta_1 = np.array([87.96, 87.01, 87.45, 86.53, 86.10, 86.11, 85.96])
    po2_workload_beta_1 = np.array([87.44, 86.80, 87.40, 86.73, 86.52, 86.68, 86.50])

    cad_workload_beta_0_8 = np.array([82.51, 81.79, 82.93, 82.97, 82.86, 83.73, 84.16])
    cd_workload_beta_0_8 = np.array([87.03, 86.14, 86.81, 86.23, 85.75, 85.88, 85.81])
    po2_workload_beta_0_8 = np.array([87.88, 87.16, 87.79, 86.87, 86.53, 86.78, 86.62])

    if plot == "service_level":
        y1 = cad_service_level_beta_1
        y2 = cd_service_level_beta_1
        y3 = po2_service_level_beta_1
        y4 = cad_service_level_beta_0_8
        y5 = cd_service_level_beta_0_8
        y6 = po2_service_level_beta_0_8

        y_label = "Service Level Percentage"
    elif plot == "pickup_time":
        y1 = cad_pickup_time_beta_1
        y2 = cd_pickup_time_beta_1
        y3 = po2_pickup_time_beta_1
        y4 = cad_pickup_time_beta_0_8
        y5 = cd_pickup_time_beta_0_8
        y6 = po2_pickup_time_beta_0_8

        y_label = "Pickup Time (min)"
    elif plot == "drive_to_charger_time":
        y1 = cad_drive_to_charger_time_beta_1
        y2 = cd_drive_to_charger_time_beta_1
        y3 = po2_drive_to_charger_time_beta_1
        y4 = cad_drive_to_charger_time_beta_0_8
        y5 = cd_drive_to_charger_time_beta_0_8
        y6 = po2_drive_to_charger_time_beta_0_8

        y_label = "Drive to Charger Time (min)"
    elif plot == "profit":
        y1 = cad_profit_beta_1 / 500
        y2 = cd_profit_beta_1 / 500
        y4 = cad_profit_beta_0_8 / 500
        y5 = cd_profit_beta_0_8 / 500

        y_label = r"$\Delta$ Profit per min $(\$)$"
    elif plot == "workload":
        y1 = cad_workload_beta_1
        y2 = cd_workload_beta_1
        y3 = po2_workload_beta_1
        y4 = cad_workload_beta_0_8
        y5 = cd_workload_beta_0_8
        y6 = po2_workload_beta_0_8

        y_label = "Percentage Workload Served"
    else:
        raise ValueError("Wait, What Plot?")
    fig, ax = plt.subplots()
    ax.plot(x, y1,
            "#377eb8",
            linewidth=3,
            markersize=8,
            label=f'CAD, Series A',
            marker="o",
            linestyle="dashdot"
            )
    ax.plot(x, y2,
            "#ff7f00",
            linewidth=3,
            markersize=8,
            label=f'CD, Series A',
            marker="s",
            linestyle="dashed"
            )
    if not plot == "profit":
        ax.plot(x, y3,
                "#9467BD",
                linewidth=3,
                markersize=12,
                label=f'PO2, Series A',
                marker="*"
                )
    ax.plot(x, y4,
            "#E6AB02",
            linewidth=3,
            markersize=8,
            label=f'CAD, Series C',
            marker="D",
            linestyle="dashdot"
            )
    ax.plot(x, y5,
            "#2CA02C",
            linewidth=3,
            markersize=8,
            label=f'CD, Series C',
            marker="p",
            linestyle="dashed"
            )
    if not plot == "profit":
        ax.plot(x, y6,
                "#7B3F00",
                linewidth=3,
                markersize=8,
                label=f'PO2 Series C',
                marker="h"
                )
    # ax.legend()
    plt.xlabel("Arrival Rate (per min)", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    # plt.show()
    plt.savefig(f"msom_sim_results/comparing_algos/{plot}.pgf", bbox_inches='tight', pad_inches=0.05)
    plt.clf()


def power_of_d(plot):

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    if plot == "service_level":
        y5 = np.array([81.29, 86.02, 87.98, 89.03, 89.49, 89.58, 89.70, 89.66, 89.59, 89.42])
        y10 = np.array([86.41, 89.11, 89.83, 90.02, 90.00, 89.91, 89.70, 89.49, 89.30, 89.07])
        y20 = np.array([88.70, 90.03, 90.12, 90.04, 89.93, 89.72, 89.57, 89.37, 89.10, 88.84])
        y40 = np.array([88.83, 90.06, 90.25, 90.22, 90.04, 89.89, 89.64, 89.37, 89.16, 88.83])

        y_label = "Service Level (%)"
        label5 = rf'$p^\star = 5, d_{{\max}} = {np.argmax(y5) + 1}$'
        label10 = rf'$p^\star = 10, d_{{\max}} = {np.argmax(y10) + 1}$'
        label20 = rf'$p^\star = 20, d_{{\max}} = {np.argmax(y20) + 1}$'
        label40 = rf'$p^\star = 40, d_{{\max}} = {np.argmax(y40) + 1}$'
    elif plot == "pickup_time":
        y5 = np.array([0.932, 0.995, 1.063, 1.132, 1.207, 1.283, 1.357, 1.429, 1.500, 1.566])
        y10 = np.array([0.980, 1.051, 1.115, 1.182, 1.250, 1.323, 1.392, 1.462, 1.532, 1.596])
        y20 = np.array([1.013, 1.078, 1.134, 1.193, 1.259, 1.328, 1.400, 1.474, 1.547, 1.614])
        y40 = np.array([1.023, 1.083, 1.140, 1.203, 1.274, 1.351, 1.421, 1.500, 1.575, 1.641])

        y_label = "Pickup Time (min)"
        label5 = r'$p^\star = 5$'
        label10 = r'$p^\star = 10$'
        label20 = r'$p^\star = 20$'
        label40 = r'$p^\star = 40$'
    else:
        raise ValueError("Wait, what plot?")
    fig, ax = plt.subplots()
    ax.plot(x, y5,
            "#377eb8",
            linewidth=3,
            markersize=5,
            label=label5,
            marker="o"
            )
    ax.plot(x, y10,
            "#ff7f00",
            linewidth=3,
            markersize=5,
            label=label10,
            marker="s",
            linestyle="dashed"
            )
    ax.plot(x, y20,
            "#9467BD",
            linewidth=3,
            markersize=7,
            label=label20,
            marker="*",
            linestyle="dashdot"
            )
    ax.plot(x, y40,
            "#E6AB02",
            linewidth=3,
            markersize=5,
            label=label40,
            marker="D",
            linestyle="dotted"
            )
    plt.xlabel(r"$d$ in Power-of-$d$", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(fontsize=20)
    plt.savefig(f"msom_sim_results/power_of_d/power_of_d_{plot}.pgf", bbox_inches='tight', pad_inches=0.05)
    plt.clf()


def fleet_size_calculation(beta):
    if beta == 1:
        x_dict = {
            "10": np.array([86.17, 88.10, 89.63, 91.11, 92.65]),
            "20": np.array([86.38, 88.20, 89.09, 90.03, 91.54]),
            "40": np.array([88.25, 89.09, 89.55, 89.95, 90.77]),
            "80": np.array([88.16, 88.98, 89.87, 90.78, 91.61]),
            "160": np.array([88.83, 89.24, 89.72, 90.09, 90.60])
        }
        y_dict = {
            "10": np.array([49.06, 53.52, 57.45, 61.80, 65.70]),
            "20": np.array([68.61, 76.50, 80.48, 84.61, 92.45]),
            "40": np.array([111.72, 119.46, 123.19, 127.29, 135.07]),
            "80": np.array([145.24, 161.29, 176.72, 192.01, 207.37]),
            "160": np.array([207.79, 222.85, 237.07, 252.51, 268.43])
        }
    elif beta == 0.8:
        x_dict = {
            "10": np.array([83.90, 85.61, 88.78, 89.99, 91.75]),
            "20": np.array([86.82, 88.65, 90.24, 91.52, 93.31]),
            "40": np.array([88.25, 89.50, 90.68, 91.99, 93.15]),
            "80": np.array([89.43, 89.78, 90.22, 90.67, 91.16]),
            "160": np.array([89.80, 90.10, 90.28, 90.50, 90.87]),
        }
        y_dict = {
            "10": np.array([57.73, 61.87, 70.26, 74.58, 78.68]),
            "20": np.array([92.65, 100.53, 108.60, 116.92, 124.88]),
            "40": np.array([152.82, 165.13, 176.61, 189.19, 202.05]),
            "80": np.array([234.45, 243.47, 251.16, 258.74, 266.00]),
            "160": np.array([349.97, 361.55, 370.30, 378.08, 389.84]),
        }
    else:
        raise ValueError("Do not have data for that beta")
    list_marker = ["o", "s", "*", "D", "p"]
    list_color = ['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#E6AB02']
    fig, ax = plt.subplots()
    count = 0
    x_min = x_dict["10"].min() - 0.5
    ylim = y_dict["160"].max() + 10
    for lamb in [10, 20, 40, 80, 160]:
        linear_regressor = LinearRegression()
        linear_regressor.fit(x_dict[f"{lamb}"].reshape(-1, 1), y_dict[f"{lamb}"])
        x = np.linspace(x_dict[f"{lamb}"].min(), x_dict[f"{lamb}"].max(), num=200).reshape(-1, 1)
        y = linear_regressor.predict(x)
        ax.scatter(x_dict[f"{lamb}"],
                   y_dict[f"{lamb}"],
                   zorder=10,
                   marker=list_marker[count],
                   s=60,
                   c=list_color[count],
                   label=rf"$\lambda$ = {lamb}")
        ax.plot(x, y, color=list_color[count], lw=3, ls="dashed")
        fleet_size_90_percent = linear_regressor.predict(np.array([90]).reshape(-1, 1))[0]
        ax.plot(np.linspace(x_min, 90, num=100),
                np.ones(100) * fleet_size_90_percent,
                ls="dotted",
                color=list_color[count],
                lw=2)
        count = count + 1
    ax.set_ylim([0, ylim])
    ax.set_xlim([x_min, 93.5])
    ax.plot(np.ones(100) * 90, np.linspace(0, fleet_size_90_percent, num=100), color="black")
    plt.xlabel("Service Level Percentage", fontsize=18)
    plt.ylabel(r"$n - (1+r)\alpha \lambda \tilde{T}_R$", fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig(f"msom_sim_results/calculating_90_percent_fleet_size_beta_{beta}.pgf", bbox_inches='tight', pad_inches=0.05)
    plt.clf()


if __name__ == "__main__":
    # stackplot(algorithm="CAD")
    # stackplot(algorithm="CD")
    # stackplot(algorithm="PO2")
    asymptotic_plots(plot="fleet_size")
    # asymptotic_plots(plot="num_chargers")
    # asymptotic_plots(plot="pickup_time")
    # asymptotic_plots(plot="drive_to_charger_time")
    # comparing_algos_data_based(plot="workload")
    # comparing_algos_data_based(plot="service_level")
    # comparing_algos_data_based(plot="pickup_time")
    # comparing_algos_data_based(plot="drive_to_charger_time")
    # comparing_algos_data_based(plot="profit")
    # stackplot(algorithm="CD",
    #           csv_path="msom_sim_results/comparing_algos/CD_beta_0_8_lambda_160_fleet_demand_curve.csv",
    #           save_fig_name="CD_stackplot.pgf")
    # stackplot(algorithm="PO2",
    #           csv_path="msom_sim_results/comparing_algos/Po2_beta_0_8_lambda_160_fleet_demand_curve.csv",
    #           save_fig_name="Po2_stackplot.pgf")
    # stackplot(algorithm="CAD",
    #           csv_path="msom_sim_results/comparing_algos/CAD_beta_0_8_lambda_160_fleet_demand_curve.csv",
    #           save_fig_name="CAD_stackplot.pgf")
    # stackplot(algorithm="CD",
    #           csv_path="msom_sim_results/comparing_algos/CD_low_pack_size_fleet_demand_curve.csv",
    #           save_fig_name="CD_low_pack_size_stackplot.pgf")
    # stackplot(algorithm="PO7",
    #           csv_path="msom_sim_results/comparing_algos/Po7_low_pack_size_fleet_demand_curve.csv",
    #           save_fig_name="Po7_stackplot.pgf")
    # power_of_d(plot="service_level")
    # power_of_d(plot="pickup_time")
    fleet_size_calculation(beta=0.8)
