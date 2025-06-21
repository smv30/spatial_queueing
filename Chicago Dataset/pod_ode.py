import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sim_metadata import SimMetaData, Dataset, DistFunc, DatasetParams
from real_life_data_input import DataInput
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import argparse
from pathlib import Path
from post_processing import consolidate_the_kpis
from sklearn.linear_model import LinearRegression


def computing_spatial_functions(dir_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfs = []
    for filepath in Path(dir_name).rglob("pickup_time_vs_available_cars.csv"):
        try:
            df_pickup = pd.read_csv(filepath)
            dfs.append(df_pickup)
        except FileNotFoundError:
            # shouldn't happen since rglob found it, but just in case
            continue
    df_pickup_consolidated = pd.concat(dfs, ignore_index=True)
    df_pickup_consolidated = df_pickup_consolidated[["n_available_cars", "mean_pickup_min", "std_pickup_min", "count_datapoints"]]
    df_pickup_consolidated = df_pickup_consolidated.dropna()
    results_file = os.path.join(output_dir, "consolidated_pickup_time_vs_available_cars.csv")
    df_pickup_consolidated.to_csv(results_file)

    df_pickup_consolidated["total_pickup_time"] = df_pickup_consolidated["mean_pickup_min"] * df_pickup_consolidated[
        "count_datapoints"]
    
    n_bins = 20
    len_each_bin_pickup = int(df_pickup_consolidated["n_available_cars"].max() / n_bins)
    # Scatter of available cars versus pickup time
    bins = [len_each_bin_pickup * i for i in range(n_bins)]
    list_mean_pickup_min = []
    n_available_cars = df_pickup_consolidated["n_available_cars"].to_numpy()
    total_pickup_time = df_pickup_consolidated["total_pickup_time"].to_numpy()
    count_datapoints = df_pickup_consolidated["count_datapoints"].to_numpy()
    for i in range(len(bins) - 1):
        indices = [idx for idx, val in enumerate(n_available_cars) if bins[i] <= val < bins[i + 1]]

        total_pickup_time_min = np.mean([total_pickup_time[idx] for idx in indices])
        total_count = np.mean([count_datapoints[idx] for idx in indices])
        list_mean_pickup_min.append(np.round(total_pickup_time_min / total_count, 2))

    first_idx_non_monotonic = next(
    (i for i in range(len(list_mean_pickup_min) - 1)
     if list_mean_pickup_min[i+1] > list_mean_pickup_min[i]),
    n_bins - 1
    ) # First index where pickup time becomes non-monotonic

    # Define the power-law function
    def power_law(x, a, b):
        return a * x ** b

    # Convert lists to numpy arrays
    x_data = np.array([int(len_each_bin_pickup / 2) + len_each_bin_pickup * i for i in range(first_idx_non_monotonic)])
    y_data = np.array(list_mean_pickup_min[0:first_idx_non_monotonic])

    # Perform the curve fitting
    params, covariance = curve_fit(power_law, x_data, y_data)

    # Get the parameters
    pickup_multiplier, pickup_exponent = params
    pickup_multiplier = round(pickup_multiplier, 2)
    pickup_exponent = round(pickup_exponent, 2)
    print(f"The pickup time power law fit: {pickup_multiplier} * x^{pickup_exponent}")

    # Generate the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), len_each_bin_pickup)
    y_fit = power_law(x_fit, pickup_multiplier, pickup_exponent)

    plt.scatter([int(len_each_bin_pickup / 2) + len_each_bin_pickup * i for i in range(n_bins - 1)], list_mean_pickup_min, linestyle='None', marker='^')
    plt.plot(x_fit, y_fit, 'r-', label='Fit: {:.2f} * x^({:.2f})'.format(pickup_multiplier, pickup_exponent))
    plt.xlabel("n_available_cars")
    plt.ylabel("pickup_time_min")
    plot_file = os.path.join(output_dir, "plot_pickup_time_vs_available_cars.png")
    plt.savefig(plot_file)
    plt.clf()

    # Drive to charger time
    dfs = []
    for filepath in Path(dir_name).rglob("drive_to_charger_time_vs_available_posts_with_driving_cars.csv"):
        try:
            df_charger = pd.read_csv(filepath)
            dfs.append(df_charger)
        except FileNotFoundError:
            # shouldn't happen since rglob found it, but just in case
            continue
    df_charger_consolidated = pd.concat(dfs, ignore_index=True)

    results_file = os.path.join(output_dir, "consolidated_charger_time_vs_available_cars.csv")
    df_charger_consolidated.to_csv(results_file)

    df_charger_consolidated["total_charger_time"] = df_charger_consolidated["mean_drive_to_charger_min"] * \
                                                    df_charger_consolidated["count_datapoints"]
    df_charger_consolidated = df_charger_consolidated.dropna()

    # Scatter of available chargers versus drive to the charger time
    len_each_bin_charger = int(df_charger_consolidated["n_available_chargers"].max() / n_bins)

    bins = [len_each_bin_charger * i for i in range(n_bins)]
    list_mean_charger_min = []
    n_available_chargers = df_charger_consolidated["n_available_chargers"].to_numpy()
    total_charger_time = df_charger_consolidated["total_charger_time"].to_numpy()
    count_datapoints = df_charger_consolidated["count_datapoints"].to_numpy()
    for i in range(len(bins) - 1):
        indices = [idx for idx, val in enumerate(n_available_chargers) if bins[i] <= val < bins[i + 1]]

        total_charger_time_min = np.mean([total_charger_time[idx] for idx in indices])
        total_count = np.mean([count_datapoints[idx] for idx in indices])
        list_mean_charger_min.append(np.round(total_charger_time_min / total_count, 2))
    for i in range(1, len(list_mean_charger_min)): # Fill all the nan values, if any
        if np.isnan(list_mean_charger_min[i]):
            list_mean_charger_min[i] = list_mean_charger_min[i-1]
    # Define the power-law function
    def power_law(x, a, b):
        return a * x ** b
    
    first_idx_non_monotonic = next(
    (i for i in range(len(list_mean_charger_min) - 1)
     if list_mean_charger_min[i+1] > list_mean_charger_min[i]),
    n_bins - 1
    ) # First index where pickup time becomes non-monotonic

    # Convert lists to numpy arrays
    x_data = np.array([int(len_each_bin_charger / 2) + len_each_bin_charger * i for i in range(first_idx_non_monotonic)])
    y_data = np.array(list_mean_charger_min[0:first_idx_non_monotonic])

    # Perform the curve fitting
    params, covariance = curve_fit(power_law, x_data, y_data)

    # Get the parameters
    charger_multiplier, charger_exponent = params
    charger_multiplier = round(charger_multiplier, 2)
    charger_exponent = round(charger_exponent, 2)
    print(f"The drive to the charger time power law fit: {charger_multiplier} * x^{charger_exponent}")

    # Generate the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), len_each_bin_charger)
    y_fit = power_law(x_fit, charger_multiplier, charger_exponent)

    plt.scatter([int(len_each_bin_charger / 2) + len_each_bin_charger * i for i in range(n_bins - 1)], list_mean_charger_min, linestyle='None', marker='^')
    plt.plot(x_fit, y_fit, 'r-', label='Fit: {:.2f} * x^({:.2f})'.format(charger_multiplier, charger_exponent))
    plt.xlabel("n_available_chargers")
    plt.ylabel("drive_to_charger_time_min")
    plot_file = os.path.join(output_dir, "plot_charger_time_vs_available_cars.png")
    plt.savefig(plot_file)
    plt.clf()

    # Calculating average pickup and drive to the charger time
    consolidated_kpi = consolidate_the_kpis(root_dir=dir_name)
    average_pickup_min = round(consolidated_kpi["avg_pickup_time_min"].mean(), 2)
    average_drive_to_charger_min = round(consolidated_kpi["avg_drive_time_to_charger"].mean(), 2)
    print(f"Average pickup time: {average_pickup_min} mins")
    print(f"Average drive to the charger time: {average_drive_to_charger_min} mins")

    return pickup_multiplier, pickup_exponent, charger_multiplier, charger_exponent, average_pickup_min, average_drive_to_charger_min


def pod_ode(df_arrival_sequence, n_cars, n_chargers, d, pickup_multiplier, pickup_exponent, charger_multiplier, charger_exponent, average_pickup_min, average_drive_to_charger_min, customer_buffer, charger_buffer, error, plot_dir):
    step_size_min = 0.1
    n_chargers_admission = n_chargers - charger_buffer
    avg_trip_time_min = df_arrival_sequence["trip_time_min"].mean()
    print(f"Average trip time: {avg_trip_time_min} mins")
    avg_total_busy_time_min = (avg_trip_time_min + average_pickup_min + average_drive_to_charger_min) * (1 + error)
    n_trips_full_charge = int(
            SimMetaData.pack_size_kwh
            / SimMetaData.avg_vel_mph
            / SimMetaData.consumption_kwhpmi
            / avg_total_busy_time_min * 60
    )
    r = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
    n_cars_charging_or_idle = np.append(np.zeros(n_trips_full_charge), n_cars)
    n_cars_driving = np.zeros(n_trips_full_charge + 1)
    delta_n_cars_driving = np.zeros(n_trips_full_charge + 1)
    delta_n_cars_charging_or_idle = np.zeros(n_trips_full_charge + 1)
    first_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].min()
    last_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].max()
    total_sim_time_datetime = last_trip_pickup_datetime - first_trip_pickup_datetime
    sim_duration_min = int(total_sim_time_datetime.total_seconds() / 60.0) + 1
    prob = np.ones(n_trips_full_charge)

    list_n_cars_driving_with_passenger = []
    list_n_cars_driving_without_passenger = []
    list_n_cars_charging = []
    list_n_cars_idle = []
    list_n_cars_driving_to_charger = []
    list_soc = []
    list_n_trips_in_progress = []

    total_incoming_workload = 0
    total_successful_workload = 0
    for k in range(int(sim_duration_min/step_size_min)):
        curr_time_minus = first_trip_pickup_datetime + timedelta(minutes=k * step_size_min - 2)
        curr_time_plus = curr_time_minus + timedelta(minutes=4)
        df_curr_trips = df_arrival_sequence[
            (df_arrival_sequence["pickup_datetime"] > curr_time_minus) &
            (df_arrival_sequence["pickup_datetime"] <= curr_time_plus)
            ]
        arrival_rate_per_min = len(df_curr_trips) / 4
        trip_time_min = df_curr_trips["trip_time_min"].mean()
        pickup_time_min = pickup_multiplier * n_cars_charging_or_idle[n_trips_full_charge] ** pickup_exponent * (1 + error)
        total_n_cars_charging = min(n_cars_charging_or_idle[n_trips_full_charge - 1], n_chargers_admission)
        if total_n_cars_charging >= customer_buffer + 1:
            drive_to_charger_time_min = charger_multiplier * (n_chargers - total_n_cars_charging) ** charger_exponent * (1 + error)
        else:
            drive_to_charger_time_min = 0

        service_rate_per_min = 1 / (trip_time_min + pickup_time_min + drive_to_charger_time_min)

        n_cars_charging = np.minimum(n_cars_charging_or_idle, n_chargers_admission)
        if n_cars_charging_or_idle[n_trips_full_charge] <= customer_buffer:
            drop_customer = 1
        else:
            drop_customer = 0
        prob[0] = 1 - (n_cars_charging_or_idle[0] / n_cars_charging_or_idle[n_trips_full_charge]) ** d
        for j in range(1, n_trips_full_charge):
            prob[j] = 1 - (n_cars_charging_or_idle[j] / n_cars_charging_or_idle[n_trips_full_charge]) ** d
            delta_n_cars_driving[j] = arrival_rate_per_min * (prob[0] - prob[j]) * (1 - drop_customer) - n_cars_driving[j] * service_rate_per_min
            delta_n_cars_charging_or_idle[j] = - arrival_rate_per_min * (prob[0] - prob[j]) * (1 - drop_customer) - (n_cars_charging[j] - n_cars_charging[j-1]) / r / avg_total_busy_time_min + n_cars_driving[j+1] * service_rate_per_min
        delta_n_cars_charging_or_idle[n_trips_full_charge] = - arrival_rate_per_min * prob[0] * (1 - drop_customer) + n_cars_driving[n_trips_full_charge] * service_rate_per_min
        delta_n_cars_driving[n_trips_full_charge] = arrival_rate_per_min * prob[0] * (1 - drop_customer) - n_cars_driving[n_trips_full_charge] * service_rate_per_min
        delta_n_cars_charging_or_idle[0] = - n_cars_charging[0] / r / avg_total_busy_time_min + n_cars_driving[1] * service_rate_per_min

        # delta_n_cars_driving[1:n_trips_full_charge] = (
        #         arrival_rate_per_min * (prob[0] - prob[1:])
        #         - n_cars_driving[1:n_trips_full_charge] * service_rate_per_min) * step_size_min
        # delta_n_cars_charging_or_idle[1:n_trips_full_charge] = (
        #         - arrival_rate_per_min * (prob[0] - prob[1:])
        #         - (n_cars_charging[1:n_trips_full_charge] - n_cars_charging[0:n_trips_full_charge-1])
        #         / r / avg_total_busy_time_min
        #         + n_cars_driving[2:] * service_rate_per_min
        # ) * step_size_min
        # delta_n_cars_charging_or_idle[n_trips_full_charge] = (
        #         - arrival_rate_per_min * prob[0]
        #         + n_cars_driving[n_trips_full_charge] * service_rate_per_min
        # ) * step_size_min
        # delta_n_cars_charging_or_idle[0] = (
        #     - n_cars_charging[0] / r / avg_total_busy_time_min
        #     + n_cars_driving[1] * service_rate_per_min
        # ) * step_size_min
        n_cars_driving += delta_n_cars_driving * step_size_min
        n_cars_charging_or_idle += delta_n_cars_charging_or_idle * step_size_min

        total_incoming_workload += arrival_rate_per_min * step_size_min * trip_time_min
        total_successful_workload += arrival_rate_per_min * step_size_min * trip_time_min * prob[0] * (1 - drop_customer)
        if k * step_size_min % 1 == 0:
            curr_time = first_trip_pickup_datetime + timedelta(minutes=k * step_size_min)

            n_cars_driving_with_passenger = n_cars_driving[n_trips_full_charge] * trip_time_min * service_rate_per_min
            n_cars_driving_without_passenger = (
                    n_cars_driving[n_trips_full_charge] * pickup_time_min * service_rate_per_min
            )
            n_cars_driving_to_charger = (
                    n_cars_driving[n_trips_full_charge] * drive_to_charger_time_min * service_rate_per_min
            )
            n_cars_idle = n_cars_charging_or_idle[n_trips_full_charge] - total_n_cars_charging
            avg_soc = sum(n_cars - n_cars_charging_or_idle - n_cars_driving) / n_trips_full_charge / n_cars
            n_trips_in_progress = len(df_arrival_sequence[
                (df_arrival_sequence["pickup_datetime"] + timedelta(minutes=pickup_time_min) <= curr_time)
                &
                (df_arrival_sequence["pickup_datetime"] + timedelta(minutes=trip_time_min + pickup_time_min) > curr_time)
                ])

            list_n_cars_driving_with_passenger.append(n_cars_driving_with_passenger)
            list_n_cars_driving_without_passenger.append(n_cars_driving_without_passenger)
            list_n_cars_driving_to_charger.append(n_cars_driving_to_charger)
            list_n_cars_charging.append(total_n_cars_charging)
            list_n_cars_idle.append(n_cars_idle)
            list_soc.append(avg_soc)
            list_n_trips_in_progress.append(n_trips_in_progress)
    # Stackplot of the state of the EVs with SoC overlaid
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(0, sim_duration_min, 1)
    ax1.stackplot(x, np.array([
        list_n_cars_driving_with_passenger,
        list_n_cars_driving_without_passenger,
        list_n_cars_idle,
        list_n_cars_driving_to_charger,
        list_n_cars_charging
    ]), colors=['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#E6AB02', '#036c5f'])
    ax2.plot(x, list_soc, 'k', linewidth=3)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Number of Cars")
    ax1.set_ylim([0, n_cars])
    ax2.set_ylabel("SOC")
    ax2.set_ylim([0, 1])
    ax1.plot(x, list_n_trips_in_progress, 'm')
    plt.title("Demand Stackplot with SOC overlaid")
    results_folder = os.path.join(plot_dir, f"{n_cars}_nev_{n_chargers}_nc_{error}_error")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    demand_curve_plot_file = os.path.join(results_folder, "ode_stack_plot.png")
    demand_curve_data_file = os.path.join(results_folder, "ode_demand_curve_data.csv")
    plt.savefig(demand_curve_plot_file)
    plt.clf()
    pd.DataFrame({
        "n_cars_driving_with_passenger": list_n_cars_driving_with_passenger,
        "n_cars_driving_without_passenger": list_n_cars_driving_without_passenger,
        "n_cars_idle": list_n_cars_idle,
        "n_cars_driving_to_charger": list_n_cars_driving_to_charger,
        "n_cars_charging": list_n_cars_charging,
        "soc": list_soc,
        "n_trips_in_progress": list_n_trips_in_progress
    }).to_csv(demand_curve_data_file)

    return total_successful_workload / total_incoming_workload * 100

def post_process(kpi_consolidated, root_dir):
    # 1) sort by fleet_size
    kpi_consolidated = kpi_consolidated.sort_values(["n_chargers", "fleet_size"])

    group_cols = ["n_chargers", "error"]

    records = []
    # 2) regress fleet_size → percentage_workload_served per group
    for vals, grp in kpi_consolidated.groupby(group_cols):
        X = grp[["fleet_size"]].values.reshape(-1,1)
        y = grp["percentage_workload_served"].values

        # skip if not enough points
        if len(grp) < 2:
            continue
        else:
            model = LinearRegression().fit(X, y)
            a, b = model.coef_[0], model.intercept_
            r2 = model.score(X, y)
            # 3) solve 90 = a*x + b  ⇒  x = (90 - b)/a
            x90 = (90 - b) / a if a != 0 else np.nan

        rec = dict(zip(group_cols, vals))
        rec["90_percent_fleet_size"] = int(x90)
        rec["r_squared"] = r2
        records.append(rec)
    
    # 4) build final DataFrame
    df_90_percent_fleet_size = pd.DataFrame.from_records(records)
    output_path = Path(root_dir) / "90_percent_fleet_size.csv"
    df_90_percent_fleet_size.to_csv(output_path)

    return df_90_percent_fleet_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_sim', '--root_dir_sim', type=str)
    parser.add_argument('-od', '--output_dir', type=str, default="ode_simulation_results")
    args = parser.parse_args()
    (
        pickup_multiplier,
        pickup_exponent,
        charger_multiplier,
        charger_exponent,
        average_pickup_min,
        average_drive_to_charger_min
    ) = computing_spatial_functions(
        dir_name=args.root_dir_sim,
        output_dir=args.output_dir
    )    
    data_input = DataInput(percentile_lat_lon=DatasetParams.percentile_lat_lon)
    DatasetParams.uniform_locations = False
    dataset_source = Dataset.CHICAGO.value
    curr_dir = os.getcwd()
    dataset_path = os.path.join(curr_dir, "Chicago_year_2022_month_06.csv")
    start_datetime = datetime(2022, 6, 14, 0, 0, 0)
    end_datetime = datetime(2022, 6, 17, 0, 0, 0)
    dist_func = DistFunc.MANHATTAN.value
    df_arrival_sequence, dist_correction_factor = data_input.real_life_dataset(
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        percent_of_trips=0.6,
        dist_func=dist_func
    )
    dfs = []
    for cars in [2100, 2200, 2300, 2400, 2500, 2600]:
        for chargers in [500, 700, 900]:
            for error in [-0.2, 0, 0.2]:
                new_cars = cars + error * 1000
                if chargers == 500:
                    d = 2.6
                elif chargers == 700:
                    d = 1.6
                else:
                    d = 1.4
                workload = pod_ode(df_arrival_sequence=df_arrival_sequence,
                                #    list_pickup_time=[9.28, 6.09, 4.7, 3.94, 3.14, 2.95, 2.72, 2.56, 2.32],
                                #    list_drive_to_charger_time_min=[14.05, 8.64, 7.11, 6.17, 6.04, 5.98, 5.09, 4.48, 4.48, 4.48, 4.48, 4.48, 4.46, 4.46, 4.46, 4.46, 3.96],
                                customer_buffer=50,
                                charger_buffer=20,
                                n_cars=new_cars,
                                n_chargers=chargers,
                                pickup_multiplier=pickup_multiplier,
                                pickup_exponent=pickup_exponent,
                                charger_multiplier=charger_multiplier,
                                charger_exponent=charger_exponent,
                                average_pickup_min=average_pickup_min,
                                average_drive_to_charger_min=average_drive_to_charger_min,
                                d=d,
                                error=error,
                                plot_dir=args.output_dir)
                kpi = pd.DataFrame({
                    "ev_type": ["Tesla_Model_3"],
                    "charge_rate_kw": [20],
                    "algo": ["ODE_POD"],
                    "d": [d],
                    "fleet_size": [new_cars],
                    "n_chargers": [chargers],
                    "perc_trip_filter": [0.6],
                    "error": [error],
                    "percentage_workload_served": [workload]
                })
                dfs.append(kpi)
    kpi_consolidated = pd.concat(dfs, ignore_index=True)
    kpi_file = os.path.join(args.output_dir, "kpi_consolidated_ode.csv")
    kpi_consolidated.to_csv(kpi_file)
    ninety_percent_fleet_size = post_process(kpi_consolidated, args.output_dir)
    print(ninety_percent_fleet_size)
