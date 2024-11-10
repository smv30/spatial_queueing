import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sim_metadata import SimMetaData, Dataset, DistFunc, DatasetParams
from real_life_data_input import DataInput
from datetime import datetime, timedelta
from scipy.optimize import curve_fit


def computing_spatial_functions(dir_name):
    df_pickup_consolidated = None
    for fname in os.listdir(dir_name):

        # build the path to the folder
        folder_path = os.path.join(dir_name, fname)

        if os.path.isdir(folder_path):
            # we are sure this is a folder; now lets iterate it
            filepath = os.path.join(folder_path, "pickup_time_vs_available_cars.csv")

            df_pickup = pd.read_csv(filepath)
            if "count_datapoints" in df_pickup.columns:
                if df_pickup_consolidated is None:
                    df_pickup_consolidated = df_pickup
                else:
                    df_pickup_consolidated = pd.concat([df_pickup_consolidated, df_pickup], axis=0)
    df_pickup_consolidated = df_pickup_consolidated[["n_available_cars", "mean_pickup_min", "std_pickup_min", "count_datapoints"]]
    df_pickup_consolidated = df_pickup_consolidated.dropna()
    results_file = os.path.join(dir_name, "consolidated_pickup_time_vs_available_cars.csv")
    df_pickup_consolidated.to_csv(results_file)

    df_pickup_consolidated["total_pickup_time"] = df_pickup_consolidated["mean_pickup_min"] * df_pickup_consolidated[
        "count_datapoints"]
    # Scatter of available chargers versus drive to the charger time
    bins = [100 * i for i in range(20)]
    list_mean_pickup_min = []
    n_available_cars = df_pickup_consolidated["n_available_cars"].to_numpy()
    total_pickup_time = df_pickup_consolidated["total_pickup_time"].to_numpy()
    count_datapoints = df_pickup_consolidated["count_datapoints"].to_numpy()
    for i in range(len(bins) - 1):
        indices = [idx for idx, val in enumerate(n_available_cars) if bins[i] <= val < bins[i + 1]]

        total_pickup_time_min = np.mean([total_pickup_time[idx] for idx in indices])
        total_count = np.mean([count_datapoints[idx] for idx in indices])
        list_mean_pickup_min.append(np.round(total_pickup_time_min / total_count, 2))

    print(df_pickup_consolidated)

    # Define the power-law function
    def power_law(x, a, b):
        return a * x ** b

    # Convert lists to numpy arrays
    x_data = np.array([50 + 100 * i for i in range(9)])
    y_data = np.array(list_mean_pickup_min[0:9])

    # Perform the curve fitting
    params, covariance = curve_fit(power_law, x_data, y_data)

    # Get the parameters
    a, b = params
    print(a, b)

    # Generate the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = power_law(x_fit, a, b)

    plt.scatter([50 + 100 * i for i in range(19)], list_mean_pickup_min, linestyle='None', marker='^')
    plt.plot(x_fit, y_fit, 'r-', label='Fit: {:.2f} * x^({:.2f})'.format(a, b))
    plt.xlabel("n_available_cars")
    plt.ylabel("pickup_time_min")
    plot_file = os.path.join(dir_name, "plot_pickup_time_vs_available_cars.png")
    plt.savefig(plot_file)
    print(list_mean_pickup_min)
    plt.clf()

    # Drive to charger time
    df_charger_consolidated = None
    for fname in os.listdir(dir_name):

        # build the path to the folder
        folder_path = os.path.join(dir_name, fname)

        if os.path.isdir(folder_path):
            # we are sure this is a folder; now lets iterate it
            filepath = os.path.join(folder_path, "drive_to_charger_time_vs_available_posts_with_driving_cars.csv")

            df_charger = pd.read_csv(filepath)
            if "count_datapoints" in df_charger.columns:
                if df_charger_consolidated is None:
                    df_charger_consolidated = df_charger
                else:
                    df_charger_consolidated = pd.concat([df_charger_consolidated, df_charger], axis=0)

    results_file = os.path.join(dir_name, "consolidated_charger_time_vs_available_cars.csv")
    df_charger_consolidated.to_csv(results_file)

    df_charger_consolidated["total_charger_time"] = df_charger_consolidated["mean_drive_to_charger_min"] * \
                                                    df_charger_consolidated["count_datapoints"]
    df_charger_consolidated = df_charger_consolidated.dropna()

    # Scatter of available chargers versus drive to the charger time
    bins = [50 * i for i in range(19)]
    list_mean_charger_min = []
    n_available_chargers = df_charger_consolidated["n_available_chargers"].to_numpy()
    total_charger_time = df_charger_consolidated["total_charger_time"].to_numpy()
    count_datapoints = df_charger_consolidated["count_datapoints"].to_numpy()
    for i in range(len(bins) - 1):
        indices = [idx for idx, val in enumerate(n_available_chargers) if bins[i] <= val < bins[i + 1]]

        total_charger_time_min = np.mean([total_charger_time[idx] for idx in indices])
        total_count = np.mean([count_datapoints[idx] for idx in indices])
        list_mean_charger_min.append(np.round(total_charger_time_min / total_count, 2))

    # Define the power-law function
    def power_law(x, a, b):
        return a * x ** b

    # Convert lists to numpy arrays
    x_data = np.array([25 + 50 * i for i in range(18)])
    y_data = np.array(list_mean_charger_min)

    # Perform the curve fitting
    params, covariance = curve_fit(power_law, x_data, y_data)

    # Get the parameters
    a, b = params
    print(a, b)

    # Generate the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = power_law(x_fit, a, b)

    plt.scatter([25 + 50 * i for i in range(18)], list_mean_charger_min, linestyle='None', marker='^')
    plt.plot(x_fit, y_fit, 'r-', label='Fit: {:.2f} * x^({:.2f})'.format(a, b))
    plt.xlabel("n_available_chargers")
    plt.ylabel("drive_to_charger_time_min")
    plot_file = os.path.join(dir_name, "plot_charger_time_vs_available_cars.png")
    plt.savefig(plot_file)
    print(list_mean_charger_min)
    plt.clf()


def pod_ode(df_arrival_sequence, list_pickup_time, list_drive_to_charger_time_min, n_cars, n_chargers, d, customer_buffer, charger_buffer, error):
    plot_dir = "simulation_results"
    step_size_min = 0.1
    n_chargers_admission = n_chargers - charger_buffer
    avg_trip_time_min = df_arrival_sequence["trip_time_min"].mean()
    avg_total_busy_time_min = (avg_trip_time_min + 13.5) * (1 + error)
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
        pickup_time_min = 60 * n_cars_charging_or_idle[n_trips_full_charge] ** (-0.47) * (1 + error)
        total_n_cars_charging = min(n_cars_charging_or_idle[n_trips_full_charge - 1], n_chargers_admission)
        if total_n_cars_charging >= customer_buffer + 1:
            drive_to_charger_time_min = 39 * (n_chargers - total_n_cars_charging) ** (-0.33) * (1 + error)
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
    demand_curve_plot_file = os.path.join(plot_dir, "ode_stack_plot.eps")
    demand_curve_data_file = os.path.join(plot_dir, "ode_demand_curve_data.csv")
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


if __name__ == "__main__":
    # computing_spatial_functions(dir_name="/Users/sushilvarma/Library/Mobile Documents/com~apple~CloudDocs/Academics/Research/EV/MS_Second_Round_Plots_Final/5_ode_comparison_45_mins_attempt_3")
    data_input = DataInput(percentile_lat_lon=99)
    dataset_source = Dataset.CHICAGO.value
    dataset_path = '/Users/sushilvarma/PycharmProjects/SpatialQueueing/Chicago_year_2022_month_06.csv'
    start_datetime = datetime(2022, 6, 14, 0, 0, 0)
    end_datetime = datetime(2022, 6, 17, 0, 0, 0)
    dist_func = DistFunc.MANHATTAN.value
    df_arrival_sequence, dist_correction_factor = data_input.real_life_dataset(
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        percent_of_trips=DatasetParams.percent_of_trips_filtered,
        dist_func=dist_func
    )
    for cars in [2200]:
        for chargers in [900]:
            workload = pod_ode(df_arrival_sequence=df_arrival_sequence,
                               list_pickup_time=[9.28, 6.09, 4.7, 3.94, 3.14, 2.95, 2.72, 2.56, 2.32],
                               list_drive_to_charger_time_min=[14.05, 8.64, 7.11, 6.17, 6.04, 5.98, 5.09, 4.48, 4.48, 4.48, 4.48, 4.48, 4.46, 4.46, 4.46, 4.46, 3.96],
                               customer_buffer=50,
                               charger_buffer=20,
                               n_cars=cars,
                               n_chargers=chargers,
                               d=1.4,
                               error=0)
            print(f"n_cars: {cars}, n_chargers: {chargers}, workload: {workload}")
