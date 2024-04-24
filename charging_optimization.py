import os
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ortools.linear_solver import pywraplp
from sim_metadata import SimMetaData
from utils import calc_dist_between_two_points
import seaborn as sns


def optimization(avg_trip_pickup_time_min, avg_driving_to_charger_time_min):
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    sim_duration = 24 * 60 - 1
    n_cars = 3000

    # n_trips: num arrivals at time t = num trips arrive between t and t + 1
    n_trips = []
    time_list = []
    df_trips = pd.read_csv(
        "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/spatial_queueing/sampledata.csv")
    df_demand_curve = pd.read_csv(
        "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/simulation_results/Apr_11_2024_20_57_20 - closest available, charge idle, send only idle=False [5-day]/demand_curve/fleet_demand_curve.csv")
    if not os.path.isfile("optimization_data.csv"):
        # n_trips = np.ones(sim_duration) * 200
        # df_trips = pd.read_csv(
        #     "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/sampledata.csv")
        # avg_trip_time (list equal to the sim duration) calculated from the dataset, for every minute
        # avg_trip_time: the avg_trip_time of all trips that arrives between t and t+1
        # calculate trip time for each trip (add a column)
        # consider the trips which has pickup time between t and t+1. For all those trips, take the average trip time
        df_trips["trip duration"] = (pd.to_datetime(df_trips["dropoff_datetime"]) - pd.to_datetime(
            df_trips["pickup_datetime"])).dt.total_seconds() / 60.0
        avg_trip_time_min_list = []
        for t in range(0, sim_duration):
            start_datetime = datetime(2010, 12, 1, int(t / 60), t % 60, 0)
            end_datetime = datetime(2010, 12, 1, int((t + 1) / 60), (t + 1) % 60, 0)
            df_trips.loc[:, "pickup_datetime"] = pd.to_datetime(df_trips['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
            num_trips_at_time_t = len(df_trips[(df_trips["pickup_datetime"] >= start_datetime) &
                                               (df_trips["pickup_datetime"] < end_datetime)])
            trip_time_min_sum = sum(df_trips[(df_trips["pickup_datetime"] >= start_datetime) &
                                             (df_trips["pickup_datetime"] < end_datetime)]["trip duration"])
            avg_trip_time_min_list.append(int(trip_time_min_sum / num_trips_at_time_t))
        for t in range(0, sim_duration):
            start_datetime = datetime(2010, 12, 1, int(t / 60), t % 60, 0)
            end_datetime = datetime(2010, 12, 1, int((t + 1) / 60), (t + 1) % 60, 0)
            df_trips.loc[:, "pickup_datetime"] = pd.to_datetime(df_trips['pickup_datetime'],
                                                                format='%Y-%m-%d %H:%M:%S')
            n_trips.append(len(df_trips[(df_trips["pickup_datetime"] >= start_datetime) &
                                        (df_trips["pickup_datetime"] < end_datetime)]))
            time_list.append(t)
        df_n_trips_and_avg_trip_time = pd.DataFrame({'Time (min)': time_list,
                                                     'Number of Trips': n_trips,
                                                     'Average Trip Time (min)': avg_trip_time_min_list})
        df_n_trips_and_avg_trip_time.to_csv(
            "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/optimization_data.csv")
    else:
        df_n_trips_and_avg_trip_time = pd.read_csv(
            "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/optimization_data.csv")
        time_list = df_n_trips_and_avg_trip_time['Time (min)'].to_list()
        n_trips = df_n_trips_and_avg_trip_time['Number of Trips'].to_list()
        avg_trip_time_min_list = df_n_trips_and_avg_trip_time['Average Trip Time (min)'].to_list()

    # n_trips = {}
    n_pickup = {}
    n_driving_with_passenger = {}
    n_driving_without_passenger = {}
    n_charging = {}
    n_to_charge = {}
    n_driving_to_charger = {}
    n_idle = {}
    soc = {}

    # n_chargers = 520 * 8
    n_chargers = 1000
    soc[0] = 0.5
    # avg_driving_time_to_charger_min = 1 / np.sqrt(n_chargers - n_charging)
    avg_driving_time_to_charger_min = 1
    # avg_trip_pickup_time_min comes from the simulation
    # replace the square root function by a set of linear lines -> linearize
    # avg_trip_pickup_time_min = 1 / np.sqrt(n_charging + n_idle)
    # c: 1, 2, 4, 8, 16, 32, 2^n ..., n_cars
    # y: avg_trip_pickup_time_min[t]; x: n_charging[t] + n_idle[t]; f(c) = 1 / np.sqrt(c); f(2c) = 1 / np.sqrt(2c)
    # (y - f(c)) >= (f(2c) - f(c)) / c * (x - c)
    discharge_rate_kw = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph  # in kW
    charge_rate_kw = SimMetaData.charge_rate_kw  # in kW
    charge_80_percent_time_min = int(SimMetaData.pack_size_kwh / charge_rate_kw * 0.8)

    # Create the variables
    for t in range(0, sim_duration):
        n_pickup[t] = solver.NumVar(0, n_cars, f"n_pickup_{t}")
        n_to_charge[t] = solver.NumVar(0, n_cars, f"n_to_charge_{t}")  # number of cars we send to charge
        # n_driving[t] = solver.NumVar(0, n_cars, f"n_driving_{t}")  # n_driving = n_driving_with_passenger + n_driving_without_passenger
        n_driving_with_passenger[t] = solver.NumVar(0, n_cars, f"n_driving_with_passenger_{t}")
        n_driving_without_passenger[t] = solver.NumVar(0, n_cars, f"n_driving_without_passenger_{t}")
        n_driving_to_charger[t] = solver.NumVar(0, n_cars, f"n_driving_to_charger_{t}")
        n_charging[t] = solver.NumVar(0, n_cars, f"n_charging_{t}")  # number of cars that are currently charging
        n_idle[t] = solver.NumVar(0, n_cars, f"n_idle_{t}")
        # avg_trip_pickup_time_min[t] = solver.NumVar(0, n_cars, f"avg_trip_pickup_time_min_{t}")
        if t >= 1:
            soc[t] = solver.NumVar(0, 1, f"soc_{t}")
    max_cars_charging = solver.NumVar(0, n_cars, "max_cars_charging")
    print("Number of variables =", solver.NumVariables())

    # Create linear constraints
    for t in range(0, sim_duration):
        solver.Add(n_pickup[t] - n_trips[t] <= 0)
        # solver.Add(n_driving_with_passenger[t] - sum(
        #     n_pickup[s] for s in range(max(0, t - avg_trip_pickup_time_min - avg_trip_time_min_list[t]),
        #                                max(0, t - avg_trip_pickup_time_min))) == 0)
        v = n_driving_with_passenger[t]
        max_avg_trip_time_min = max(avg_trip_time_min_list)
        for s in range(max(0, t - max_avg_trip_time_min - avg_trip_pickup_time_min[t]), t):
            if s + avg_trip_pickup_time_min[s] + avg_trip_time_min_list[s] >= t and s + avg_trip_pickup_time_min[s] <= t:
                v = v - n_pickup[s]
        solver.Add(v == 0)
        solver.Add(n_driving_without_passenger[t] - sum(
            n_pickup[s] for s in range(max(0, t - avg_trip_pickup_time_min[t]), t)) == 0)
        solver.Add(
            n_driving_to_charger[t] - sum(
                n_to_charge[s] for s in range(max(0, t - avg_driving_time_to_charger_min), t)) == 0)
        solver.Add(n_charging[t] - sum(
            n_to_charge[s] for s in range(max(0, t - charge_80_percent_time_min - avg_driving_time_to_charger_min),
                                          max(0, t - avg_driving_time_to_charger_min))) == 0)
        solver.Add(
            n_idle[t] == n_cars - n_driving_with_passenger[t] - n_driving_without_passenger[t] - n_driving_to_charger[
                t] - n_charging[t])
        solver.Add(n_charging[t] <= n_chargers)
        solver.Add(max_cars_charging >= n_charging[t])
        if t >= 1:
            solver.Add(
                soc[t] == soc[t - 1] - ((n_driving_with_passenger[t] + n_driving_without_passenger[t] +
                                         n_driving_to_charger[t]) / n_cars) * (
                        discharge_rate_kw / 60) / SimMetaData.pack_size_kwh + (
                        n_charging[t] / n_cars) * (charge_rate_kw / 60) / SimMetaData.pack_size_kwh)
            solver.Add(n_pickup[t] - n_pickup[t - 1] <= abs(n_trips[t] - n_trips[t - 1]))
            solver.Add(n_pickup[t - 1] - n_pickup[t] <= abs(n_trips[t - 1] - n_trips[t]))
        solver.Add(soc[t] - 0.1 >= 0)
        solver.Add(n_idle[t] + n_charging[t] >= 1)
    solver.Add(soc[sim_duration - 1] == soc[0])
    print("Number of constraints =", solver.NumConstraints())

    # Create the objective function
    solver.Maximize(sum(n_pickup[t] * avg_trip_time_min_list[t] for t in
                        range(0, sim_duration)) - 0.01 * max_cars_charging)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    time_of_logging = []
    n_cars_idle = []
    n_cars_driving_to_charger = []
    n_cars_charging = []
    n_cars_driving_with_passenger = []
    n_cars_driving_without_passenger = []
    n_trip_accepted = []
    # n_cars_waiting_for_charger = []
    avg_soc = []
    if status == pywraplp.Solver.OPTIMAL:
        print('\nSolution: OPTIMAL')
        print('Objective value = ', solver.Objective().Value())
        for t in range(1, sim_duration):
            time_of_logging.append(t)
            n_cars_idle.append(n_idle[t].solution_value())
            n_cars_driving_to_charger.append(n_driving_to_charger[t].solution_value())
            n_cars_charging.append(n_charging[t].solution_value())
            n_cars_driving_with_passenger.append(n_driving_with_passenger[t].solution_value())
            n_cars_driving_without_passenger.append(n_driving_without_passenger[t].solution_value())
            avg_soc.append(soc[t].solution_value())
            n_trip_accepted.append(n_pickup[t].solution_value())
            # print(f"driving at time {t} = ", n_driving[t].solution_value())
            # print(f"soc at time {t} = ", soc[t].solution_value())

    # plot the graph
    results_folder = "simulation_results/"
    today = datetime.now()
    curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
    top_level_dir = os.path.join(results_folder, curr_date_and_time)
    if not os.path.isdir(top_level_dir):
        os.makedirs(top_level_dir)
    demand_curve_data_file = os.path.join(top_level_dir, "simulation_curve.csv")
    df_simulation_data = pd.DataFrame({
        "time": time_of_logging,
        "idle": n_cars_idle,
        "driving_to_charger": n_cars_driving_to_charger,
        "charging": n_cars_charging,
        "driving_with_passenger": n_cars_driving_with_passenger,
        "driving_without_passenger": n_cars_driving_without_passenger,
        # "waiting_for_charger": n_cars_waiting_for_charger,
        "avg_soc": avg_soc,
        # "stdev_soc": self.stdev_soc
    })
    df_simulation_data.to_csv(demand_curve_data_file)

    # pink line: num trips started before the current time & finished after the current time
    num_trips_demand_list = []
    for t in range(0, sim_duration):
        # num_trips_demand = (sum(
        #     n_pickup[s].solution_value() for s in
        #     range(max(0, t - avg_trip_time_min_list[t] - avg_trip_pickup_time_min),
        #           max(0, t - avg_trip_pickup_time_min))) +
        #                     sum(n_trips[s] - n_pickup[s].solution_value() for s in
        #                         range(max(0, t - avg_trip_time_min_list[t]), t))
        #                     )
        v1 = 0
        max_avg_trip_time_min = max(avg_trip_time_min_list)
        for s in range(max(0, t - max_avg_trip_time_min - avg_trip_pickup_time_min[t]), t):
            if s + avg_trip_pickup_time_min[s] + avg_trip_time_min_list[s] >= t:
                v1 = v1 + n_pickup[s].solution_value()
        v2 = 0
        for s in range(max(0, t - max_avg_trip_time_min), t):
            if s + avg_trip_pickup_time_min[s] + avg_trip_time_min_list[s] >= t:
                v2 = v2 + n_trips[s] - n_pickup[s].solution_value()
        num_trips_demand = v1 + v2
        if num_trips_demand > n_cars:
            num_trips_demand = n_cars
        num_trips_demand_list.append(num_trips_demand)

    pink_area = sum(num_trips_demand_list)
    df_simulation_data["delta_time"] = df_simulation_data["time"].shift(-1) - df_simulation_data["time"]
    df_simulation_data["delta_time"].fillna(0, inplace=True)
    blue_area = (df_simulation_data["delta_time"] * df_simulation_data["driving_with_passenger"]).sum()
    workload = blue_area / pink_area
    print(f"Workload: {workload}")

    # Plot Results
    plot_dir = os.path.join(top_level_dir, "plots")
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = df_simulation_data["time"].to_numpy()
    soc = df_simulation_data["avg_soc"].to_numpy()
    ax1.stackplot(x, np.transpose(df_simulation_data[[
        "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging"]].to_numpy()),
                  colors=['b', 'tab:orange', 'g', 'tab:purple', 'r'])
    ax2.plot(x, soc, 'k', linewidth=3)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Number of Cars")
    ax1.set_ylim([0, n_cars])
    ax2.set_ylabel("SOC")
    ax2.set_ylim([0, 1])
    ax1.plot(time_list, num_trips_demand_list, 'm')
    plt.title("Optimization Stackplot with SOC overlaid")
    demand_curve_plot_file = os.path.join(plot_dir, "optimization_stackplot.png")
    plt.savefig(demand_curve_plot_file)
    plt.clf()

    print(sum(n_trips))

    df_trips["pickup_datetime"] = pd.to_datetime(df_trips["pickup_datetime"])
    df_trips["dropoff_datetime"] = pd.to_datetime(df_trips["dropoff_datetime"])
    df_trips["trip_time_min"] = (df_trips["dropoff_datetime"] - df_trips["pickup_datetime"]).dt.total_seconds() / 60.0
    SimMetaData.avg_vel_mph = sum(df_trips["trip_distance"]) / sum(df_trips["trip_time_min"]) * 60
    print(f"Average velocity (mph): {SimMetaData.avg_vel_mph}")

    for t in range(len(df_simulation_data)):
        # shortest_pickup_time_min_list = []
        n_dropoff_locations = df_simulation_data["idle"][t] + df_simulation_data["charging"][t]
        if int(n_dropoff_locations) != 0:
            avg_shortest_pickup_time_min = 0
            for i in range(5):
                # time interval: [t-30, t+30]
                if t > 30:
                    start_day = int((t - 30) / 1440) + 1  # +1 because day starts at 1
                    start_hour = int((t - 30) % 1440 / 60)
                    start_min = (t - 30) - (start_day - 1) * 1440 - start_hour * 60
                else:
                    start_day, start_hour, start_min = 1, 0, 0
                end_day = int((t + 30) / 1440) + 1
                end_hour = int((t + 30) % 1440 / 60)
                end_min = (t + 30) - (end_day - 1) * 1440 - end_hour * 60
                filtered_df_trips = df_trips[
                    (df_trips["pickup_datetime"] >= datetime(2010, 12, start_day, start_hour, start_min, 0)) &
                    (df_trips["pickup_datetime"] < datetime(2010, 12, end_day, end_hour, end_min, 0))]
                sample_dropoff = filtered_df_trips.sample(int(n_dropoff_locations), replace=True)
                sample_pickup = filtered_df_trips.sample(1)
                if int(n_dropoff_locations) == 1:
                    shortest_pickup_time_min = calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"].values[0],
                        end_lon=sample_dropoff["dropoff_longitude"].values[0],
                        dist_correction_factor=1.3966190351834704
                    ) / SimMetaData.avg_vel_mph * 60
                else:
                    shortest_pickup_time_min = min(calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"],
                        end_lon=sample_dropoff["dropoff_longitude"],
                        dist_correction_factor=1.3966190351834704
                    )) / SimMetaData.avg_vel_mph * 60
                avg_shortest_pickup_time_min = (avg_shortest_pickup_time_min * i + shortest_pickup_time_min) / (i + 1)
                # shortest_pickup_time_min_list.append(shortest_pickup_time_min)
            avg_trip_pickup_time_min[t] = int(avg_shortest_pickup_time_min) + 1  # round up
        # avg_trip_pickup_time_min[t] = int(sum(shortest_pickup_time_min_list) / len(shortest_pickup_time_min_list)) + 1

        n_dropoff_locations = n_chargers - df_simulation_data["charging"][t]
        if int(n_dropoff_locations) != 0:
            avg_shortest_driving_to_charger_time_min = 0
            for i in range(5):
                # time interval: [t-30, t+30]
                if t > 30:
                    start_day = int((t - 30) / 1440) + 1  # +1 because day starts at 1
                    start_hour = int((t - 30) % 1440 / 60)
                    start_min = (t - 30) - (start_day - 1) * 1440 - start_hour * 60
                else:
                    start_day, start_hour, start_min = 1, 0, 0
                end_day = int((t + 30) / 1440) + 1
                end_hour = int((t + 30) % 1440 / 60)
                end_min = (t + 30) - (end_day - 1) * 1440 - end_hour * 60
                filtered_df_trips = df_trips[
                    (df_trips["pickup_datetime"] >= datetime(2010, 12, start_day, start_hour, start_min, 0)) &
                    (df_trips["pickup_datetime"] < datetime(2010, 12, end_day, end_hour, end_min, 0))]
                sample_dropoff = filtered_df_trips.sample(int(n_dropoff_locations), replace=True)
                sample_pickup = df_trips.sample(1)
                if int(n_dropoff_locations) == 1:
                    shortest_driving_to_charger_time_min = calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"].values[0],
                        end_lon=sample_dropoff["dropoff_longitude"].values[0],
                        dist_correction_factor=1.3966190351834704
                    ) / SimMetaData.avg_vel_mph * 60
                else:
                    shortest_driving_to_charger_time_min = min(calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"],
                        end_lon=sample_dropoff["dropoff_longitude"],
                        dist_correction_factor=1.3966190351834704
                    )) / SimMetaData.avg_vel_mph * 60
                avg_shortest_driving_to_charger_time_min = (avg_shortest_driving_to_charger_time_min * i + shortest_driving_to_charger_time_min) / (i + 1)
            avg_driving_to_charger_time_min[t] = int(avg_shortest_driving_to_charger_time_min) + 1

    avg_trip_pickup_time_min_available = [1] * len(df_demand_curve["n_cars_available"])
    for count in range(len(df_demand_curve["time"])):
        t = df_demand_curve["time"][count]
        n_cars_available = df_demand_curve["n_cars_available"][count]
        if int(n_cars_available) > 0:
            avg_shortest_pickup_time_min = 0
            for i in range(5):
                # time interval: [t-30, t+30]
                if t > 30:
                    start_day = int((t - 30) / 1440) + 1  # +1 because day starts at 1
                    start_hour = int((t - 30) % 1440 / 60)
                    start_min = int((t - 30) - (start_day - 1) * 1440 - start_hour * 60)
                else:
                    start_day, start_hour, start_min = 1, 0, 0
                end_day = int((t + 30) / 1440) + 1
                end_hour = int((t + 30) % 1440 / 60)
                end_min = int((t + 30) - (end_day - 1) * 1440 - end_hour * 60)
                filtered_df_trips = df_trips[
                    (df_trips["pickup_datetime"] >= datetime(2010, 12, start_day, start_hour, start_min, 0)) &
                    (df_trips["pickup_datetime"] < datetime(2010, 12, end_day, end_hour, end_min, 0))]
                sample_dropoff = filtered_df_trips.sample(int(n_cars_available), replace=True)
                sample_pickup = filtered_df_trips.sample(1)
                if int(n_cars_available) == 1:
                    shortest_pickup_time_min = calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"].values[0],
                        end_lon=sample_dropoff["dropoff_longitude"].values[0],
                        dist_correction_factor=1.3966190351834704
                    ) / SimMetaData.avg_vel_mph * 60
                else:
                    shortest_pickup_time_min = min(calc_dist_between_two_points(
                        start_lat=sample_pickup["pickup_latitude"].values[0],
                        start_lon=sample_pickup["pickup_longitude"].values[0],
                        end_lat=sample_dropoff["dropoff_latitude"],
                        end_lon=sample_dropoff["dropoff_longitude"],
                        dist_correction_factor=1.3966190351834704
                    )) / SimMetaData.avg_vel_mph * 60
                avg_shortest_pickup_time_min = (avg_shortest_pickup_time_min * i + shortest_pickup_time_min) / (i + 1)
            # avg_trip_pickup_time_min_available[t] = int(avg_shortest_pickup_time_min) + 1

    # fig, ax = plt.subplots()
    # x = df_demand_curve["n_cars_available"]
    # y = avg_trip_pickup_time_min_available
    # ax.plot(x, y)
    # ax.set_xlabel("Number of Cars Available")
    # ax.set_ylabel("Average Trip Pickup Time (min)")
    # ax.set_xlim([0, n_cars])
    # plt.title("avg_trip_pickup_time_min vs. n_cars_available")
    # demand_curve_plot_file = os.path.join(plot_dir, "avg_trip_pickup_time_and_n_cars_available.png")
    # plt.savefig(demand_curve_plot_file)
    # plt.clf()

    # df_demand_curve["n_cars_available"], df_demand_curve["pickup_time_min"], avg_trip_pickup_time_min_available
    bins = np.arange(0, df_demand_curve['n_cars_available'].max() + 100, 100)

    # pd.cut(): separate the array elements into different bins
    df_demand_curve['a'] = 1
    df_demand_curve['bin'] = pd.cut(df_demand_curve['n_cars_available'], bins, right=False, labels=np.arange(len(bins) - 1))
    df_demand_curve_filtered_sim = df_demand_curve.dropna(subset=['pickup_time_min'])
    number_in_each_bin = df_demand_curve_filtered_sim.groupby('bin', observed=True)["a"].sum()
    print(number_in_each_bin)
    # print(df_demand_curve[(df_demand_curve["n_cars_available"] <= 100) & (df_demand_curve["n_cars_available"] >= 0)]["pickup_time_min"].mean())

    # Filter out rows where 'pickup_time_min' is 0
    df_demand_curve_filtered_sim = df_demand_curve[df_demand_curve['pickup_time_min'] != 0]
    # Filter out rows where 'pickup_time_min' is NaN
    simulation_avg_pickup_time = df_demand_curve_filtered_sim.groupby('bin', observed=True)["pickup_time_min"]
    simulation_avg_pickup_time_mean = simulation_avg_pickup_time.mean()

    # df_demand_curve["pickup_time_min_optimization"] = avg_trip_pickup_time_min_available
    # Filter out rows where 'pickup_time_min_optimization' is 0
    # df_demand_curve_filtered_opt = df_demand_curve[df_demand_curve['pickup_time_min_optimization'] != 0]
    # optimization_avg_pickup_time = df_demand_curve_filtered_opt.groupby('bin', observed=True)["pickup_time_min_optimization"].mean()

    bins_df = pd.DataFrame(index=np.arange(len(bins) - 1))
    simulation_avg_pickup_time_mean = bins_df.join(simulation_avg_pickup_time_mean, how='left').fillna(0)
    # optimization_avg_pickup_time = bins_df.join(optimization_avg_pickup_time, how='left').fillna(0)
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_avg_pickup_time_mean.index * 100, simulation_avg_pickup_time_mean, label='Simulation', marker='o')
    # plt.plot(optimization_avg_pickup_time.index * 100, optimization_avg_pickup_time, label='Optimization', marker='x')
    plt.xlabel('Number of Cars Available (bin)')
    plt.ylabel('Average Pickup Time (min)')
    plt.legend()
    plt.grid(True)
    plt.title('Average Pickup Time vs. Number of Cars Available')
    simulation_optimization_plot_file = os.path.join(plot_dir, "avg_pickup_time_vs_n_cars_available.png")
    plt.savefig(simulation_optimization_plot_file)
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bin', y='pickup_time_min', data=df_demand_curve_filtered_sim)
    plt.xlabel('Number of Cars Available (bin)')
    plt.ylabel('Average Pickup Time (min)')
    plt.legend()
    plt.grid(True)
    plt.title('Average Pickup Time vs. Number of Cars Available')
    simulation_optimization_plot_file = os.path.join(plot_dir, "avg_pickup_time_vs_n_cars_available_boxplot.png")
    plt.savefig(simulation_optimization_plot_file)
    plt.clf()

    return avg_trip_pickup_time_min, avg_driving_to_charger_time_min


if __name__ == "__main__":
    sim_duration = 24 * 60 - 1
    avg_trip_pickup_time_min = [1] * sim_duration
    avg_driving_to_charger_time_min = [1] * sim_duration
    # for i in range(10):
    avg_trip_pickup_time_min, avg_driving_to_charger_time_min = optimization(avg_trip_pickup_time_min, avg_driving_to_charger_time_min)
        # print(avg_trip_pickup_time_min)
        # print(avg_driving_to_charger_time_min)
