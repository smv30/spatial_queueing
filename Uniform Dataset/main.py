import numpy as np
import pandas as pd
import simpy
import os
import argparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger
from sim_metadata import SimMetaData, TripState, MatchingAlgo, ChargingAlgoParams, ChargingAlgo


def run_simulation(
        sim_duration,
        n_cars,
        arrival_rate_pmin,
        n_chargers,
        n_posts,
        d,
        matching_algo=MatchingAlgo.POWER_OF_D_IDLE.value,
        charging_algo=ChargingAlgo.CHARGE_AFTER_TRIP_END.value,
        renege_time_min=None,
        results_folder=None,
        infinite_chargers=None,
        trip_data_csv_path="",
        keyword_folder=""
):
    start_time = time.time()
    env = simpy.Environment()

    # add variables not in outside to main.py
    if infinite_chargers is not None:
        ChargingAlgoParams.infinite_chargers = infinite_chargers

    if SimMetaData.random_data is False:
        df_trip_data = pd.read_csv(trip_data_csv_path)
        sim_duration = df_trip_data["arrival_time"].max()
    else:
        df_trip_data = None

    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts,
                               env=env)
        list_chargers.append(charger)

    # Initializing all the cars
    car_tracker = []
    for car_id in range(n_cars):
        car = Car(car_id=car_id, env=env, list_chargers=list_chargers)
        car_tracker.append(car)

    for charger in list_chargers:
        charger.car_tracker = car_tracker

    fleet_manager = FleetManager(arrival_rate_pmin=arrival_rate_pmin,
                                 env=env,
                                 car_tracker=car_tracker,
                                 n_cars=n_cars,
                                 renege_time_min=renege_time_min,
                                 list_chargers=list_chargers,
                                 matching_algo=matching_algo,
                                 charging_algo=charging_algo,
                                 d=d,
                                 df_trip_data=df_trip_data)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration)

    # Saving KPIs and sim metadata
    total_n_trips = len(fleet_manager.list_trips)
    avg_trip_time_min = np.mean([fleet_manager.list_trips[trip].calc_trip_time() for trip in range(total_n_trips)])
    list_trip_time_fulfilled_trips = []
    for trip in range(total_n_trips):
        if fleet_manager.list_trips[trip].state == TripState.MATCHED:
            list_trip_time_fulfilled_trips.append(fleet_manager.list_trips[trip].calc_trip_time())
    avg_trip_time_fulfilled_min = np.mean(list_trip_time_fulfilled_trips)
    list_trip_time_fulfilled_trips_second_half = []
    list_avg_pickup_time_min_second_half = []
    for trip in range(total_n_trips):
        if fleet_manager.list_trips[trip].state == TripState.MATCHED and fleet_manager.list_trips[trip].arrival_time_min >= sim_duration / 2:
            list_trip_time_fulfilled_trips_second_half.append(fleet_manager.list_trips[trip].calc_trip_time())
            list_avg_pickup_time_min_second_half.append(fleet_manager.list_trips[trip].pickup_time_min)
    avg_trip_time_fulfilled_min_second_half = np.mean(list_trip_time_fulfilled_trips_second_half)
    avg_pickup_time_min_second_half = np.mean(list_avg_pickup_time_min_second_half)
    avg_trip_dist_mi = avg_trip_time_min / 60 * SimMetaData.avg_vel_mph

    total_n_of_successful_trips = sum([int(fleet_manager.list_trips[trip].state == TripState.MATCHED)
                                       for trip in range(total_n_trips)])
    avg_pickup_time_min = sum(
        [fleet_manager.list_trips[trip].pickup_time_min for trip in range(total_n_trips)]
    ) / total_n_of_successful_trips
    
    service_level_percentage = total_n_of_successful_trips / total_n_trips * 100

    total_n_of_successful_trips_second_half = sum(
        [
            int(fleet_manager.list_trips[trip].state == TripState.MATCHED)
            * int(fleet_manager.list_trips[trip].arrival_time_min >= sim_duration / 2)
            for trip in range(total_n_trips)
        ]
                                                 )
    total_n_trips_second_half = sum(
        [int(fleet_manager.list_trips[trip].arrival_time_min >= sim_duration / 2) for trip in range(total_n_trips)]
                                   )
    service_level_percentage_second_half = total_n_of_successful_trips_second_half / total_n_trips_second_half * 100

    avg_soc = sum([car_tracker[car].soc for car in range(n_cars)]) / n_cars
    avg_n_of_charging_trips = (
            sum([car_tracker[car].n_of_charging_stops for car in range(n_cars)])
            / n_cars
            / (sim_duration / 60)
    )
    avg_drive_to_charger_time_min = sum(
        car_tracker[car].total_drive_to_charge_time for car in range(n_cars)
    ) / sum(car_tracker[car].n_of_charging_stops for car in range(n_cars))

    if matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value and d == 1:
        kpi_matching_algo = "Closest Dispatch"
    elif matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value and d > 1:
        kpi_matching_algo = f"Power-of-{d}"
    elif matching_algo == MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value:
        kpi_matching_algo = "Closest Available Dispatch"
    else:
        kpi_matching_algo = "Unknown Matching Algorithm"

    kpi = pd.DataFrame({
        "fleet_size": n_cars,
        "pack_size_kwh": SimMetaData.pack_size_kwh,
        "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
        "charge_rate_kw": SimMetaData.charge_rate_kw,
        "avg_vel_mph": SimMetaData.avg_vel_mph,
        "n_chargers": n_chargers,
        "n_posts": n_posts,
        "total_sim_duration_min": sim_duration,
        "arrival_rate_pmin": arrival_rate_pmin,
        "total_n_trips": total_n_trips,
        "avg_trip_time_min": avg_trip_time_min,
        "avg_trip_time_fulfilled_min": avg_trip_time_fulfilled_min,
        "avg_trip_time_fulfilled_min_second_half": avg_trip_time_fulfilled_min_second_half,
        "avg_trip_dist_mi": avg_trip_dist_mi,
        "avg_pickup_time_min": avg_pickup_time_min,
        "avg_pickup_time_min_second_half": avg_pickup_time_min_second_half,
        "avg_drive_time_to_charger": avg_drive_to_charger_time_min,
        "number_of_trips_to_charger_per_car_per_hr": avg_n_of_charging_trips,
        "avg_soc_over_time_over_cars": avg_soc,
        "service_level_percentage": service_level_percentage,
        "service_level_percentage_second_half": service_level_percentage_second_half,
        "matching_algorithm": kpi_matching_algo
    }, index=[0])

    if SimMetaData.save_results:
        # Save Results

        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        folder_name = keyword_folder + curr_date_and_time
        top_level_dir = os.path.join(results_folder, folder_name)

        soc_data_folder = "soc_time_series"
        soc_data_dir = os.path.join(top_level_dir, soc_data_folder)
        if not os.path.isdir(soc_data_dir):
            os.makedirs(soc_data_dir)
        soc_data_file = os.path.join(soc_data_dir, "soc_time_series.csv")
        soc_time_series_data = np.array(fleet_manager.data_logging.list_soc)
        np.savetxt(soc_data_file, soc_time_series_data, delimiter=",")

        demand_curve_folder = "demand_curve"
        demand_curve_dir = os.path.join(top_level_dir, demand_curve_folder)
        if not os.path.isdir(demand_curve_dir):
            os.makedirs(demand_curve_dir)
        demand_curve_data_file = os.path.join(demand_curve_dir, "fleet_demand_curve.csv")
        df_demand_curve_data = fleet_manager.data_logging.demand_curve_to_dict()
        df_demand_curve_data.to_csv(demand_curve_data_file)

        soc_dist_folder = "soc"
        soc_dist_dir = os.path.join(top_level_dir, soc_dist_folder)
        if not os.path.isdir(soc_dist_dir):
            os.makedirs(soc_dist_dir)
        soc_dist_data_file = os.path.join(soc_dist_dir, "fleet_soc_dist.csv")
        df_soc_dist = fleet_manager.data_logging.soc_dist_to_dict()
        df_soc_dist.to_csv(soc_dist_data_file)

        kpi_csv = os.path.join(top_level_dir, "kpi.csv")
        kpi.to_csv(kpi_csv)

        # Plot Results
        plot_dir = os.path.join(top_level_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        # Plotting SOC time series data for the first 20 cars
        for i in range(min(20, n_cars)):
            plt.plot(soc_time_series_data[:, i], label=f'Car {i}')
        plt.xlabel("Time (min)")
        plt.ylabel("State of charge")
        plt.title("Time series SOC data for 10 cars")
        soc_plot_file = os.path.join(plot_dir, "soc_time_series.png")
        plt.savefig(soc_plot_file)
        plt.clf()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = df_demand_curve_data["time"].to_numpy()
        soc = df_demand_curve_data["avg_soc"].to_numpy()
        stdev_soc = df_demand_curve_data["stdev_soc"].to_numpy()
        ax1.stackplot(x, np.transpose(df_demand_curve_data[[
            "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging",
            "waiting_for_charger"]].to_numpy()), colors=['b', 'tab:orange', 'g', 'tab:purple', 'r', 'y'])
        ax2.plot(x, soc, 'k', linewidth=3)
        ax2.fill_between(x, (soc - stdev_soc), (soc + stdev_soc), color='k', alpha=0.2)
        print(arrival_rate_pmin)
        lambda_times_s = arrival_rate_pmin * 0.5214 * SimMetaData.max_lon / SimMetaData.avg_vel_mph * 60
        ax1.plot(x, np.ones(len(df_demand_curve_data)) * lambda_times_s,
                 'tab:brown', linewidth=2)
        discharge_rate_by_charge_rate = (
                SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
        )
        frac_of_cars_driving = n_cars / (1 + discharge_rate_by_charge_rate)
        ax1.plot(x, np.ones(len(df_demand_curve_data)) * frac_of_cars_driving, 'tab:brown', linewidth=2)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Number of Cars")
        ax1.set_ylim([0, n_cars])
        ax2.set_ylabel("SOC")
        ax2.set_ylim([0, 1])
        plt.title("Demand Stackplot with SOC overlaid")
        demand_curve_plot_file = os.path.join(plot_dir, "demand_curve_stackplot.png")
        plt.savefig(demand_curve_plot_file)
        plt.clf()

        fig, ax = plt.subplots()
        x = df_soc_dist["time"].to_numpy()
        ax.stackplot(x, np.transpose(
            df_soc_dist[np.arange(1, 101, 5).astype(str)].to_numpy()
        ))
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Number of Cars")
        ax.set_ylim([0, n_cars])
        plt.title("SoC Distribution over Time")
        soc_dist_plot_file = os.path.join(plot_dir, "soc_dist_stackplot.png")
        plt.savefig(soc_dist_plot_file)
        plt.clf()

    print(f"Simulation Time: {time.time() - start_time} secs")
    print(service_level_percentage)
    print(service_level_percentage_second_half)
    return kpi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nev', '--fleet_size', type=int)
    parser.add_argument("-nc", "--n_chargers", type=int)
    parser.add_argument("-lambda", "--arrival_rate_p_min", type=int)
    parser.add_argument("-r", "--repeat", type=int)
    parser.add_argument("-rf", "--results_folder", type=str)
    parser.add_argument("-ps", "--pack_size_kwh", type=int, default=None)
    parser.add_argument("-np", "--n_posts", type=int, default=8)
    parser.add_argument("-mp", "--matching_policy", type=str, default="PO2")
    args = parser.parse_args()
    if args.pack_size_kwh is not None:
        SimMetaData.pack_size_kwh = args.pack_size_kwh
    if args.matching_policy == "PO2":
        d = 2
        matching_algo = MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value
    elif args.matching_policy == "CD":
        d = 1
        matching_algo = MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value
    elif args.matching_policy == "CAD":
        d = 0
        matching_algo = MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value
    else:
        raise ValueError(f"Matching algorithm {args.matching_policy} is not defined")
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(curr_dir, args.results_folder)
    dataset_path = os.path.join(curr_dir, f"data/random_data_{args.repeat}_with_arrival_rate_{args.arrival_rate_p_min}_per_min_and_sim_duration_1000_mins.csv")
    run_simulation(sim_duration=1000,
                   n_cars=args.fleet_size,
                   arrival_rate_pmin=args.arrival_rate_p_min,
                   n_chargers=args.n_chargers,
                   n_posts=args.n_posts,
                   renege_time_min=1,
                   matching_algo=matching_algo,
                   charging_algo=ChargingAlgo.CHARGE_AFTER_TRIP_END.value,
                   d=d,
                   infinite_chargers=False,
                   results_folder=results_folder,
                   trip_data_csv_path=dataset_path,
                   keyword_folder=f"nev_{args.fleet_size}_nc_{args.n_chargers}_lambda_{args.arrival_rate_p_min}_repeat_{args.repeat}_ps_{args.pack_size_kwh}_np_{args.n_posts}_"
                   )
