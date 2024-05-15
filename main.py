import numpy as np
import pandas as pd
import simpy
import os
import matplotlib.pyplot as plt
import time
import csv
import json
from datetime import datetime
from datetime import timedelta
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger
from sim_metadata import SimMetaData, TripState, MatchingAlgo, ChargingAlgoParams, Dataset, DatasetParams, \
    AdaptivePowerOfDParams, ChargingAlgo, Initialize, AvailableCarsForMatching, DistFunc, PickupThresholdType
from real_life_data_input import DataInput
from ev_database import ElectricVehicleDatabase


def run_simulation(
        n_cars,
        n_chargers,
        n_posts,
        d,
        arrival_rate_pmin=None,
        pickup_threshold_type=PickupThresholdType.NO_THRESHOLD.value,
        sim_duration_min=None,
        dataset_source=None,
        dist_func=DistFunc.MANHATTAN.value,
        adaptive_d=None,
        active_threshold=None,
        start_datetime=None,
        end_datetime=None,
        matching_algo=None,
        charging_algo=None,
        available_cars_for_matching=None,
        infinite_chargers=None,
        renege_time_min=1,
        results_folder=None,
        dataset_path=None,
        ev_model=None,
):
    start_time = time.time()
    env = simpy.Environment()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    home_dir = os.path.join(current_dir, "spatial_queueing")

    if dataset_source in [Dataset.NYTAXI.value, Dataset.CHICAGO.value, Dataset.OLD_NYTAXI.value]:
        data_input = DataInput(percentile_lat_lon=DatasetParams.percentile_lat_lon)
        # call the NY taxi dataset function to get the real life dataframe
        df_arrival_sequence, dist_correction_factor = data_input.real_life_dataset(
            dataset_source=dataset_source,
            dataset_path=dataset_path,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            percent_of_trips=DatasetParams.percent_of_trips_filtered,
            dist_func=dist_func
        )

        first_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].min()
        last_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].max()
        total_sim_time_datetime = last_trip_pickup_datetime - first_trip_pickup_datetime
        sim_duration_min = int(total_sim_time_datetime.total_seconds() / 60.0) + 1

    elif dataset_source == Dataset.RANDOMLYGENERATED.value:
        data_input = DataInput()
        # call the randomly generated dataset function to get the random dataframe
        df_arrival_sequence, dist_correction_factor = data_input.randomly_generated_dataframe(
            sim_duration_min=sim_duration_min,
            arrival_rate_pmin=arrival_rate_pmin,
            data_dir=home_dir,
            start_datetime=start_datetime)
    else:
        raise ValueError("No such dataset exists")

    if infinite_chargers is not None:
        ChargingAlgoParams.infinite_chargers = infinite_chargers
        print(f"infinite_chargers is set to {infinite_chargers} manually by the user")
    if adaptive_d is not None:
        AdaptivePowerOfDParams.adaptive_d = adaptive_d
        print(f"adaptive_d is set to {adaptive_d} manually by the user")
    if ev_model is not None:
        ev_database = ElectricVehicleDatabase()
        ev_info = ev_database.get_vehicle_info(ev_model)
        SimMetaData.pack_size_kwh = ev_info.pack_size_kwh * (1 - ev_info.battery_degradation_perc)
        SimMetaData.consumption_kwhpmi = ev_info.consumption_kwhpmi
        print(f"Running simulation for {ev_info.name}:")
        print(f"Battery Degradation: {ev_info.battery_degradation_perc}")
    else:
        print(f"Running simulation for an unspecified EV")
    print(f"Battery Pack Size: {SimMetaData.pack_size_kwh} kWh (after degradation)")
    print(f"Consumption: {SimMetaData.consumption_kwhpmi} kWh/mi")

    # Initialize all supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts,
                               env=env,
                               df_arrival_sequence=df_arrival_sequence,
                               initialize_chargers=Initialize.RANDOM_DESTINATION.value)
        list_chargers.append(charger)

    # Initializing all cars
    car_tracker = []
    for car_id in range(n_cars):
        car = Car(car_id=car_id,
                  env=env,
                  list_chargers=list_chargers,
                  df_arrival_sequence=df_arrival_sequence,
                  initialize_car=Initialize.RANDOM_PICKUP.value)
        car_tracker.append(car)

    for charger in list_chargers:
        charger.car_tracker = car_tracker

    fleet_manager = FleetManager(env=env,
                                 car_tracker=car_tracker,
                                 n_cars=n_cars,
                                 renege_time_min=renege_time_min,
                                 list_chargers=list_chargers,
                                 trip_data=df_arrival_sequence,
                                 matching_algo=matching_algo,
                                 charging_algo=charging_algo,
                                 dist_correction_factor=dist_correction_factor,
                                 pickup_threshold_type=pickup_threshold_type,
                                 available_cars_for_matching=available_cars_for_matching,
                                 dist_func=dist_func,
                                 d=d)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration_min)

    # Saving KPIs and sim metadata
    total_n_trips = len(fleet_manager.list_trips)
    avg_trip_time_min = np.mean([fleet_manager.list_trips[trip].trip_time_min for trip in range(total_n_trips)])
    avg_trip_dist_mi = np.mean([fleet_manager.list_trips[trip].trip_distance_mi for trip in range(total_n_trips)])

    total_n_of_successful_trips = sum([int(fleet_manager.list_trips[trip].state == TripState.MATCHED.value)
                                       for trip in range(total_n_trips)])
    list_pickup_time_min = [fleet_manager.list_trips[trip].pickup_time_min
                            for trip in range(total_n_trips)
                            if fleet_manager.list_trips[trip].pickup_time_min != 0
                            ]
    avg_pickup_time_min = sum(list_pickup_time_min) / total_n_of_successful_trips
    service_level_percentage = total_n_of_successful_trips / total_n_trips * 100

    avg_soc = sum([car_tracker[car].soc for car in range(n_cars)]) / n_cars
    avg_n_of_charging_trips = (
            sum([car_tracker[car].n_of_charging_stops for car in range(n_cars)])
            / n_cars
            / (sim_duration_min / 60)
    )
    avg_drive_to_charger_time_min = sum(
        car_tracker[car].total_drive_to_charge_time for car in range(n_cars)
    ) / sum(car_tracker[car].n_of_charging_stops for car in range(n_cars))

    current_min_list = []
    num_trips_in_progress_list = []
    df_arrival_sequence["list_pickup_time"] = np.array([
        timedelta(minutes=fleet_manager.list_trips[trip].pickup_time_min) for trip in range(total_n_trips)
    ])
    df_arrival_sequence["actual_pickup_time"] = (pd.to_datetime(df_arrival_sequence["pickup_datetime"]) +
                                                 df_arrival_sequence["list_pickup_time"])
    df_arrival_sequence["actual_dropoff_time"] = (pd.to_datetime(df_arrival_sequence["dropoff_datetime"]) +
                                                  df_arrival_sequence["list_pickup_time"])
    start_year = start_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day
    start_hour = start_datetime.hour
    start_minute = start_datetime.minute
    for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
        current_min_total = (current_index * SimMetaData.demand_curve_res_min
                             + start_day * 1440 + start_hour * 60 + start_minute)
        current_day = int(current_min_total / 1440)
        current_hour = int(current_min_total % 1440 / 60)
        current_min = current_min_total - current_day * 1440 - current_hour * 60
        trips_in_progress = df_arrival_sequence[
            (df_arrival_sequence["actual_pickup_time"] <= datetime(start_year, start_month, current_day,
                                                                   current_hour, current_min, 0)) &
            (df_arrival_sequence["actual_dropoff_time"] > datetime(start_year, start_month, current_day,
                                                                   current_hour, current_min, 0))
            ]
        num_trips_in_progress = len(trips_in_progress)
        if num_trips_in_progress > n_cars:
            num_trips_in_progress = n_cars
        current_min_list.append(current_index * SimMetaData.demand_curve_res_min)
        num_trips_in_progress_list.append(num_trips_in_progress)

    total_incoming_workload = sum(num_trips_in_progress_list)
    df_demand_curve_data = fleet_manager.data_logging.demand_curve_to_dict()
    df_demand_curve_data["delta_time"] = df_demand_curve_data["time"].shift(-1) - df_demand_curve_data["time"]
    df_demand_curve_data["delta_time"] = df_demand_curve_data["delta_time"].fillna(0)
    total_served_workload = (df_demand_curve_data["delta_time"] * df_demand_curve_data["driving_with_passenger"]).sum()
    percentage_workload_served = total_served_workload / total_incoming_workload

    kpi = pd.DataFrame({
        "ev_type": ev_model,
        "charge_rate_kw": SimMetaData.charge_rate_kw,
        "matching_algorithm": matching_algo,
        "charging_algorithm": charging_algo,
        "available_ev_for_matching": available_cars_for_matching,
        "d": d,
        "adaptive_d": adaptive_d,
        "fleet_size": n_cars,
        "n_chargers": n_chargers,
        "pack_size_kwh": SimMetaData.pack_size_kwh,
        "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
        "avg_vel_mph": SimMetaData.avg_vel_mph,
        "n_posts": n_posts,
        "total_sim_duration_min": sim_duration_min,
        "arrival_rate_pmin": arrival_rate_pmin,
        "total_n_trips": total_n_trips,
        "avg_trip_time_min": avg_trip_time_min,
        "avg_trip_dist_mi": avg_trip_dist_mi,
        "avg_pickup_time_min": avg_pickup_time_min,
        "avg_drive_time_to_charger": avg_drive_to_charger_time_min,
        "number_of_trips_to_charger_per_car_per_hr": avg_n_of_charging_trips,
        "avg_soc_over_time_over_cars": avg_soc,
        "service_level_percentage": service_level_percentage,
        "percentage_workload_served": percentage_workload_served
    }, index=[0])
    if SimMetaData.save_results:
        # Save Results
        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        top_level_dir = os.path.join(results_folder, curr_date_and_time)

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

        kpi_csv = os.path.join(top_level_dir, "kpi.csv")
        kpi.to_csv(kpi_csv)

        trips_in_progress_csv = os.path.join(top_level_dir, "trips_in_progress.csv")
        with open(trips_in_progress_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["time (min)", "number of trips"])
            writer.writerows(zip(current_min_list, num_trips_in_progress_list))

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

        plt.hist(list_pickup_time_min)
        plt.xlabel("Frequency")
        plt.ylabel("Pickup Time (min)")
        pickup_time_plot_file = os.path.join(plot_dir, "pickup_time_hist.png")
        plt.savefig(pickup_time_plot_file)
        plt.clf()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = df_demand_curve_data["time"].to_numpy()
        soc = df_demand_curve_data["avg_soc"].to_numpy()
        ax1.stackplot(x, np.transpose(df_demand_curve_data[[
            "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging",
            "waiting_for_charger"]].to_numpy()),
                      colors=['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#E6AB02', '#036c5f'])
        ax2.plot(x, soc, 'k', linewidth=3)
        ax1.set_xlabel("Time (min)", fontsize=18)
        ax1.set_ylabel("Number of Cars", fontsize=18)
        ax1.set_ylim([0, n_cars])
        ax2.set_ylabel("SOC")
        ax2.set_ylim([0, 1])
        ax1.plot(current_min_list, num_trips_in_progress_list, 'm')
        plt.title("Demand Stackplot with SOC overlaid")
        demand_curve_plot_file = os.path.join(plot_dir, "demand_curve_stackplot.png")
        plt.savefig(demand_curve_plot_file)
        plt.clf()

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = df_demand_curve_data["time"].to_numpy()
        soc = df_demand_curve_data["avg_soc"].to_numpy()
        n_charging_and_idle = (df_demand_curve_data["idle"] + df_demand_curve_data["charging"]).to_numpy()
        n_cars_available = df_demand_curve_data["n_cars_available"].to_numpy()
        ax1.plot(x, n_charging_and_idle, 'b')
        ax1.plot(x, n_cars_available, 'g')
        ax2.plot(x, soc, 'k', linewidth=3)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Number of Cars")
        ax1.set_ylim([0, n_cars])
        ax2.set_ylabel("SOC")
        ax2.set_ylim([0, 1])
        plt.title("n_available, n_charging_idle and avg_soc")
        demand_curve_plot_file = os.path.join(plot_dir, "n_available_and_n_charging_idle.png")
        plt.savefig(demand_curve_plot_file)
        plt.clf()

        fig, ax = plt.subplots()
        x = df_demand_curve_data["n_cars_available"].to_numpy()
        y = df_demand_curve_data["pickup_time_min"].to_numpy()
        ax.plot(x, y)
        ax.set_xlabel("Number of Cars Available")
        ax.set_ylabel("Time (min)")
        ax.set_xlim([0, n_cars])
        plt.title("pickup_time_min vs. n_cars_available")
        demand_curve_plot_file = os.path.join(plot_dir, "pickup_time_and_n_cars_available.png")
        plt.savefig(demand_curve_plot_file)
        plt.clf()

    print(f"Simulation Time: {time.time() - start_time} secs")
    return kpi


if __name__ == "__main__":
    # ev_database = ElectricVehicleDatabase()
    # ev_database.add_vehicle(name="Tesla Model 3",
    #                         pack_size_kwh=57.5,
    #                         consumption_kwhpmi=0.25,
    #                         battery_degradation_perc=0.1
    #                         )

    # Uncomment the above if you want to add another EV to the database or add them directly to ev_database.py
    run_simulation(n_cars=750,
                   n_chargers=200,
                   n_posts=1,
                   d=1,
                   dataset_source=Dataset.CHICAGO.value,
                   start_datetime=datetime(2022, 6, 14, 0, 0, 0),
                   end_datetime=datetime(2022, 6, 17, 0, 0, 0),
                   matching_algo=MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value,
                   charging_algo=ChargingAlgo.CHARGE_ALL_IDLE_CARS.value,
                   available_cars_for_matching=AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value,
                   pickup_threshold_type=PickupThresholdType.EITHER_PERCENT_OR_CONSTANT.value,
                   adaptive_d=True,
                   infinite_chargers=False,
                   dist_func=DistFunc.MANHATTAN.value,
                   results_folder="simulation_results/",
                   dataset_path='/Users/sushilvarma/PycharmProjects/SpatialQueueing/Chicago_year_2022_month_06.csv',
                   ev_model="Tesla Model 3"
                   )
