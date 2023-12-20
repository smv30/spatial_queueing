import numpy as np
import pandas as pd
import simpy
import os
import matplotlib.pyplot as plt
import time
import csv
from datetime import datetime
from datetime import timedelta
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger
from sim_metadata import SimMetaData, TripState, MatchingAlgo, ChargingAlgoParams, Dataset, DatasetParams
from spatial_queueing.real_life_data_input import DataInput


def run_simulation(
        sim_duration_min,
        n_cars,
        arrival_rate_pmin,
        n_chargers,
        n_posts,
        d,
        dataset=None,
        start_datetime=None,
        end_datetime=None,
        matching_algo=None,
        infinite_chargers=None,
        renege_time_min=None,
        results_folder=None,
        dataset_path=None
):
    start_time = time.time()
    env = simpy.Environment()

    if dataset == Dataset.NYTAXI.value:
        data_input = DataInput(percentile_lat_lon=DatasetParams.percentile_lat_lon)
        # call the NY taxi dataset function to get the real life dataframe
        df_arrival_sequence, correction_factor = data_input.ny_taxi_dataset(
            dataset_path=dataset_path,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            percent_of_trips=SimMetaData.percent_of_trips)

        first_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].min()
        last_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].max()
        total_sim_time_sec = last_trip_pickup_datetime - first_trip_pickup_datetime
        sim_duration_min = int(total_sim_time_sec.total_seconds() / 60.0) + 1

        # max_longitude = df_arrival_sequence[["pickup_longitude", "dropoff_longitude"]].max().max()
        # max_latitude = df_arrival_sequence[["pickup_latitude", "dropoff_latitude"]].max().max()
        # min_longitude = df_arrival_sequence[["pickup_longitude", "dropoff_longitude"]].min().min()
        # min_latitude = df_arrival_sequence[["pickup_latitude", "dropoff_latitude"]].min().min()
        # SimMetaData.max_lon = max_longitude
        # SimMetaData.max_lat = max_latitude
        # SimMetaData.min_lon = min_longitude
        # SimMetaData.min_lat = min_latitude

    elif dataset == Dataset.RANDOMLYGENERATED.value:
        data_input = DataInput()
        # call the randomly generated dataset function to get the random dataframe
        df_arrival_sequence, correction_factor = data_input.randomly_generated_dataframe(
            sim_duration_min=sim_duration_min,
            arrival_rate_pmin=arrival_rate_pmin,
            data_dir=SimMetaData.home_dir)
    else:
        raise ValueError("No such dataset exists")

    if infinite_chargers is not None:
        ChargingAlgoParams.infinite_chargers = infinite_chargers

    # Initialize all supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts,
                               env=env)
        list_chargers.append(charger)

    # Initializing all cars
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
                                 dataset=dataset,
                                 trip_data=df_arrival_sequence,
                                 matching_algo=matching_algo,
                                 correction_factor=correction_factor,
                                 d=d)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration_min)

    # Saving KPIs and sim metadata
    total_n_trips = len(fleet_manager.list_trips)
    avg_trip_time_min = np.mean([fleet_manager.list_trips[trip].trip_time_min for trip in range(total_n_trips)])
    avg_trip_dist_mi = avg_trip_time_min / 60 * SimMetaData.avg_vel_mph

    total_n_of_successful_trips = sum([int(fleet_manager.list_trips[trip].state == TripState.MATCHED)
                                       for trip in range(total_n_trips)])
    avg_pickup_time_min = sum(
        [fleet_manager.list_trips[trip].pickup_time_min for trip in range(total_n_trips)]
    ) / total_n_of_successful_trips
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
    if dataset == Dataset.NYTAXI.value:
        df_arrival_sequence["list_pickup_time"] = np.array([
            timedelta(minutes=fleet_manager.list_trips[trip].pickup_time_min) for trip in range(total_n_trips)
        ])
        df_arrival_sequence["actual_pickup_time"] = pd.to_datetime(df_arrival_sequence["pickup_datetime"]) + \
                                                    df_arrival_sequence["list_pickup_time"]
        df_arrival_sequence["actual_dropoff_time"] = pd.to_datetime(df_arrival_sequence["dropoff_datetime"]) + \
                                                     df_arrival_sequence["list_pickup_time"]
        for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
            current_min_total = current_index * SimMetaData.demand_curve_res_min
            current_hour = int(current_min_total / 60)
            current_min_minus_hour = current_min_total - current_hour * 60
            trips_in_progress = df_arrival_sequence[
                (df_arrival_sequence["actual_pickup_time"] <= datetime(2010, 12, 1, current_hour,
                                                                       current_min_minus_hour,
                                                                       0)) &
                (df_arrival_sequence["actual_dropoff_time"] > datetime(2010, 12, 1, current_hour,
                                                                       current_min_minus_hour,
                                                                       0))
                ]
            num_trips_in_progress = len(trips_in_progress)
            if num_trips_in_progress > n_cars:
                num_trips_in_progress = n_cars
            current_min_list.append(current_min_total)
            num_trips_in_progress_list.append(num_trips_in_progress)
    elif dataset == Dataset.RANDOMLYGENERATED.value:
        df_arrival_sequence["list_pickup_time"] = np.array([
            fleet_manager.list_trips[trip].pickup_time_min for trip in range(total_n_trips)
        ])
        df_arrival_sequence["actual_pickup_time"] = (df_arrival_sequence["pickup_datetime"] +
                                                     df_arrival_sequence["list_pickup_time"])
        df_arrival_sequence["actual_dropoff_time"] = (df_arrival_sequence["dropoff_datetime"] +
                                                      df_arrival_sequence["list_pickup_time"])
        for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
            current_min = current_index * SimMetaData.demand_curve_res_min
            trips_in_progress = df_arrival_sequence[(df_arrival_sequence["actual_pickup_time"] <= current_min) &
                                                    (df_arrival_sequence["actual_dropoff_time"] > current_min)]
            num_trips_in_progress = len(trips_in_progress)
            if num_trips_in_progress > n_cars:
                num_trips_in_progress = n_cars
            current_min_list.append(current_min)
            num_trips_in_progress_list.append(num_trips_in_progress)

    pink_area = sum(num_trips_in_progress_list)
    df_demand_curve_data = fleet_manager.data_logging.demand_curve_to_dict()
    df_demand_curve_data["delta_time"] = df_demand_curve_data["time"].shift(-1) - df_demand_curve_data["time"]
    df_demand_curve_data["delta_time"].fillna(0, inplace=True)
    blue_area = (df_demand_curve_data["delta_time"] * df_demand_curve_data["driving_with_passenger"]).sum()
    work_load = blue_area / pink_area

    kpi = pd.DataFrame({
        "fleet_size": n_cars,
        "pack_size_kwh": SimMetaData.pack_size_kwh,
        "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
        "charge_rate_kw": SimMetaData.charge_rate_kw,
        "avg_vel_mph": SimMetaData.avg_vel_mph,
        "n_chargers": n_chargers,
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
        "matching_algorithm": f"Power of {d}",
        "work_load": work_load
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
        # lambda_times_s = arrival_rate_pmin * 0.5214 * DatasetParams.longitude_range_max / SimMetaData.avg_vel_mph * 60
        # ax1.plot(x, np.ones(len(df_demand_curve_data)) * lambda_times_s,
        #          'tab:brown', linewidth=2)
        # discharge_rate_by_charge_rate = (
        #         SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
        # )
        # frac_of_cars_driving = n_cars / (1 + discharge_rate_by_charge_rate)
        # ax1.plot(x, np.ones(len(df_demand_curve_data)) * frac_of_cars_driving, 'tab:brown', linewidth=2)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Number of Cars")
        ax1.set_ylim([0, n_cars])
        ax2.set_ylabel("SOC")
        ax2.set_ylim([0, 1])
        ax1.plot(current_min_list, num_trips_in_progress_list, 'm')

        plt.title("Demand Stackplot with SOC overlaid")
        demand_curve_plot_file = os.path.join(plot_dir, "demand_curve_stackplot.png")
        plt.savefig(demand_curve_plot_file)
        plt.clf()

    print(f"Simulation Time: {time.time() - start_time} secs")
    return kpi


if __name__ == "__main__":
    run_simulation(sim_duration_min=1000,
                   n_cars=3000,
                   arrival_rate_pmin=50,
                   n_chargers=520,
                   n_posts=8,
                   d=25,
                   dataset=Dataset.NYTAXI.value,
                   start_datetime=datetime(2010, 12, 1, 0, 0, 0),
                   end_datetime=datetime(2010, 12, 2, 0, 0, 0),
                   matching_algo=MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value,
                   infinite_chargers=False,
                   renege_time_min=1,
                   results_folder="simulation_results/",
                   dataset_path='/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/yellow_tripdata_2010-12.parquet'
                   )
