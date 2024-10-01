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
    AdaptivePowerOfDParams, ChargingAlgo, Initialize, AvailableCarsForMatching, DistFunc, PickupThresholdType, \
    PickupThresholdMatchingParams
from real_life_data_input import DataInput
from ev_database import ElectricVehicleDatabase
import argparse
from distutils.util import strtobool
from mpl_toolkits.basemap import Basemap


def run_simulation(
        n_cars,
        n_chargers,
        n_posts,
        d,
        arrival_rate_pmin=None,
        pickup_threshold_type=PickupThresholdType.NO_THRESHOLD.value,
        pickup_threshold_min=None,
        sim_duration_min=None,
        charge_rate_kw=None,
        dataset_source=None,
        uniform_locations=None,
        perc_trip_filter=None,
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
    if uniform_locations is not None:
        DatasetParams.uniform_locations = uniform_locations
    if perc_trip_filter is not None:
        DatasetParams.percent_of_trips_filtered = perc_trip_filter
        print(f"{perc_trip_filter} percentage of trips are filtered")
    if DatasetParams.uniform_locations is True:
        print(f"The locations are resampled uniformly at random in the dataset")

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
        df_arrival_sequence, dist_correction_factor = data_input.randomly_generated_dataframe(
            sim_duration_min=sim_duration_min,
            arrival_rate_pmin=arrival_rate_pmin,
            data_dir=home_dir,
            start_datetime=start_datetime
        )
    else:
        raise ValueError("No such dataset exists")

    if infinite_chargers is not None:
        ChargingAlgoParams.infinite_chargers = infinite_chargers
        print(f"infinite_chargers is set to {infinite_chargers} manually by the user")
    if adaptive_d is not None:
        AdaptivePowerOfDParams.adaptive_d = adaptive_d
        print(f"adaptive_d is set to {adaptive_d} manually by the user")
    if charge_rate_kw is not None:
        SimMetaData.charge_rate_kw = charge_rate_kw
        print(f"Charge rate is set to {charge_rate_kw} kW manually by the user")
    if pickup_threshold_min is not None:
        PickupThresholdMatchingParams.threshold_min = pickup_threshold_min
        print(f"Any trips with pickup time more than {pickup_threshold_min} min will be dropped")
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

    df_arrival_sequence["pickup_time"] = pd.to_datetime(df_arrival_sequence["pickup_datetime"])
    df_arrival_sequence["dropoff_time"] = pd.to_datetime(df_arrival_sequence["dropoff_datetime"])
    peak_demand = 0
    for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
        current_min_total = (current_index * SimMetaData.demand_curve_res_min
                             + start_datetime.day * 1440 + start_datetime.hour * 60 + start_datetime.minute)
        current_day = int(current_min_total / 1440)
        current_hour = int(current_min_total % 1440 / 60)
        current_min = current_min_total - current_day * 1440 - current_hour * 60
        num_trips_in_progress = len(df_arrival_sequence[
                                        (df_arrival_sequence["pickup_time"] <= datetime(start_datetime.year,
                                                                                        start_datetime.month,
                                                                                        current_day,
                                                                                        current_hour, current_min, 0)) &
                                        (df_arrival_sequence["dropoff_time"] > datetime(start_datetime.year,
                                                                                        start_datetime.month,
                                                                                        current_day,
                                                                                        current_hour, current_min, 0))
                                        ])
        peak_demand = max(peak_demand, num_trips_in_progress)
    n_cars = peak_demand
    print(f"Number of cars is set to {n_cars} based on the peak demand.")

    n_chargers = int(n_cars * SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw)
    print(f"Number of posts is set to {n_posts}. Number of chargers is set to {n_chargers}.")

    # Initialize all supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts,
                               env=env,
                               df_arrival_sequence=df_arrival_sequence,
                               initialize_chargers=Initialize.RANDOM_UNIFORM.value)
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
                                 start_datetime=start_datetime,
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

    workload_soc_penalty = (
                                   df_demand_curve_data["avg_soc"][len(df_demand_curve_data) - 1]
                                   - df_demand_curve_data["avg_soc"][0]
                           ) * SimMetaData.pack_size_kwh / SimMetaData.charge_rate_kw * 60 / total_incoming_workload

    kpi = pd.DataFrame({
        "ev_type": ev_model,
        "dataset_source": dataset_source,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "charge_rate_kw": SimMetaData.charge_rate_kw,
        "matching_algorithm": matching_algo,
        "charging_algorithm": charging_algo,
        "available_ev_for_matching": available_cars_for_matching,
        "d": d,
        "adaptive_d": adaptive_d,
        "fleet_size": n_cars,
        "n_chargers": n_chargers,
        "n_posts": n_posts,
        "pack_size_kwh": SimMetaData.pack_size_kwh,
        "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
        "avg_vel_mph": SimMetaData.avg_vel_mph,
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
        "percentage_workload_served": percentage_workload_served,
        "workload_soc_penalty": workload_soc_penalty
    }, index=[0])

    if SimMetaData.save_results:
        # Save Results
        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        top_level_dir = os.path.join(results_folder, f"{curr_date_and_time}_{d}_{ev_model}_ev_{n_cars}_nc_{n_chargers}")
        plot_dir = os.path.join(top_level_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # soc_data_folder = "soc_time_series"
        # soc_data_dir = os.path.join(top_level_dir, soc_data_folder)
        # if not os.path.isdir(soc_data_dir):
        #     os.makedirs(soc_data_dir)
        # soc_data_file = os.path.join(soc_data_dir, "soc_time_series.csv")
        # soc_time_series_data = np.array(fleet_manager.data_logging.list_soc)
        # np.savetxt(soc_data_file, soc_time_series_data, delimiter=",")

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

        # Plotting histogram of pickup times
        plt.hist(list_pickup_time_min)
        plt.ylabel("Frequency")
        plt.xlabel("Pickup Time (min)")
        pickup_time_plot_file = os.path.join(plot_dir, "pickup_time_hist.png")
        plt.savefig(pickup_time_plot_file)
        plt.clf()

        # Stack plot of the state of the EVs with SoC overlaid
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = df_demand_curve_data["time"].to_numpy()
        soc = df_demand_curve_data["avg_soc"].to_numpy()
        ax1.stackplot(x, np.transpose(df_demand_curve_data[[
            "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging",
            "waiting_for_charger"]].to_numpy()),
                      colors=['#1F77B4', '#FC8D62', '#2CA02C', '#9467BD', '#E6AB02', '#036c5f'])
        ax2.plot(x, soc, 'k', linewidth=3)
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

        # Scatter of available EVs versus pickup times
        list_n_available_cars_to_match = [fleet_manager.list_trips[trip].n_available_cars_to_match
                                          for trip in range(total_n_trips)
                                          if fleet_manager.list_trips[trip].pickup_time_min != 0
                                          ]
        bins = [n_cars / 20 * i for i in range(21)]
        list_mean_pickup_times = []
        list_std_pickup_times = []
        list_pickup_datapoints = []
        # Iterate over bins
        for i in range(len(bins) - 1):
            # Find indices of values in list_n_available_cars_to_match within bin range
            indices = [idx for idx, val in enumerate(list_n_available_cars_to_match) if bins[i] < val <= bins[i + 1]]
            # Calculate mean of corresponding values in list_pickup_time
            list_mean_pickup_times.append(np.mean([list_pickup_time_min[idx] for idx in indices]))
            list_std_pickup_times.append(np.std([list_pickup_time_min[idx] for idx in indices]))
            list_pickup_datapoints.append(len(indices))

        plt.errorbar(bins[0:len(bins) - 1], list_mean_pickup_times, list_std_pickup_times, linestyle='None', marker='^')
        plt.xlabel("n_available_cars_to_match")
        plt.ylabel("pickup_times_min")
        pickup_time_plot_file = os.path.join(plot_dir, "pickup_time_vs_available_cars.png")
        plt.savefig(pickup_time_plot_file)
        plt.clf()

        # Save the pickup time data
        pickup_time_data_file = os.path.join(top_level_dir, "pickup_time_vs_available_cars.csv")
        pd.DataFrame({
            "n_available_cars": bins[0:len(bins) - 1],
            "mean_pickup_min": list_mean_pickup_times,
            "std_pickup_min": list_std_pickup_times,
            "count_datapoints": list_pickup_datapoints
        }).to_csv(pickup_time_data_file)

        # Scatter of available chargers versus drive to the charger time
        list_n_available_chargers_to_match = []
        list_drive_to_charger_time_min = []
        list_n_available_posts = []
        list_n_available_posts_with_driving_cars = []
        file_name = ["drive_to_charger_time_vs_available_chargers", "drive_to_charger_time_vs_available_posts",
                     "drive_to_charger_time_vs_available_posts_with_driving_cars"]
        count = 0
        for car in car_tracker:
            list_n_available_chargers_to_match.extend(car.list_n_available_chargers)
            list_drive_to_charger_time_min.extend(car.list_drive_to_charger_time_min)
            list_n_available_posts.extend(car.list_n_available_posts)
            list_n_available_posts_with_driving_cars.extend(
                [posts - ChargingAlgoParams.n_cars_driving_to_charger_discounter * to_charger for posts, to_charger in
                 zip(car.list_n_available_posts, car.list_n_cars_driving_to_charger)])
        for x_data in [list_n_available_chargers_to_match, list_n_available_posts,
                       list_n_available_posts_with_driving_cars]:
            bins = [max(x_data) / 20 * i for i in range(21)]
            list_drive_to_charger_datapoints = []
            list_mean_drive_to_charger_time_min = []
            list_std_drive_to_charger_time_min = []
            # Iterate over bins
            for i in range(len(bins) - 1):
                # Find indices of values in list_n_available_cars_to_match within bin range
                indices = [idx for idx, val in enumerate(x_data) if bins[i] < val <= bins[i + 1]]
                # Calculate mean of corresponding values in list_pickup_time
                list_mean_drive_to_charger_time_min.append(
                    np.mean([list_drive_to_charger_time_min[idx] for idx in indices]))
                list_std_drive_to_charger_time_min.append(
                    np.std([list_drive_to_charger_time_min[idx] for idx in indices]))
                list_drive_to_charger_datapoints.append(len(indices))

            plt.errorbar(bins[0:len(bins) - 1], list_mean_drive_to_charger_time_min, list_std_drive_to_charger_time_min,
                         linestyle='None', marker='^')
            plt.xlabel("n_available_chargers_to_match")
            plt.ylabel("drive_to_charger_time_min")
            drive_to_charger_time_plot_file = os.path.join(plot_dir, f"{file_name[count]}.png")
            plt.savefig(drive_to_charger_time_plot_file)
            plt.clf()

            # Save the drive to charger time data
            drive_to_charger_time_data_file = os.path.join(top_level_dir, f"{file_name[count]}.csv")
            pd.DataFrame({
                "n_available_chargers": bins[0:len(bins) - 1],
                "mean_drive_to_charger_min": list_mean_drive_to_charger_time_min,
                "std_drive_to_charger_min": list_std_drive_to_charger_time_min,
                "count_datapoints": list_drive_to_charger_datapoints
            }).to_csv(drive_to_charger_time_data_file)
            count += 1

        # Spatial Plots
        if dataset_source in [Dataset.NYTAXI.value, Dataset.OLD_NYTAXI.value]:
            city_name = "New York City"
        elif dataset_source == Dataset.CHICAGO.value:
            city_name = "Chicago"
        else:
            city_name = "Random City"

        # Get latitude and longitude data
        list_charger_lat = [charger.lat for charger in list_chargers]
        list_charger_lon = [charger.lon for charger in list_chargers]

        # Create a map of the city
        plt.figure(figsize=(10, 8))
        if dataset_source in [Dataset.NYTAXI.value, Dataset.OLD_NYTAXI.value]:
            m = Basemap(projection='merc', llcrnrlat=40.48, urcrnrlat=40.92, llcrnrlon=-74.26, urcrnrlon=-73.7,
                        resolution='h')
        elif dataset_source == Dataset.CHICAGO.value:
            m = Basemap(projection='merc', llcrnrlat=41.6, urcrnrlat=42, llcrnrlon=-87.9, urcrnrlon=-87.5, resolution='h')
        else:
            m = Basemap(projection='merc', llcrnrlat=DatasetParams.latitude_range_min,
                        urcrnrlat=DatasetParams.latitude_range_max, llcrnrlon=DatasetParams.longitude_range_min,
                        urcrnrlon=DatasetParams.longitude_range_max, resolution='h')
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='lightgray', lake_color='aqua')

        # Convert latitude and longitude to map coordinates
        x, y = m(list_charger_lon, list_charger_lat)

        # Plot the data as a scatter plot
        m.scatter(x, y, marker='o', color='r', alpha=0.7)

        # Add title and show plot
        plt.title(f'Scatter Plot of Latitude and Longitude in {city_name}')
        pickup_time_plot_file = os.path.join(plot_dir, "charger_locations.png")
        plt.savefig(pickup_time_plot_file)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-ev', '--ev_type', type=str, default="Tesla_Model_3")
    parser.add_argument('-ckw', '--charge_rate_kw', type=int, default=20)
    parser.add_argument('-ma', '--matching_algorithm', type=int, default=MatchingAlgo.POWER_OF_D.value)
    parser.add_argument('-ca', '--charging_algorithm', type=int, default=ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value)
    parser.add_argument('-aev', '--available_ev_for_matching', type=int,
                        default=AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value)
    parser.add_argument('-d', '--power_of_d', type=float, default=5)
    parser.add_argument('-ad', '--adaptive_d', type=str, default="True")  # Accepts string values
    parser.add_argument('-nev', '--n_cars', type=int, default=2200)
    parser.add_argument('-nc', '--n_chargers', type=int, default=500)
    parser.add_argument('-pt', '--perc_trip_filter', type=float, default=0.6)
    parser.add_argument('-l', '--bool_uniform_loc', type=str, default="False")
    parser.add_argument('-t', '--pickup_threshold_min', type=int, default=45)
    parser.add_argument('-rf', '--results_folder', type=str, default="simulation_results")
    args = parser.parse_args()

    # Convert 'adaptive_d' argument to boolean
    args.adaptive_d = bool(strtobool(args.adaptive_d))

    # Convert "bool_uniform_loc" argument to boolean
    args.bool_uniform_loc = bool(strtobool(args.bool_uniform_loc))

    if args.pickup_threshold_min == 0:
        input_pickup_threshold_type = PickupThresholdType.NO_THRESHOLD.value
    else:
        input_pickup_threshold_type = PickupThresholdType.CONSTANT_THRESHOLD.value

    # Convert d to integer if it is xyz.0
    if args.power_of_d % 1 == 0:
        args.power_of_d = int(args.power_of_d)
    run_simulation(n_cars=args.n_cars,
                   n_chargers=args.n_chargers,
                   n_posts=4,
                   d=args.power_of_d,
                   charge_rate_kw=args.charge_rate_kw,
                   dataset_source=Dataset.NYTAXI.value,
                   uniform_locations=args.bool_uniform_loc,
                   start_datetime=datetime(2024, 5, 1, 0, 0, 0),
                   end_datetime=datetime(2024, 5, 4, 0, 0, 0),
                   matching_algo=args.matching_algorithm,
                   charging_algo=args.charging_algorithm,
                   available_cars_for_matching=args.available_ev_for_matching,
                   pickup_threshold_type=input_pickup_threshold_type,
                   adaptive_d=args.adaptive_d,
                   perc_trip_filter=args.perc_trip_filter,
                   pickup_threshold_min=args.pickup_threshold_min,
                   infinite_chargers=False,
                   dist_func=DistFunc.MANHATTAN.value,
                   results_folder=f"simulation_results",
                   dataset_path='yellow_tripdata_2024-05.parquet',
                   ev_model=args.ev_type
                   )
