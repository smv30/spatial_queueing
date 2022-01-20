import numpy as np
import pandas as pd
import simpy
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger
from sim_metadata import SimMetaData, TripState, MatchingAlgo


def run_simulation(
        sim_duration,
        n_cars,
        arrival_rate_pmin,
        n_chargers,
        n_posts,
        d,
        matching_algo=MatchingAlgo.POWER_OF_D_IDLE.value,
        renege_time_min=None,
        results_folder=None,
):
    start_time = time.time()
    env = simpy.Environment()

    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts)
        list_chargers.append(charger)

    # Initializing all the cars
    car_tracker = []
    for car_id in range(n_cars):
        car = Car(car_id=car_id,
                  env=env,
                  list_chargers=list_chargers)
        car_tracker.append(car)

    fleet_manager = FleetManager(arrival_rate_pmin=arrival_rate_pmin,
                                 env=env,
                                 car_tracker=car_tracker,
                                 n_cars=n_cars,
                                 renege_time_min=renege_time_min,
                                 list_chargers=list_chargers,
                                 matching_algo=matching_algo,
                                 d=d)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration)

    # Saving KPIs and sim metadata
    total_n_trips = len(fleet_manager.list_trips)
    avg_trip_time_min = np.mean([fleet_manager.list_trips[trip].calc_trip_time() for trip in range(total_n_trips)])
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
            / (sim_duration / 60)
    )
    avg_drive_to_charger_time_min = sum(
        car_tracker[car].total_drive_to_charge_time for car in range(n_cars)
    ) / sum(car_tracker[car].n_of_charging_stops for car in range(n_cars))
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
        "avg_trip_dist_mi": avg_trip_dist_mi,
        "avg_pickup_time_min": avg_pickup_time_min,
        "avg_drive_time_to_charger": avg_drive_to_charger_time_min,
        "number_of_trips_to_charger_per_car_per_hr": avg_n_of_charging_trips,
        "avg_soc_over_time_over_cars": avg_soc,
        "service_level_percentage": service_level_percentage,
        "matching_algorithm": f"Power of {d}"
    }, index=[0])
    if SimMetaData.save_results:
        # Save Results

        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        top_level_dir = os.path.join(results_folder, curr_date_and_time)

        soc_time_series_data = np.array(fleet_manager.data_logging.list_soc)
        df_demand_curve_data = fleet_manager.data_logging.demand_curve_to_dict()

        # Plot Results
        save_and_plot_results(df_demand_curve_data=df_demand_curve_data,
                              soc_time_series_data=soc_time_series_data,
                              kpi=kpi,
                              top_level_dir=top_level_dir,
                              n_cars=n_cars,
                              arrival_rate_pmin=arrival_rate_pmin
                              )

    print(f"Simulation Time: {int((time.time() - start_time) / 60)} mins")
    return kpi


def save_and_plot_results(df_demand_curve_data, soc_time_series_data, kpi, top_level_dir, n_cars, arrival_rate_pmin):
    # Save Data
    soc_data_folder = "soc_time_series"
    soc_data_dir = os.path.join(top_level_dir, soc_data_folder)
    if not os.path.isdir(soc_data_dir):
        os.makedirs(soc_data_dir)
    soc_data_file = os.path.join(soc_data_dir, "soc_time_series.csv")
    np.savetxt(soc_data_file, soc_time_series_data, delimiter=",")

    demand_curve_folder = "demand_curve"
    demand_curve_dir = os.path.join(top_level_dir, demand_curve_folder)
    if not os.path.isdir(demand_curve_dir):
        os.makedirs(demand_curve_dir)
    demand_curve_data_file = os.path.join(demand_curve_dir, "fleet_demand_curve.csv")
    df_demand_curve_data.to_csv(demand_curve_data_file)

    kpi_csv = os.path.join(top_level_dir, "kpi.csv")
    kpi.to_csv(kpi_csv)

    # Plot Data
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
        "driving_with_passenger", "driving_without_passenger", "idle", "driving_to_charger", "charging"
    ]].to_numpy()), colors=['b', 'tab:orange', 'g', 'tab:purple', 'r'])
    ax2.plot(x, soc, 'k', linewidth=3)
    ax2.fill_between(x, (soc - stdev_soc), (soc + stdev_soc), color='k', alpha=0.2)
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


if __name__ == "__main__":
    run_simulation(sim_duration=500,
                   n_cars=10,
                   arrival_rate_pmin=1 / 10,
                   n_chargers=10,
                   n_posts=1,
                   renege_time_min=1,
                   matching_algo=MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value
                   )

