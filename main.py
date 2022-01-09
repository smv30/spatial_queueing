import numpy as np
import pandas as pd
import simpy
import os
import matplotlib.pyplot as plt
from datetime import datetime
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger
from sim_metadata import SimMetaData, TripState, MatchingAlgoParams


def run_simulation(
        sim_duration,
        n_cars,
        arrival_rate_pmin,
        n_chargers,
        n_posts,
        renege_time_min=None
):
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
                                 list_chargers=list_chargers)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration)

    if SimMetaData.save_results:
        # Saving KPIs and sim metadata
        total_n_trips = len(fleet_manager.list_trips)
        avg_trip_time_min = np.mean([fleet_manager.list_trips[trip].calc_trip_time() for trip in range(total_n_trips)])
        avg_pickup_time_min = np.mean([fleet_manager.list_trips[trip].pickup_time_min for trip in range(total_n_trips)])
        avg_trip_dist_mi = avg_trip_time_min / 60 * SimMetaData.avg_vel_mph

        total_n_of_successful_trips = sum([int(fleet_manager.list_trips[trip].state == TripState.MATCHED)
                                          for trip in range(total_n_trips)])
        service_level_percentage = total_n_of_successful_trips / total_n_trips * 100
        kpi = pd.DataFrame({
            "fleet_size": n_cars,
            "pack_size_kwh": SimMetaData.pack_size_kwh,
            "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
            "charge_rate_kw": SimMetaData.charge_rate_kw,
            "avg_vel_mph": SimMetaData.avg_vel_mph,
            "n_chargers": SimMetaData.n_charger_loc,
            "n_posts": SimMetaData.n_posts,
            "total_sim_duration_min": sim_duration,
            "arrival_rate_pmin": arrival_rate_pmin,
            "total_n_trips": total_n_trips,
            "avg_trip_time_min": avg_trip_time_min,
            "avg_trip_dist_mi": avg_trip_dist_mi,
            "avg_pickup_time_min": avg_pickup_time_min,
            "service_level_percentage": service_level_percentage,
            "matching_algorithm": f"Power of {MatchingAlgoParams.d}"
        }, index=[0])

        # Save Results

        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        top_level_dir = os.path.join(SimMetaData.results_folder, curr_date_and_time)

        soc_data_folder = "soc_time_series"
        soc_data_dir = os.path.join(top_level_dir, soc_data_folder)
        if not os.path.isdir(soc_data_dir):
            os.makedirs(soc_data_dir)
        soc_data_file = os.path.join(soc_data_dir, "soc_time_series.csv")
        soc_time_series_data = np.array(fleet_manager.soc_logging.list_soc)
        np.savetxt(soc_data_file, soc_time_series_data, delimiter=",")

        kpi_csv = os.path.join(top_level_dir, "kpi.csv")
        kpi.to_csv(kpi_csv)

        # Plot Results
        plot_dir = os.path.join(top_level_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        # Plotting SOC time series data for the first 10 cars
        for i in range(10):
            plt.plot(soc_time_series_data[:, i], label=f'Car {i}')
        plt.xlabel("Time (min)")
        plt.ylabel("State of charge")
        plt.title("Time series SOC data for 10 cars")
        soc_plot_file = os.path.join(plot_dir, "soc_time_series.png")
        plt.savefig(soc_plot_file)



if __name__ == "__main__":
    run_simulation(sim_duration=50000,
                   n_cars=10,
                   arrival_rate_pmin=1 / 10,
                   n_chargers=10,
                   n_posts=1,
                   renege_time_min=1
                   )

