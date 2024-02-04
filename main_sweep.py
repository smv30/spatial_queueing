from main import run_simulation
from sim_metadata import MatchingAlgo, ChargingAlgo, SimMetaData
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import os
from tabulate import tabulate


def main_sweep(sim_duration,
               n_cars,
               arrival_rate_pmin,
               n_chargers,
               n_posts,
               renege_time_min,
               infinite_chargers,
               d):
    consolidated_kpi = None
    dt = [('sim_duration', '<i8'), ('n_cars', '<i8'), ('arrival_rate_pmin', '<f8'), ('n_chargers', '<i8'),
          ('n_posts', '<i8'), ('renege_time_min', '<i8'), ('infinite_chargers', '<?'), ('d', '<i8'),]
    list_run_instances = np.array(list(itertools.product(
        sim_duration, n_cars, arrival_rate_pmin, n_chargers, n_posts, renege_time_min, infinite_chargers, d)), dtype=dt
                                                        )

    print_list_run_instances = np.array(
        list(itertools.product(sim_duration, n_cars, arrival_rate_pmin, n_chargers, n_posts, infinite_chargers, d))
                                        )
    print_list_run_instances[print_list_run_instances[:, 5] is True, 3] = 10000
    print_list_run_instances = np.delete(print_list_run_instances, 5, axis=1)
    header = ['Time (min)', 'Cars', 'Arrival (per min)', 'Chargers', 'Posts', 'd']
    print(tabulate(print_list_run_instances, header))

    today = datetime.now()
    curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
    top_level_dir = os.path.join("simulation_results/sweep_folder", curr_date_and_time)

    for j in range(0, len(list_run_instances["sim_duration"])):
        kpi = run_simulation(sim_duration=list_run_instances["sim_duration"][j],
                             n_cars=list_run_instances["n_cars"][j],
                             arrival_rate_pmin=list_run_instances["arrival_rate_pmin"][j],
                             n_chargers=list_run_instances["n_chargers"][j],
                             n_posts=list_run_instances["n_posts"][j],
                             renege_time_min=list_run_instances["renege_time_min"][j],
                             matching_algo=MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value,
                             charging_algo=ChargingAlgo.CHARGE_AFTER_TRIP_END.value,
                             d=list_run_instances["d"][j],
                             infinite_chargers=list_run_instances["infinite_chargers"][j],
                             results_folder=top_level_dir)
        # Append kpi to consolidate_kpi dataframe
        if consolidated_kpi is None:
            consolidated_kpi = kpi.copy()
        else:
            consolidated_kpi = pd.concat([consolidated_kpi, kpi], axis=0)

    kpi_data_file = os.path.join(top_level_dir, "consolidated_kpi.csv")
    consolidated_kpi.to_csv(kpi_data_file)


if __name__ == "__main__":
    list_arrival_rate_per_min = np.array([2, 5, 10, 20, 40, 60, 80, 100])
    r = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
    average_trip_dist_mi = 0.5214 * SimMetaData.max_lat # Assuming max lat is equal to max lon
    average_trip_time_min = average_trip_dist_mi * 60 / SimMetaData.avg_vel_mph
    target_service_level = 0.9
    gamma = 0.6
    list_n_cars = (
            (1 + r) * average_trip_time_min * target_service_level * list_arrival_rate_per_min
            + (average_trip_time_min * list_arrival_rate_per_min) ** gamma
                  )
    list_m_cars = 2 * r * average_trip_time_min * target_service_level * list_arrival_rate_per_min

    main_sweep(sim_duration=[5000],
               n_cars=[100],
               arrival_rate_pmin=[3],
               n_chargers=[13],
               n_posts=[8],
               renege_time_min=[1],
               infinite_chargers=[False],
               d=[1, 2, 3, 4])

