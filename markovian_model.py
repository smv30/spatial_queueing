import numpy as np
import os
import pandas as pd
import time
from sim_metadata import SimMetaData, MarkovianModelParams
from data_logging import DataLogging
from main import save_and_plot_results
from datetime import datetime


def get_arrival_rates(array_b, array_q):
    e_rate = SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh / 60
    array_queue_full_check = (array_q < 100).astype(int)
    array_ea = e_rate * 100 * (1 - array_b) * array_queue_full_check / MarkovianModelParams.charge_in_one_transition
    trip_time = SimMetaData.AVG_TRIP_DIST_PER_MILE_SQ * SimMetaData.max_lat / SimMetaData.avg_vel_mph * 60
    pickup_time = MarkovianModelParams.pickup_time_const / max((len(array_b) - sum(array_b)), 1) ** 0.5
    array_es = 1 / (trip_time + pickup_time) * array_b
    return array_ea, array_es


def generate_transition_and_time(array_ea, array_es, arrival_rate):
    total_arrival_rate = sum(array_ea) + sum(array_es) + arrival_rate
    array_prob = np.concatenate(([arrival_rate], array_ea, array_es)) / total_arrival_rate
    transition = SimMetaData.random_seed_gen.choice(len(array_ea) * 2 + 1, p=array_prob)
    time = SimMetaData.random_seed_gen.exponential(1 / total_arrival_rate)
    return transition, time


def power_of_d(array_q, array_b):
    available_soc_mask = (array_q >= SimMetaData.min_allowed_soc * 100).astype(int)
    n_choices = sum((1 - array_b) * available_soc_mask)
    if n_choices == 0:
        return None
    array_prob_tmp = np.ones(len(array_q)) / n_choices
    array_prob = array_prob_tmp * (1 - array_b) * available_soc_mask
    n_of_non_zero_prob = sum(array_prob != 0)
    array_queues_idx = SimMetaData.random_seed_gen.choice(a=len(array_q),
                                                          size=min(MarkovianModelParams.d, n_of_non_zero_prob),
                                                          replace=False,
                                                          p=array_prob
                                                          )
    max_soc_queue_idx_tmp = array_q[array_queues_idx].argmax()
    max_soc_queue_idx = array_queues_idx[max_soc_queue_idx_tmp]
    return max_soc_queue_idx


def markovian_sim(n_cars, sim_duration, arrival_rate, n_chargers, n_posts, results_folder = "simulation_results"):
    start_time = time.time()
    curr_time = 0
    array_q = np.ones(n_cars) * 50
    array_b = np.zeros(n_cars)
    time_to_go_for_data_logging = 0
    avg_trip_time = SimMetaData.AVG_TRIP_DIST_PER_MILE_SQ * SimMetaData.max_lat / SimMetaData.avg_vel_mph * 60
    data_logging = DataLogging()
    total_n_trips = 0
    total_met_trip = 0
    total_trip_time_min = 0
    while curr_time < sim_duration:
        array_ea, array_es = get_arrival_rates(array_b, array_q)
        transition_idx, transition_time = generate_transition_and_time(array_ea, array_es, arrival_rate)
        if transition_idx == 0:
            total_n_trips += 1
            queue_idx_to_add_to = power_of_d(array_q, array_b)
            if queue_idx_to_add_to is not None:
                total_met_trip += 1
                array_b[queue_idx_to_add_to] = 1
                if not SimMetaData.quiet_sim:
                    print(f"Customer Arrival at {curr_time}, matched to car {queue_idx_to_add_to}"
                          f" with SOC {array_q[queue_idx_to_add_to]}")
            else:
                if not SimMetaData.quiet_sim:
                    print(f"Customer Arrival at {curr_time}, abandoned the system due to unavailability."
                          f"Currently there are {n_cars - sum(array_b)} idle cars with SOC"
                          f"{array_q[(array_b == 0)]}")
        elif transition_idx <= len(array_q):
            idx = transition_idx - 1
            array_q[idx] = min(array_q[idx] + MarkovianModelParams.charge_in_one_transition, 100)
            if not SimMetaData.quiet_sim:
                print(f"Car {transition_idx - 1} is charged by {MarkovianModelParams.charge_in_one_transition}"
                      f" percent at {curr_time}")
        else:
            idx = transition_idx - 1 - len(array_q)
            array_b[idx] = 0
            service_time_min = 1 / array_es[idx]
            delta_soc = int(
                service_time_min / 60
                * SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph
                / SimMetaData.pack_size_kwh
                * 100
            )
            array_q[idx] -= delta_soc
            total_trip_time_min += service_time_min
            if not SimMetaData.quiet_sim:
                print(f"Car {transition_idx - 1 - len(array_q)} finished serving trip at {curr_time} and "
                      f"consumed {delta_soc} SOC")
        curr_time += transition_time

        # Data Logging
        if SimMetaData.save_results:
            if time_to_go_for_data_logging <= 0:
                n_cars_driving_to_charger = 0
                avg_service_time_min = total_trip_time_min / total_met_trip
                avg_pickup_time_min = avg_service_time_min - avg_trip_time
                n_cars_driving_without_passenger = (
                        sum(array_b) * avg_pickup_time_min / (avg_pickup_time_min + avg_trip_time)
                )
                n_cars_driving_with_passenger = sum(array_b) - n_cars_driving_without_passenger
                avg_soc = np.mean(array_q / 100)
                stdev_soc = np.std(array_q / 100)
                n_cars_idle = sum((array_q == 100).astype(int) * (1 - array_b))
                n_cars_charging = sum((array_q < 100).astype(int) * (1 - array_b))
                data_logging.update_data(curr_list_soc=array_q / 100,
                                         n_cars_idle=n_cars_idle,
                                         n_cars_charging=n_cars_charging,
                                         n_cars_driving_to_charger=n_cars_driving_to_charger,
                                         n_cars_driving_without_passenger=n_cars_driving_without_passenger,
                                         n_cars_driving_with_passenger=n_cars_driving_with_passenger,
                                         time_of_logging=curr_time,
                                         avg_soc=avg_soc,
                                         stdev_soc=stdev_soc
                                         )
                time_to_go_for_data_logging = SimMetaData.freq_of_data_logging_min - transition_time
            else:
                time_to_go_for_data_logging = time_to_go_for_data_logging - transition_time
    # KPI
    service_level_percentage = total_met_trip / total_n_trips * 100
    avg_service_time_min = total_trip_time_min / total_met_trip
    avg_trip_dist_mi = avg_trip_time * SimMetaData.avg_vel_mph / 60
    avg_pickup_time_min = avg_service_time_min - avg_trip_time
    avg_drive_to_charger_time_min = 0
    avg_n_of_charging_trips = 0
    d = MarkovianModelParams.d
    avg_soc = np.mean(data_logging.avg_soc)

    kpi = pd.DataFrame({
        "fleet_size": n_cars,
        "pack_size_kwh": SimMetaData.pack_size_kwh,
        "consumption_kwhpmi": SimMetaData.consumption_kwhpmi,
        "charge_rate_kw": SimMetaData.charge_rate_kw,
        "avg_vel_mph": SimMetaData.avg_vel_mph,
        "n_chargers": n_chargers,
        "n_posts": n_posts,
        "total_sim_duration_min": sim_duration,
        "arrival_rate_pmin": arrival_rate,
        "total_n_trips": total_n_trips,
        "avg_trip_time_min": avg_trip_time,
        "avg_trip_dist_mi": avg_trip_dist_mi,
        "avg_pickup_time_min": avg_pickup_time_min,
        "avg_drive_time_to_charger": avg_drive_to_charger_time_min,
        "number_of_trips_to_charger_per_car_per_hr": avg_n_of_charging_trips,
        "avg_soc_over_time_over_cars": avg_soc,
        "service_level_percentage": service_level_percentage,
        "matching_algorithm": f"Power of {d}"
    }, index=[0])

    if SimMetaData.save_results:

        today = datetime.now()
        curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
        top_level_dir = os.path.join(results_folder, curr_date_and_time + "_markovian_model")

        soc_time_series_data = np.array(data_logging.list_soc)
        df_demand_curve_data = data_logging.demand_curve_to_dict()

        save_and_plot_results(df_demand_curve_data=df_demand_curve_data,
                              soc_time_series_data=soc_time_series_data,
                              kpi=kpi,
                              top_level_dir=top_level_dir,
                              n_cars=n_cars,
                              arrival_rate_pmin=arrival_rate
                              )
    print(f"Simulation Time: {int((time.time() - start_time) / 60)} mins")
    return kpi


if __name__ == "__main__":
    markovian_sim(839, 3000, 63.93, 10, 1)
