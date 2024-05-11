import pandas as pd
import matplotlib.pyplot as plt
from sim_metadata import SimMetaData, Dataset, DistFunc, DatasetParams
from datetime import datetime, timedelta
from real_life_data_input import DataInput
import numpy as np


# def fleet_size(
#         demand_curve,
#         service_level,
#         pickup_factor=0.1,
# ):
#     soc = 0.5
#     n_cars_estimate = 400
#     count = 0
#     adjust_fleet_size_by = 0.05
#     list_soc = []
#     list_n_cars_driving = []
#     for count in range(len(demand_curve["time"]) - 1):
#         count += 1
#         time_elapsed_min = (demand_curve["time"].iloc[count] - demand_curve["time"].iloc[count - 1])
#         n_incoming_trips = demand_curve["n_trips_in_progress"].iloc[count] * (1 + pickup_factor)
#         n_trips_in_progress = min(demand_curve["n_trips_in_progress"].iloc[count] * (1 + pickup_factor),
#                                   n_cars_estimate)
#         n_cars_charging = max(n_cars_estimate - n_trips_in_progress, 0)
#         soc = min((
#                 soc
#                 + (n_cars_charging * SimMetaData.charge_rate_kw
#                    - n_trips_in_progress * SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph
#                    )
#                 * time_elapsed_min / SimMetaData.pack_size_kwh / n_cars_estimate / 60
#         ), 1)
#         # Drop trips if SoC does not allow
#         # Finally calculate the service level
#         # If the service level is not met, repeat
#         list_soc.append(soc)
#         list_n_cars_driving.append(n_trips_in_progress)
#         if soc <= SimMetaData.min_allowed_soc:
#             n_cars_estimate = (1 + adjust_fleet_size_by) * n_cars_estimate
#             count = 0
#             list_soc = []
#             list_n_cars_driving = []
#     return n_cars_estimate, list_soc, list_n_cars_driving


def fleet_size_ode(df_arrival_sequence):
    n_cars = 700
    n_chargers = 300
    list_soc = []
    list_n_cars_driving_with_passenger = []
    list_n_cars_driving_without_passenger = []
    list_n_cars_driving_to_charger = []
    list_n_cars_charging = []
    list_n_cars_idle = []
    start_year = start_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day
    start_hour = start_datetime.hour
    start_minute = start_datetime.minute
    first_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].min()
    last_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].max()
    total_sim_time_datetime = last_trip_pickup_datetime - first_trip_pickup_datetime
    sim_duration_min = int(total_sim_time_datetime.total_seconds() / 60.0) + 1
    n_active_trips = 0
    soc = 1
    n_cars_charging = 0
    pickup_time_const = 20
    drive_to_charger_time_const = 20
    charge_time_min = 0.2 * SimMetaData.pack_size_kwh / SimMetaData.charge_rate_kw * 60
    total_workload = 0
    served_workload = 0
    for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
        current_min_total = (current_index * SimMetaData.demand_curve_res_min
                             + start_day * 1440 + start_hour * 60 + start_minute)
        current_day = int(current_min_total / 1440)
        current_hour = int(current_min_total % 1440 / 60)
        current_min = current_min_total - current_day * 1440 - current_hour * 60
        pickup_time_min = pickup_time_const / (max(n_cars - n_active_trips, 1)) ** 0.5
        drive_to_charger_time_min = drive_to_charger_time_const / (max(n_chargers - n_cars_charging, 1)) ** 0.5
        df_pickup_trips = df_arrival_sequence[
            (
                    df_arrival_sequence["pickup_datetime"] <=
                    datetime(start_year, start_month, current_day, current_hour, current_min, 0)
            )
            &
            (
                    df_arrival_sequence["pickup_datetime"] + timedelta(minutes=pickup_time_min) >
                    datetime(start_year, start_month, current_day, current_hour, current_min, 0)
            )
            ]
        df_en_route_trips = df_arrival_sequence[
            (
                    df_arrival_sequence["pickup_datetime"] + timedelta(minutes=pickup_time_min) <=
                    datetime(start_year, start_month, current_day, current_hour, current_min, 0)
            )
            &
            (
                    df_arrival_sequence["dropoff_datetime"] + timedelta(minutes=pickup_time_min) >
                    datetime(start_year, start_month, current_day, current_hour, current_min, 0)
            )
            ]
        n_cars_driving_without_passenger = len(df_pickup_trips)
        n_cars_driving_with_passenger = len(df_en_route_trips)
        incoming_workload = len(df_en_route_trips)
        n_active_trips = min(n_cars_driving_without_passenger + n_cars_driving_with_passenger, n_cars)
        n_cars_driving_without_passenger = (
                n_cars_driving_without_passenger * n_active_trips
                / (n_cars_driving_without_passenger + n_cars_driving_with_passenger)
        )
        n_cars_driving_with_passenger = (
                n_cars_driving_with_passenger * n_active_trips
                / (n_cars_driving_without_passenger + n_cars_driving_with_passenger)
        )
        n_cars_driving_to_charger = (
                min((n_cars - n_active_trips), n_chargers) * drive_to_charger_time_min
                / (charge_time_min + drive_to_charger_time_min)
        )
        n_cars_charging = (
                min((n_cars - n_active_trips), n_chargers) * charge_time_min
                / (charge_time_min + drive_to_charger_time_min)
        )
        n_cars_idle = (
                n_cars - n_cars_driving_with_passenger - n_cars_driving_without_passenger
                - n_cars_driving_to_charger - n_cars_charging
        )
        # SoC increases due to charging
        soc_plus = (
                n_cars_charging * SimMetaData.charge_rate_kw * SimMetaData.demand_curve_res_min
                / 60 / n_cars / SimMetaData.pack_size_kwh
        )
        # SoC decreases due to driving
        soc_minus = (
            (n_active_trips + n_cars_driving_to_charger) * SimMetaData.consumption_kwhpmi
            * SimMetaData.avg_vel_mph * SimMetaData.demand_curve_res_min / 60 / n_cars / SimMetaData.pack_size_kwh
        )
        soc = min(soc + soc_plus - soc_minus, 1)
        # Update data
        list_soc.append(soc)
        list_n_cars_charging.append(n_cars_charging)
        list_n_cars_driving_to_charger.append(n_cars_driving_to_charger)
        list_n_cars_driving_without_passenger.append(n_cars_driving_without_passenger)
        list_n_cars_driving_with_passenger.append(n_cars_driving_with_passenger)
        list_n_cars_idle.append(n_cars_idle)
        total_workload += incoming_workload
        served_workload += n_cars_driving_with_passenger

    # Stackplot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(1, sim_duration_min, SimMetaData.demand_curve_res_min)
    ax1.stackplot(x,
                  list_n_cars_driving_with_passenger,
                  list_n_cars_driving_without_passenger,
                  list_n_cars_idle,
                  list_n_cars_driving_to_charger,
                  list_n_cars_charging,
                  colors=['b', 'tab:orange', 'g', 'tab:purple', 'y'])
    ax2.plot(x, list_soc, 'k', linewidth=3)
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Number of Cars")
    ax1.set_ylim([0, n_cars])
    ax2.set_ylabel("SOC")
    ax2.set_ylim([0, 1])
    plt.show()
    return served_workload/total_workload * 100
    # run every minute:
    # calculate pickup time -> T_P(t)
    # calculate drive to charger time -> T_DC(t)
    # calculate average trip time -> T_R(t)
    # update the number of active trips (accounting for pickup times) -> q(t)
    # assume every car charges from 20-80% when it reaches the charger -> T_C(t)
    # (n - q(t))*T_DC(t)/(T_C(t)+T_DC(t)) EVs will be driving to the charger
    # remaining will be charging
    # update the SoC
    # If SoC drops below 5%, increase the fleet size and re-run.


# def clipped_demand_curve(demand_curve, service_level):
#     clipped_demand_curve = None
#     return clipped_demand_curve
#
#
# def demand_curve_gen(df_arrival_sequence, start_datetime):
#     current_min_list = []
#     num_trips_in_progress_list = []
#     start_year = start_datetime.year
#     start_month = start_datetime.month
#     start_day = start_datetime.day
#     start_hour = start_datetime.hour
#     start_minute = start_datetime.minute
#     demand_curve = pd.DataFrame({})
#     first_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].min()
#     last_trip_pickup_datetime = df_arrival_sequence["pickup_datetime"].max()
#     total_sim_time_datetime = last_trip_pickup_datetime - first_trip_pickup_datetime
#     sim_duration_min = int(total_sim_time_datetime.total_seconds() / 60.0) + 1
#     for current_index in range(1, int(sim_duration_min / SimMetaData.demand_curve_res_min)):
#         current_min_total = (current_index * SimMetaData.demand_curve_res_min
#                              + start_day * 1440 + start_hour * 60 + start_minute)
#         current_day = int(current_min_total / 1440)
#         current_hour = int(current_min_total % 1440 / 60)
#         current_min = current_min_total - current_day * 1440 - current_hour * 60
#         trips_in_progress = df_arrival_sequence[
#             (df_arrival_sequence["pickup_datetime"] <= datetime(start_year, start_month, current_day,
#                                                                 current_hour, current_min, 0)) &
#             (df_arrival_sequence["dropoff_datetime"] > datetime(start_year, start_month, current_day,
#                                                                 current_hour, current_min, 0))
#             ]
#         num_trips_in_progress = len(trips_in_progress)
#         current_min_list.append(current_index * SimMetaData.demand_curve_res_min)
#         num_trips_in_progress_list.append(num_trips_in_progress)
#     demand_curve["n_trips_in_progress"] = num_trips_in_progress_list
#     demand_curve["time"] = current_min_list
#     return demand_curve


if __name__ == "__main__":
    data_input = DataInput(percentile_lat_lon=99)
    dataset_source = Dataset.CHICAGO.value
    dataset_path = '/Users/sushilvarma/PycharmProjects/SpatialQueueing/Chicago_year_2022_month_06.csv'
    start_datetime = datetime(2022, 6, 1, 0, 0, 0)
    end_datetime = datetime(2022, 6, 2, 0, 0, 0)
    dist_func = DistFunc.MANHATTAN.value
    df_arrival_sequence, dist_correction_factor = data_input.real_life_dataset(
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        percent_of_trips=DatasetParams.percent_of_trips_filtered,
        dist_func=dist_func
    )
    # demand_curve = demand_curve_gen(df_arrival_sequence=df_arrival_sequence, start_datetime=start_datetime
    #                                 )
    # n_cars, list_soc, list_n_cars_driving = fleet_size(demand_curve=demand_curve, service_level=0.8)
    workload_percentage = fleet_size_ode(df_arrival_sequence)
    print(workload_percentage)
    # print(n_cars)
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(demand_curve["n_trips_in_progress"])
    # ax2.plot(list_soc)
    # ax1.plot(list_n_cars_driving)
    # plt.show()
