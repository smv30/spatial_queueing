import argparse
import os
from sim_metadata import SimMetaData, MatchingAlgo,ChargingAlgo
from main import run_simulation


def asymptotic_sim(
        arrival_rate_per_min,
        gamma,
        target_service_level,
        beta,
        top_level_dir,
        extra_cars,
        random_data=False,
        trip_data_csv_path=None
                    ):
    c = 4
    r = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
    average_trip_dist_mi = 0.5214 * SimMetaData.max_lat  # Assuming max lat is equal to max lon
    average_trip_time_min = average_trip_dist_mi * 60 / SimMetaData.avg_vel_mph
    if extra_cars is None:
        average_trip_dist_mi = 0.5214 * SimMetaData.max_lat  # Assuming max lat is equal to max lon
        average_trip_time_min = average_trip_dist_mi * 60 / SimMetaData.avg_vel_mph
        extra_cars = int((average_trip_time_min * arrival_rate_per_min) ** gamma)
    n_cars = int(
            (1 + r) * average_trip_time_min * target_service_level * arrival_rate_per_min
            + extra_cars)
    m_chargers = int(
            r * average_trip_time_min * target_service_level * arrival_rate_per_min
            + c * (average_trip_time_min * arrival_rate_per_min) ** beta
                      )
    top_level_dir = os.path.join(top_level_dir, f"lambda_{arrival_rate_per_min}_gamma_{gamma}_beta_{beta}_c_{c}")

    print(f"Running asymptotic simulation instance with arrival rate = {arrival_rate_per_min} per min")

    matching_algo = MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value
    d = 1
    SimMetaData.pack_size_kwh

    if matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value:
        keyword_folder = f"{arrival_rate_per_min}_lambda_power_of_{d}"
    else:
        keyword_folder = f"{arrival_rate_per_min}_lambda_closest_available_dispatch"

    sim_duration = 1000
    if random_data is False and trip_data_csv_path is None:
        trip_data_csv_path = (f"data/random_data_with_arrival_rate_" +
                              f"{arrival_rate_per_min}_per_min_and_sim_duration_{sim_duration}_mins.csv")

    kpi = run_simulation(sim_duration=sim_duration,
                         n_cars=n_cars,
                         arrival_rate_pmin=arrival_rate_per_min,
                         n_chargers=m_chargers,
                         n_posts=1,
                         renege_time_min=1,
                         matching_algo=matching_algo,
                         charging_algo=ChargingAlgo.CHARGE_AFTER_TRIP_END.value,
                         d=d,
                         infinite_chargers=False,
                         keyword_folder=keyword_folder,
                         results_folder=top_level_dir,
                         trip_data_csv_path=trip_data_csv_path)
    kpi["gamma"] = gamma
    kpi["beta"] = beta
    kpi["target_service_level"] = target_service_level
    kpi["constant"] = f"{c} Times"
    # Append kpi to consolidate_kpi dataframe
    kpi_folder = os.path.join(top_level_dir, "kpi_folder")
    if not os.path.isdir(kpi_folder):
        os.makedirs(kpi_folder)
    if matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value:
        kpi_data_file = os.path.join(kpi_folder, f"kpi_power_of_{d}_lambda_{arrival_rate_per_min}_gamma_{gamma}_beta_{beta}_target_service_{target_service_level}.csv")
    else:
        kpi_data_file = os.path.join(kpi_folder, f"kpi_closest_available_dispatch_lambda_{arrival_rate_per_min}_gamma_{gamma}_beta_{beta}_target_service_{target_service_level}.csv")
    kpi.to_csv(kpi_data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lambda_per_min', type=int, default=1)
    parser.add_argument('-g', '--gamma', type=float, default=0.6)
    parser.add_argument('-b', '--beta', type=float, default=1)
    parser.add_argument('-s', '--service_level', type=float, default=1)
    parser.add_argument('-e', '--extra_cars', type=int, default=None)
    args = parser.parse_args()
    input_arrival_rate_per_min = args.lambda_per_min
    input_gamma = args.gamma
    input_beta = args.beta
    input_service_level = args.service_level
    input_extra_cars = args.extra_cars
    input_top_level_dir = "simulation_results/"
    asymptotic_sim(arrival_rate_per_min=input_arrival_rate_per_min,
                   gamma=input_gamma,
                   beta=input_beta,
                   target_service_level=input_service_level,
                   top_level_dir=input_top_level_dir,
                   extra_cars=input_extra_cars)

