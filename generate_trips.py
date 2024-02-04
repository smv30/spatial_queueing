import os
import pandas as pd
from sim_metadata import SimMetaData


def generate_trips(sim_duration_min, arrival_rate_pmin, data_dir):
    curr_time_min = 0
    df_trips = pd.DataFrame({
        "arrival_time": [],
        "start_lat": [],
        "start_lon": [],
        "end_lat": [],
        "end_lon": []
    })
    while curr_time_min <= sim_duration_min:
        start_lat = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lat)
        start_lon = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lon)
        end_lat = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lat)
        end_lon = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lon)
        df_this_trip = pd.DataFrame({
            "arrival_time": [curr_time_min],
            "start_lat": [start_lat],
            "start_lon": [start_lon],
            "end_lat": [end_lat],
            "end_lon": [end_lon]
        })
        df_trips = pd.concat([df_trips, df_this_trip], ignore_index=True)
        inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / arrival_rate_pmin)
        curr_time_min = curr_time_min + inter_arrival_time_min
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    trips_csv_path = os.path.join(
        data_dir,
        f"random_data_with_arrival_rate_{arrival_rate_pmin}_per_min_and_sim_duration_{sim_duration_min}_mins.csv"
                                )
    df_trips.to_csv(trips_csv_path)


if __name__ == "__main__":
    for input_arrival_rate_pmin in [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]:
        generate_trips(arrival_rate_pmin=input_arrival_rate_pmin,
                       sim_duration_min=1000,
                       data_dir="data")