import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def boxplot_arrivals(year,
                     month,
                     max_date=30,
                     dataset="nyc",
                     testing=False):
    if dataset == "nyc":
        dataset_path = f"yellow_tripdata_{year}-{month:02d}.parquet"
        entire_df = pd.read_parquet(dataset_path, engine='fastparquet')
        pickup_datetime = pd.to_datetime(entire_df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S")
    elif dataset == "chicago":
        dataset_path = f"Chicago_year_{year}_month_{month:02d}.csv"
        entire_df = pd.read_csv(dataset_path)
        pickup_datetime = pd.to_datetime(entire_df["Trip Start Timestamp"], format="%m/%d/%Y %I:%M:%S %p")
    else:
        raise ValueError("No such dataset exists")
    list_date = []
    list_hour = []
    list_n_trips = []
    for date in range(1, max_date + 1):
        for hour in range(24):
            min_datetime = datetime(year, month, date, hour, 0, 0)
            max_datetime = min_datetime + timedelta(hours=1)
            n_trips = len(pickup_datetime[(pickup_datetime >= min_datetime) & (pickup_datetime < max_datetime)])
            list_date.append(date)
            list_hour.append(hour)
            list_n_trips.append(n_trips)
    peak_demand = max(list_n_trips)

    arrival_statistics = pd.DataFrame({
        "date": list_date,
        "hour": list_hour,
        "n_trips": 100 * np.array(list_n_trips) / peak_demand
    })
    fig, ax = plt.subplots()
    ax = arrival_statistics.boxplot(column="n_trips", by="hour", ax=ax)
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Percentage of the Peak Demand")
    ax.set_title(f"Boxplot of Trip Demand: Year={year}, month={month:02d}")
    fig.suptitle("")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    if testing is True:
        fig.savefig(f"MS_R_and_R_Plots/{dataset}_boxplot_trip_demand_year_{year}_month_{month}.png")
    else:
        fig.savefig(f"MS_R_and_R_Plots/{dataset}_boxplot_trip_demand_year_{year}_month_{month}.eps",
                    bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    boxplot_arrivals(year=2022,
                     month=6,
                     max_date=30,
                     dataset="chicago",
                     testing=True
                     )

