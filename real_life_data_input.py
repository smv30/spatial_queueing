import os
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os.path
from sim_metadata import SimMetaData, DatasetParams
from utils import sample_unif_points_on_sphere
import warnings
from utils import calc_dist_between_two_points
import datetime


class DataInput:
    def __init__(self,
                 percentile_lat_lon=None,
                 ):
        self.percentile_lat_lon = percentile_lat_lon

    def plotuniftrip(self):
        sim_duration_min = 100
        arrival_rate_pmin = 5
        curr_time_min = 0
        df_trips_cartesian = pd.DataFrame({
            "cartesian_x": [],
            "cartesian_y": [],
            "cartesian_z": [],
        })
        while curr_time_min <= sim_duration_min:
            # get spherical lat & lon in degrees
            start_lat, start_lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                                lon_max=DatasetParams.longitude_range_max,
                                                                lat_min=DatasetParams.latitude_range_min,
                                                                lat_max=DatasetParams.latitude_range_max)
            # convert spherical lat & lon from degrees back to radius
            unif_lon_start = (start_lon + 180) / 360
            unif_lat_start = (np.cos(np.pi / 180 * (start_lat + 90)) + 1) / 2

            # convert spherical to cartesian coordinates: x = ρsinφcosθ; y = ρsinφsinθ; z = ρcosφ
            cartesian_x = 1 * np.sin(unif_lat_start) * np.cos(unif_lon_start)
            cartesian_y = 1 * np.sin(unif_lat_start) * np.sin(unif_lon_start)
            cartesian_z = 1 * np.cos(unif_lat_start)

            df_this_trip_cartesian = pd.DataFrame({
                "cartesian_x": [cartesian_x],  # pickup_datetime
                "cartesian_y": [cartesian_y],
                "cartesian_z": [cartesian_z]
            })
            df_trips_cartesian = pd.concat([df_trips_cartesian, df_this_trip_cartesian], ignore_index=True)
            inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / arrival_rate_pmin)
            curr_time_min = curr_time_min + inter_arrival_time_min

        ax = plt.axes(projection='3d')
        z_data = df_trips_cartesian["cartesian_z"]
        x_data = df_trips_cartesian["cartesian_x"]
        y_data = df_trips_cartesian["cartesian_y"]
        ax.scatter3D(x_data, y_data, z_data)
        plt.show()
        plt.close()

    def randomly_generated_dataframe(self, sim_duration_min, arrival_rate_pmin, data_dir, start_datetime):
        curr_time_datetime = start_datetime
        df_trips = pd.DataFrame({
            # "arrival_time": [],
            "pickup_datetime": pd.to_datetime([], format="%Y-%m-%d %H:%M:%S"),  # arrival_time
            "dropoff_datetime": pd.to_datetime([], format="%Y-%m-%d %H:%M:%S"),
            "pickup_latitude": [],
            "pickup_longitude": [],
            "dropoff_latitude": [],
            "dropoff_longitude": []
        })
        time_passed = int((curr_time_datetime - start_datetime).total_seconds() / 60)
        while time_passed <= sim_duration_min:
            start_lat, start_lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                                lon_max=DatasetParams.longitude_range_max,
                                                                lat_min=DatasetParams.latitude_range_min,
                                                                lat_max=DatasetParams.latitude_range_max)
            end_lat, end_lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                            lon_max=DatasetParams.longitude_range_max,
                                                            lat_min=DatasetParams.latitude_range_min,
                                                            lat_max=DatasetParams.latitude_range_max)
            trip_time_min = calc_dist_between_two_points(start_lat=start_lat,
                                                         start_lon=start_lon,
                                                         end_lat=end_lat,
                                                         end_lon=end_lon) / SimMetaData.avg_vel_mph * 60
            trip_time_datetime = datetime.timedelta(minutes=trip_time_min)
            dropoff_datetime = curr_time_datetime + trip_time_datetime
            df_this_trip = pd.DataFrame({
                # "arrival_time": [curr_time_min],
                # datetime(year, month, day, hour, minute, second, microsecond)
                "pickup_datetime": [curr_time_datetime],
                "dropoff_datetime": [dropoff_datetime],
                "pickup_latitude": [start_lat],
                "pickup_longitude": [start_lon],
                "dropoff_latitude": [end_lat],
                "dropoff_longitude": [end_lon]
            })
            df_trips = pd.concat([df_trips, df_this_trip], ignore_index=True)
            inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / arrival_rate_pmin)
            inter_arrival_time_datetime = datetime.timedelta(minutes=inter_arrival_time_min)
            curr_time_datetime = curr_time_datetime + inter_arrival_time_datetime
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        trips_csv_path = os.path.join(
            data_dir,
            f"random_data_with_arrival_rate_{arrival_rate_pmin}_per_min_and_sim_duration_{sim_duration_min}_mins.csv"
        )
        df_trips.to_csv(trips_csv_path)
        return df_trips, 1

    def ny_taxi_dataset(self, dataset_path, start_datetime, end_datetime, percent_of_trips):
        # Step 1: read the dataset and get useful columns
        if not os.path.isfile("sampledata.csv") or SimMetaData.test is False:
            entire_df = pd.read_parquet(
                dataset_path,
                engine='fastparquet')
            df_filtered = entire_df[
                ["pickup_datetime", "dropoff_datetime", "trip_distance", "pickup_longitude", "pickup_latitude",
                 "dropoff_longitude", "dropoff_latitude"]
            ]

            # Step 2: filter out the trips with invalid time and location, and then sample the dataset
            # fetch all rows that are valid between a specific time range
            df_filtered.loc[:, "pickup_datetime"] = pd.to_datetime(df_filtered['pickup_datetime'],
                                                                   format='%Y-%m-%d %H:%M:%S')
            df_filtered.loc[:, "dropoff_datetime"] = pd.to_datetime(df_filtered['dropoff_datetime'],
                                                                    format='%Y-%m-%d %H:%M:%S')
            df_filtered = df_filtered[
                (df_filtered["pickup_datetime"] >= start_datetime) &
                (df_filtered["pickup_datetime"] < end_datetime)]

            # fetch all rows with valid latitude and longitude
            lower_percentile_lat_lon = (100 - self.percentile_lat_lon) / 2
            upper_percentile_lat_lon = (100 - self.percentile_lat_lon) / 2 + self.percentile_lat_lon
            DatasetParams.latitude_range_min, DatasetParams.latitude_range_max = np.percentile(
                df_filtered["pickup_latitude"],
                [lower_percentile_lat_lon,
                 upper_percentile_lat_lon])
            DatasetParams.longitude_range_min, DatasetParams.longitude_range_max = np.percentile(
                df_filtered["pickup_longitude"],
                [lower_percentile_lat_lon,
                 upper_percentile_lat_lon])
            while abs(DatasetParams.latitude_range_max - DatasetParams.latitude_range_min) > DatasetParams.delta_latitude:
                warnings.warn("The max and min latitudes are too far away from each other, reducing the percentile by 5%.")
                lower_percentile_lat_lon_latitude = np.copy(lower_percentile_lat_lon)
                upper_percentile_lat_lon_latitude = np.copy(upper_percentile_lat_lon)
                lower_percentile_lat_lon_latitude = lower_percentile_lat_lon_latitude + 2.5
                upper_percentile_lat_lon_latitude = upper_percentile_lat_lon_latitude - 2.5
                DatasetParams.latitude_range_min, DatasetParams.latitude_range_max = np.percentile(
                    df_filtered["pickup_latitude"],
                    [lower_percentile_lat_lon_latitude, upper_percentile_lat_lon_latitude])
            while abs(DatasetParams.longitude_range_max - DatasetParams.longitude_range_min) > DatasetParams.delta_longitude:
                warnings.warn("The max and min longitudes are too far away from each other, reducing the percentile by 5%.")
                lower_percentile_lat_lon_longitude = np.copy(lower_percentile_lat_lon)
                upper_percentile_lat_lon_longitude = np.copy(upper_percentile_lat_lon)
                lower_percentile_lat_lon_longitude = lower_percentile_lat_lon_longitude + 2.5
                upper_percentile_lat_lon_longitude = upper_percentile_lat_lon_longitude - 2.5
                DatasetParams.longitude_range_min, DatasetParams.longitude_range_max = np.percentile(
                    df_filtered["pickup_longitude"],
                    [lower_percentile_lat_lon_longitude, upper_percentile_lat_lon_longitude])
            df_output = df_filtered[
                (df_filtered["pickup_longitude"] <= DatasetParams.longitude_range_max) &
                (df_filtered["pickup_latitude"] >= DatasetParams.latitude_range_min) &
                (df_filtered["dropoff_longitude"] <= DatasetParams.longitude_range_max) &
                (df_filtered["dropoff_latitude"] >= DatasetParams.latitude_range_min) &
                (df_filtered["pickup_longitude"] >= DatasetParams.longitude_range_min) &
                (df_filtered["pickup_latitude"] <= DatasetParams.latitude_range_max) &
                (df_filtered["dropoff_longitude"] >= DatasetParams.longitude_range_min) &
                (df_filtered["dropoff_latitude"] <= DatasetParams.latitude_range_max)
                ]

            # sample a certain percent of data
            sample_size = int(len(df_output) * percent_of_trips)
            df_output = df_output.sample(sample_size)
            df_output.sort_values(by='pickup_datetime', inplace=True)
            df_output.reset_index(drop=True)

            # Step 3: save the dataframe into a CSV file or read the dataframe from a CSV file
            df_output.to_csv(
                "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/sampledata.csv")
        else:
            df_output = pd.read_csv(
                "/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/sampledata.csv")

        # Plot all trips' start longitude in a histogram
        if not SimMetaData.quiet_sim:
            fig1, ax1 = plt.subplots()
            ax1.hist(df_output["pickup_longitude"], 5, ec='blue', label='pickup_longitude')
            ax1.legend(loc='upper left')
            plt.show()
            plt.close()

        # Choose the range that 99.9% data of start longitude falls in
        # lower_lon, upper_lon = np.percentile(df_output["pickup_longitude"], [0.05, 99.95])
        # df_output = df_output[(df_output["pickup_longitude"] > lower_lon) & (df_output["pickup_longitude"] < upper_lon)]

        # Plot the start longitude of trips in a histogram after choosing the 99.9% range
        if not SimMetaData.quiet_sim:
            fig2, ax2 = plt.subplots()
            ax2.hist(df_output["pickup_longitude"], 5, ec='blue', label='pickup_longitude')
            ax2.legend(loc='upper left')
            plt.show()
            plt.close()

        # Plot all trips' start latitude in a histogram
        if not SimMetaData.quiet_sim:
            fig3, ax3 = plt.subplots()
            ax3.hist(df_output["pickup_latitude"], 5, ec='green', label='pickup_latitude')
            ax3.legend(loc='upper left')
            plt.show()
            plt.close()

        # Choose the range that 99.9% data falls in
        # lower_lat, upper_lat = np.percentile(df_output["pickup_latitude"], [0.05, 99.95])
        # df_output = df_output[(df_output["pickup_latitude"] > lower_lat) & (df_output["pickup_latitude"] < upper_lat)]

        # Plot all trips' start latitude in a histogram
        if not SimMetaData.quiet_sim:
            fig4, ax4 = plt.subplots()
            ax4.hist(df_output["pickup_latitude"], 5, ec='green', label='pickup_latitude')
            ax4.legend(loc='upper left')
            plt.show()
            plt.close()

        # Step 4: Calculate the Haversine distance
        df_output["haversine_distance"] = ""
        df_output["pickup"] = list(zip(df_output["pickup_latitude"], df_output["pickup_longitude"]))
        df_output["dropoff"] = list(zip(df_output["dropoff_latitude"], df_output["dropoff_longitude"]))
        df_output["haversine_distance"] = haversine_vector(list(df_output["pickup"]),
                                                           list(df_output["dropoff"]),
                                                           unit=Unit.MILES)  # in miles
        if not SimMetaData.quiet_sim:
            print("-------------Haversine Distance------------")
            print(df_output[["haversine_distance", "trip_distance"]])

        # Step 5: Linear Regression for Haversine distance
        #         -> Plot, R-Squared value, linear regression function, Mean Squared Error
        sampleDF_80_percent = df_output.sample(int(len(df_output) * 0.8))
        x = sampleDF_80_percent["haversine_distance"].values.reshape(-1, 1)  # "values" converts it into a numpy array
        y = sampleDF_80_percent["trip_distance"].values.reshape(-1,
                                                                1)  # -1 means calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression(fit_intercept=False)  # create object for the class
        linear_regressor.fit(x, y)  # perform linear regression
        r_squared = linear_regressor.score(x, y)  # calculate R-Squared of regression model
        if not SimMetaData.quiet_sim:
            print(f"R-Squared value: {r_squared}")

        y_pred = linear_regressor.predict(x)  # make predictions
        if not SimMetaData.quiet_sim:
            plt.scatter(x, y)
            plt.plot(x, y_pred, color='red')
            plt.xlabel("Haversine Distance")
            plt.ylabel("Trip Distance")
            plt.show()
            plt.close()

        # get the linear regression function with slope and intercept
        slope = linear_regressor.coef_[0][0]
        if not SimMetaData.quiet_sim:
            print('Slope:', slope)
        intercept = linear_regressor.intercept_
        if not SimMetaData.quiet_sim:
            print('Intercept:', intercept)

        extract_idx = list(set(df_output.index) - set(sampleDF_80_percent.index))
        sampleDF_20_percent = df_output.loc[extract_idx]
        MSE = mean_squared_error(y, y_pred)  # calculate Mean Squared Error
        if not SimMetaData.quiet_sim:
            print(f"Mean Squared Error: {MSE}")

        return df_output, slope


# # Step 6: Calculate the Euclidean distance
# print("-------------Euclidean Distance------------")
# sampleDF["euclidean_distance"] = ""
# sampleDF["pickup"] = list(zip(sampleDF["pickup_latitude"], sampleDF["pickup_longitude"]))
# sampleDF["dropoff"] = list(zip(sampleDF["dropoff_latitude"], sampleDF["dropoff_longitude"]))
# sampleDF["euclidean_distance"] = np.sqrt(np.sum(np.square(sampleDF["pickup"] - sampleDF["dropoff"])))
# sampleDF["euclidean_distance"] = np.sqrt(np.sum(tuple(map(lambda i, j: np.square(i - j),
#                                          sampleDF["pickup"], sampleDF["dropoff"]))))
# sampleDF["euclidean_distance"] = np.sqrt((np.square(sampleDF["pickup_latitude"] - sampleDF["dropoff_latitude"]) +
#                                          np.square(sampleDF["pickup_longitude"] - sampleDF["dropoff_longitude"])))
# print(sampleDF[["euclidean_distance", "trip_distance"]])
# # Step 7: Calculate the L1 Distance
# # Step 8: Calculate the Lp Distance

# To Do:
# 1. Add a feature to NY dataset: matching every minute (now we are matching every trip) -> change matching algorithm
# 2. Essay: compare matching instantaneously vs. waiting for some time
# 3. Create a random dataset which has same columns as the NY one (another dataset)

if __name__ == "__main__":
    data_input = DataInput(percentile_lat_lon=99.9)
    DataInput.plotuniftrip(data_input)
