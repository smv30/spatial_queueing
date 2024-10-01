import os
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os.path
from sim_metadata import SimMetaData, DatasetParams, Dataset, DistFunc
from utils import sample_unif_points_on_sphere
import warnings
from utils import calc_dist_between_two_points
import geopandas as gpd
from datetime import datetime, timedelta


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
        time_passed = 0
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
            trip_time_datetime = timedelta(minutes=trip_time_min)
            dropoff_datetime = curr_time_datetime + trip_time_datetime
            trip_distance_mi = calc_dist_between_two_points(start_lat=start_lat, start_lon=start_lon,
                                                            end_lat=end_lat, end_lon=end_lon)
            df_this_trip = pd.DataFrame({
                # "arrival_time": [curr_time_min],
                "pickup_datetime": [curr_time_datetime],
                "dropoff_datetime": [dropoff_datetime],
                "trip_distance": [trip_distance_mi],
                "pickup_latitude": [start_lat],
                "pickup_longitude": [start_lon],
                "dropoff_latitude": [end_lat],
                "dropoff_longitude": [end_lon]
            })
            df_trips = pd.concat([df_trips, df_this_trip], ignore_index=True)
            inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / arrival_rate_pmin)
            inter_arrival_time_datetime = timedelta(minutes=inter_arrival_time_min)
            curr_time_datetime = curr_time_datetime + inter_arrival_time_datetime
            time_passed = (curr_time_datetime - start_datetime).total_seconds() / 60.0
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        trips_csv_path = os.path.join(
            data_dir,
            f"random_data_with_arrival_rate_{arrival_rate_pmin}_per_min_and_sim_duration_{sim_duration_min}_mins.csv"
        )
        df_trips.to_csv(trips_csv_path)
        dist_correction_factor = 1
        return df_trips, dist_correction_factor

    def real_life_dataset(self,
                          dataset_source,
                          dataset_path,
                          start_datetime,
                          end_datetime,
                          percent_of_trips,
                          dist_func,
                          centroid=False):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sample_data_path = os.path.join(dir_path, "sampledata.csv")
        # Step 1: read the dataset and get useful columns
        if os.path.isfile(sample_data_path) and SimMetaData.test is True:
            df_output = pd.read_csv(sample_data_path)
            DatasetParams.latitude_range_min = df_output["pickup_latitude"].min()
            DatasetParams.latitude_range_max = df_output["pickup_latitude"].max()
            DatasetParams.longitude_range_min = df_output["pickup_longitude"].min()
            DatasetParams.longitude_range_max = df_output["pickup_longitude"].max()
        else:
            if dataset_source == Dataset.NYTAXI.value:

                entire_df = pd.read_parquet(
                    dataset_path,
                    engine='fastparquet')

                df_filtered = entire_df[["tpep_pickup_datetime",
                                         "tpep_dropoff_datetime",
                                         "PULocationID",
                                         "DOLocationID",
                                         "trip_distance"]]

                df_filtered = df_filtered.rename(
                    columns={"tpep_pickup_datetime": "pickup_datetime",
                             "tpep_dropoff_datetime": "dropoff_datetime",
                             "Trip Miles": "trip_distance"
                             }
                )

                # fetch all rows that are valid between a specific time range
                df_filtered.loc[:, "pickup_datetime"] = pd.to_datetime(df_filtered["pickup_datetime"],
                                                                       format="%Y-%m-%d %H:%M:%S")
                df_filtered.loc[:, "dropoff_datetime"] = pd.to_datetime(df_filtered["dropoff_datetime"],
                                                                        format="%Y-%m-%d %H:%M:%S")
                df_filtered = df_filtered[
                    (df_filtered["pickup_datetime"] >= start_datetime) &
                    (df_filtered["pickup_datetime"] < end_datetime)]
                if df_filtered.empty:
                    raise ValueError("The filtered dataset is empty. Try adjusting state and end dates")

                # sample a certain percent of data
                sample_size = int(len(df_filtered) * percent_of_trips)
                df_filtered = df_filtered.sample(sample_size, random_state=SimMetaData.random_seed)

                # Add pickup and dropoff latitudes and longitudes from the shapefile and zones
                shapefile = gpd.read_file("taxi_zones/taxi_zones.shp")

                shapefile["PULocationID"] = shapefile["LocationID"]
                shapefile["DOLocationID"] = shapefile["LocationID"]

                if centroid is True:  # Handling the case of centroid separately to optimize the code for this case
                    shapefile["pickup_longitude"] = shapefile.centroid.to_crs(4326).x
                    shapefile["pickup_latitude"] = shapefile.centroid.to_crs(4326).y
                    shapefile["dropoff_longitude"] = shapefile.centroid.to_crs(4326).x
                    shapefile["dropoff_latitude"] = shapefile.centroid.to_crs(4326).y
                    shapefile["PULocationID"] = shapefile["LocationID"]
                    shapefile["DOLocationID"] = shapefile["LocationID"]
                    df_filtered = df_filtered.merge(
                        shapefile[["PULocationID", "pickup_longitude", "pickup_latitude"]],
                        on="PULocationID",
                        how="inner"
                    )
                    df_filtered = df_filtered.merge(
                        shapefile[["DOLocationID", "dropoff_longitude", "dropoff_latitude"]],
                        on="DOLocationID",
                        how="inner"
                    )
                else:  # Randomly selecting one point in the polygon
                    df_filtered = shapefile[["PULocationID", "geometry"]].merge(df_filtered,
                                                                                on="PULocationID",
                                                                                how="inner"
                                                                                )
                    sample_pickup = df_filtered.sample_points(1, random_state=SimMetaData.random_seed).to_crs(4326)
                    df_filtered["pickup_longitude"] = sample_pickup.x
                    df_filtered["pickup_latitude"] = sample_pickup.y

                    df_filtered = df_filtered.rename(columns={"geometry": "geometry_x"})
                    df_filtered = shapefile[["DOLocationID", "geometry"]].merge(df_filtered,
                                                                                on="DOLocationID",
                                                                                how="inner"
                                                                                )
                    sample_dropoff = df_filtered.sample_points(1, random_state=SimMetaData.random_seed).to_crs(4326)
                    df_filtered["dropoff_longitude"] = sample_dropoff.x
                    df_filtered["dropoff_latitude"] = sample_dropoff.y
                    df_filtered = df_filtered[["pickup_datetime",
                                               "dropoff_datetime",
                                               "trip_distance",
                                               "pickup_longitude",
                                               "pickup_latitude",
                                               "dropoff_longitude",
                                               "dropoff_latitude"]]

            elif dataset_source == Dataset.CHICAGO.value:
                entire_df = pd.read_csv(dataset_path)
                df_filtered = entire_df[
                    ["Trip Start Timestamp", "Trip End Timestamp", "Trip Miles", "Pickup Centroid Longitude",
                     "Pickup Centroid Latitude", "Dropoff Centroid Longitude", "Dropoff Centroid Latitude",
                     "Trip Seconds"]
                ]

                df_filtered = df_filtered.rename(
                    columns={"Trip Start Timestamp": "pickup_datetime",
                             "Trip End Timestamp": "dropoff_datetime",
                             "Trip Miles": "trip_distance",
                             "Pickup Centroid Longitude": "pickup_longitude",
                             "Pickup Centroid Latitude": "pickup_latitude",
                             "Dropoff Centroid Longitude": "dropoff_longitude",
                             "Dropoff Centroid Latitude": "dropoff_latitude"
                             }
                )
                # fetch all rows that are valid between a specific time range
                df_filtered.loc[:, "pickup_datetime"] = pd.to_datetime(df_filtered["pickup_datetime"],
                                                                       format="%m/%d/%Y %I:%M:%S %p")
                df_filtered.loc[:, "dropoff_datetime"] = pd.to_datetime(df_filtered["dropoff_datetime"],
                                                                        format="%m/%d/%Y %I:%M:%S %p")
                df_filtered = df_filtered[
                    (df_filtered["pickup_datetime"] >= start_datetime) &
                    (df_filtered["pickup_datetime"] < end_datetime)]
                if df_filtered.empty:
                    raise ValueError("The filtered dataset is empty. Try adjusting state and end dates")

                # adding a random amount of minutes as pickups are rounded off to the nearest 15 mins in the dataset
                df_filtered.loc[:, "pickup_datetime"] += pd.to_timedelta(
                    np.random.randint(0, 15 * 60, len(df_filtered)), unit="m"
                ) / 60

                df_filtered.loc[:, "dropoff_datetime"] = (
                        df_filtered.loc[:, "pickup_datetime"]
                        + pd.to_timedelta(df_filtered.loc[:, "Trip Seconds"], unit="s")
                )

                # sample a certain percent of data
                sample_size = int(len(df_filtered) * percent_of_trips)
                df_filtered = df_filtered.sample(sample_size, ignore_index=True, axis="index",
                                                 random_state=SimMetaData.random_seed)
            elif dataset_source == Dataset.OLD_NYTAXI.value:
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
                if df_filtered.empty:
                    raise ValueError("The filtered dataset is empty. Try adjusting state and end dates")

                # sample a certain percent of data
                sample_size = int(len(df_filtered) * percent_of_trips)
                df_filtered = df_filtered.sample(sample_size, ignore_index=True, axis="index",
                                                 random_state=SimMetaData.random_seed)
            else:
                raise ValueError("No such dataset origin exists")

            # remove invalid rows based on pickup and dropoff datetime
            df_filtered = df_filtered[df_filtered["pickup_datetime"] <= df_filtered["dropoff_datetime"]]

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
            lower_percentile_lat_lon_latitude = np.copy(lower_percentile_lat_lon)
            upper_percentile_lat_lon_latitude = np.copy(upper_percentile_lat_lon)
            lower_percentile_lat_lon_longitude = np.copy(lower_percentile_lat_lon)
            upper_percentile_lat_lon_longitude = np.copy(upper_percentile_lat_lon)
            while abs(
                    DatasetParams.latitude_range_max - DatasetParams.latitude_range_min) > DatasetParams.delta_latitude:
                warnings.warn("The max and min latitudes are too far away from each other, "
                              "reducing the percentile by 5%.")
                lower_percentile_lat_lon_latitude = lower_percentile_lat_lon_latitude + 2.5
                upper_percentile_lat_lon_latitude = upper_percentile_lat_lon_latitude - 2.5
                DatasetParams.latitude_range_min, DatasetParams.latitude_range_max = np.percentile(
                    df_filtered["pickup_latitude"],
                    [lower_percentile_lat_lon_latitude, upper_percentile_lat_lon_latitude])
            while abs(
                    DatasetParams.longitude_range_max - DatasetParams.longitude_range_min) > DatasetParams.delta_longitude:
                warnings.warn("The max and min longitudes are too far away from each other, "
                              "reducing the percentile by 5%.")
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

            df_output = df_output.sort_values(by='pickup_datetime')
            df_output.reset_index(drop=True)
            # Step 3: save the dataframe into a CSV file or read the dataframe from a CSV file
            df_output.to_csv(sample_data_path)
        df_output["pickup_datetime"] = pd.to_datetime(df_output["pickup_datetime"])
        df_output["dropoff_datetime"] = pd.to_datetime(df_output["dropoff_datetime"])
        df_output["trip_time_min"] = (df_output["dropoff_datetime"] - df_output[
            "pickup_datetime"]).dt.total_seconds() / 60.0
        SimMetaData.avg_vel_mph = np.mean(df_output["trip_distance"]) / np.mean(df_output["trip_time_min"]) * 60
        warnings.warn(
            f"Average velocity changed based on the input data. New average velocity = {SimMetaData.avg_vel_mph}.")
        if DatasetParams.uniform_locations is True:
            df_output["pickup_latitude"], df_output["pickup_longitude"] = sample_unif_points_on_sphere(
                lon_max=DatasetParams.longitude_range_max,
                lon_min=DatasetParams.longitude_range_max - 0.1,
                lat_max=DatasetParams.latitude_range_max,
                lat_min=DatasetParams.latitude_range_max - 0.1,
                size=len(df_output)
            )
            df_output["dropoff_latitude"], df_output["dropoff_longitude"] = sample_unif_points_on_sphere(
                lon_max=DatasetParams.longitude_range_max,
                lon_min=DatasetParams.longitude_range_max - 0.1,
                lat_max=DatasetParams.latitude_range_max,
                lat_min=DatasetParams.latitude_range_max - 0.1,
                size=len(df_output)
            )
            df_output["trip_distance"] = calc_dist_between_two_points(
                start_lat=df_output["pickup_latitude"],
                start_lon=df_output["pickup_longitude"],
                end_lat=df_output["dropoff_latitude"],
                end_lon=df_output["dropoff_longitude"],
                dist_func=DistFunc.MANHATTAN.value,
                dist_correction_factor=1
            )
            df_output["trip_time_min"] = df_output["trip_distance"] / SimMetaData.avg_vel_mph * 60
            df_output["dropoff_datetime"] = df_output["pickup_datetime"] + pd.to_timedelta(df_output["trip_time_min"],
                                                                                           'm')

        # Plot all trips' start longitude in a histogram
        if not SimMetaData.quiet_sim:
            fig1, ax1 = plt.subplots()
            ax1.hist(df_output["pickup_longitude"], 5, ec='blue', label='pickup_longitude')
            ax1.legend(loc='upper left')
            plt.show()
            plt.close()

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

        # Plot all trips' start latitude in a histogram
        if not SimMetaData.quiet_sim:
            fig4, ax4 = plt.subplots()
            ax4.hist(df_output["pickup_latitude"], 5, ec='green', label='pickup_latitude')
            ax4.legend(loc='upper left')
            plt.show()
            plt.close()

        # Step 4: Calculate the Haversine distance correction factor
        if dist_func == DistFunc.MANHATTAN.value:
            dist_correction_factor = 1
            return df_output, dist_correction_factor
        elif dist_func == DistFunc.HAVERSINE.value:
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
            sampleDF_80_percent = df_output.sample(int(len(df_output) * 0.8), random_state=SimMetaData.random_seed)
            x = sampleDF_80_percent["haversine_distance"].values.reshape(-1,
                                                                         1)  # "values" converts it into a numpy array
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
        else:
            raise ValueError("No such distance function exists")


if __name__ == "__main__":
    # data_input = DataInput(percentile_lat_lon=99.9)
    # DataInput.plotuniftrip(data_input)

    # Use the code below to get the header of the input dataset
    entire_df = pd.read_parquet(
        'yellow_tripdata_2010-12.parquet',
        engine='fastparquet')
    print(entire_df.columns.values.tolist())