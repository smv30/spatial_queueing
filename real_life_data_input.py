import pandas as pd
import random
import numpy as np
import geopandas as gpd
from haversine import haversine, haversine_vector, Unit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os.path
from sim_metadata import SimMetaData
from dateutil import parser

# Step 1
if not os.path.isfile("sampledata.csv") or SimMetaData.test is False:
    dataset = pd.read_parquet('D:\GT\Research\Spatial Queueing\spatial_queueing\yellow_tripdata_2010-12.parquet',
                              engine='fastparquet')
    df = dataset[["pickup_datetime", "dropoff_datetime", "trip_distance", "pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"]]

    # Step 2: fetch all rows that match a randomly selected number from 1-31
    df.sort_values(by='pickup_datetime', inplace=True)
    index = 0
    sampleDF = pd.DataFrame()
    while index < df.shape[0]:
        pickup_datetime = parser.parse(df["pickup_datetime"].iloc[index])
        if pickup_datetime.day == 1 and pickup_datetime.hour == 0:
            if df.iloc[index].loc["pickup_longitude"] <= -70 and df.iloc[index].loc["pickup_latitude"] >= 40 \
                    and df.iloc[index].loc["dropoff_longitude"] <= -70 and df.iloc[index].loc["dropoff_latitude"] >= 40:
                if df.iloc[index].loc["pickup_longitude"] >= -90 and df.iloc[index].loc["pickup_latitude"] <= 90 \
                        and df.iloc[index].loc["dropoff_longitude"] >= -90 and df.iloc[index].loc["dropoff_latitude"] <= 90:
                    sampleDF = sampleDF.append(df.iloc[index])
            index += 1
        else:
            break

    # Step 2: generate sample data and sort the rows by pickup datetime
    # sampleDF = df.sample(n=100, random_state=1)
    # sampleDF.sort_values(by='pickup_datetime', inplace=True)

    # Step 3: save the dataframe into CSV or read from CSV file
    sampleDF.to_csv("D:\GT\Research\Spatial Queueing\spatial_queueing\sampledata.csv")
else:
    sampleDF = pd.read_csv("D:\GT\Research\Spatial Queueing\spatial_queueing\sampledata.csv")

# # Step 4-1: Calculate Haversine distance
print("-------------Haversine Distance------------")
sampleDF["haversine_distance"] = ""
sampleDF["pickup"] = list(zip(sampleDF["pickup_latitude"], sampleDF["pickup_longitude"]))
sampleDF["dropoff"] = list(zip(sampleDF["dropoff_latitude"], sampleDF["dropoff_longitude"]))
sampleDF["haversine_distance"] = haversine_vector(list(sampleDF["pickup"]), list(sampleDF["dropoff"]),
                                                  unit=Unit.MILES)  # in miles
print(sampleDF[["haversine_distance", "trip_distance"]])

# Step 4-2: Linear Regression for Haversine distance -> Plot, R-Squared value, function, Mean Squared Error
sampleDF_80_percent = sampleDF.sample(80)
X = sampleDF_80_percent["haversine_distance"].values.reshape(-1, 1)  # values converts it into a numpy array
Y = sampleDF_80_percent["trip_distance"].values.reshape(-1, 1)  # -1 means calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression(fit_intercept=False)  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
r_squared = linear_regressor.score(X, Y)  # calculate R-Squared of regression model
print(f"R-Squared value: {r_squared}")  # Question: calculate r-squared value for 80% or 20%?

Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

# get the function (slope and intercept)
slope = linear_regressor.coef_[0][0]
print('Slope:', slope)
intercept = linear_regressor.intercept_
print('Intercept:', intercept)

extract_idx = list(set(sampleDF.index) - set(sampleDF_80_percent.index))
sampleDF_20_percent = sampleDF.loc[extract_idx]
MSE = mean_squared_error(Y, Y_pred)  # calculate Mean Squared Error
print(f"Mean Squared Error: {MSE}")

# # Datetime subtraction test
# # General example
# a = parser.parse("2012-01-19 17:21:00")
# b = parser.parse("2012-01-20 17:21:00")
# c = b - a
# # Specific example using sampleDF
# x = parser.parse(sampleDF.iloc[3, 1])
# y = parser.parse(sampleDF.iloc[4, 1])
# z = y - x
# subtraction_time = int(z.total_seconds() / 60)
# print(f"Total minutes between two datetime is: {subtraction_time}")


# # Step 5-1: Calculate Euclidean distance
# print("-------------Euclidean Distance------------")
# sampleDF["euclidean_distance"] = ""
# # sampleDF["pickup"] = list(zip(sampleDF["pickup_latitude"], sampleDF["pickup_longitude"]))
# # sampleDF["dropoff"] = list(zip(sampleDF["dropoff_latitude"], sampleDF["dropoff_longitude"]))
# # sampleDF["euclidean_distance"] = np.sqrt(np.sum(np.square(sampleDF["pickup"] - sampleDF["dropoff"])))
# # sampleDF["euclidean_distance"] = np.sqrt(np.sum(tuple(map(lambda i, j: np.square(i - j), sampleDF["pickup"], sampleDF["dropoff"]))))
# print(sampleDF)
# sampleDF["euclidean_distance"] = np.sqrt((np.square(sampleDF["pickup_latitude"] - sampleDF["dropoff_latitude"]) +
#                                             np.square(sampleDF["pickup_longitude"] - sampleDF["dropoff_longitude"])))
# print(sampleDF[["euclidean_distance", "trip_distance"]])

# # Step 6-1: Calculate L1 Distance
# print("-------------L1 Distance------------")
# sampleDF["l1_distance"] = ""
# # have to use for loop
#
# # Step 6-1: Calculate Lp Distance
# print("-------------Lp Distance------------")
# sampleDF["lp_distance"] = ""
# # Did not find Lp distance

# # Step 10
# shapefile = gpd.read_file("taxi_zones.shp")
# shapefile["PULocationID"] = shapefile["LocationID"]
# # combine LocationID with geometry to a new gpd dataframe (only two columns)
# # print(shapefile["LocationID"])
# # print(shapefile['geometry'].centroid)
# # use pd.merge() to do inner join
# print(shapefile.merge(sampleDF, on="PULocationID"))
# # print(shapefile.merge(sampleDF, on="DOLocationID"))
#
# # import the dataset
# # See what entries it has & how many columns want to keep
# # Step 1: filter these columns (refer to "Summary")
# # Step 2: select random rows using df.sample(# of rows)
# # Step 3: order the rows based on the time (request_datetime)
# # Step 4: Use taxi_zones.shp to calculate the midpoint of each PULocationID, DOLocationID for each row
# # Step 5: Add two columns of midpoints (PULocationIDMidpoint, DOLocationIDMidpoint) to the table
# # Step 6: save as csv file
#
# # NY dataset should have: (do this first)
# # 1. change the interarrival time as the latter arrival time subtract the previous arrival time
# # 2. change the origin and destination
# # use enum to see whether use random dataset or NY dataset
# # add feature: matching every minute (now we are matching every trip) -> change matching algorithm
# # Essay: compare matching instantaneously vs. waiting for some time
#
# # create a random dataset which has same columns as the NY one (another dataset)


# To-do list:
# 1. add a function in this file to return the slope back to utils
