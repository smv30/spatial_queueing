import numpy as np
from haversine import haversine, haversine_vector, Unit
# from spatial_queueing.sim_metadata import SimMetaData
from sim_metadata import SimMetaData, DatasetParams
import itertools


def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon, dist_correction_factor=1):
    if np.array(start_lat).size != np.array(end_lat).size:
        if np.array(start_lat).size == 1:
            start_lat_list = np.ones(len(end_lat)) * start_lat
            start_lon_list = np.ones(len(end_lon)) * start_lon
            start = list(zip(start_lat_list, start_lon_list))
            end = list(zip(end_lat, end_lon))
            haversine_distance = haversine_vector(start, end, unit=Unit.MILES)
        elif np.array(end_lat).size == 1:
            end_lat_list = np.ones(len(start_lat)) * end_lat
            end_lon_list = np.ones(len(start_lon)) * end_lon
            start = list(zip(start_lat, start_lon))
            end = list(zip(end_lat_list, end_lon_list))
            haversine_distance = haversine_vector(start, end, unit=Unit.MILES)
        else:
            raise ValueError("The number of start locations and end locations do not match.")
    else:
        if np.array(start_lat).size == 1:
            start = (start_lat, start_lon)
            end = (end_lat, end_lon)
            haversine_distance = haversine(start, end, unit=Unit.MILES)
        else:
            start = list(zip(start_lat, start_lon))
            end = list(zip(end_lat, end_lon))
            haversine_distance = haversine_vector(start, end, unit=Unit.MILES)
    return dist_correction_factor * haversine_distance


def sample_unif_points_on_sphere(lon_min, lon_max, lat_min, lat_max):
    theta_min = lon_min + 180
    theta_max = lon_max + 180
    phi_min = lat_min + 90
    phi_max = lat_max + 90
    # theta is longitude; phi is latitude
    unif_lon = SimMetaData.random_seed_gen.uniform(theta_min / 360, theta_max / 360)  # radians
    unif_lat = SimMetaData.random_seed_gen.uniform(min((np.cos(phi_min * np.pi / 180) + 1) / 2,
                                                       (np.cos(phi_max * np.pi / 180) + 1) / 2),
                                                   max((np.cos(phi_min * np.pi / 180) + 1) / 2,
                                                       (np.cos(phi_max * np.pi / 180) + 1) / 2))  # radians
    lon = unif_lon * 360 - 180  # degrees
    lat = 180 / np.pi * np.arccos(2 * unif_lat - 1) - 90  # degrees
    return lat, lon


def sample_random_chargers(lon_min, lon_max, lat_min, lat_max):
    theta_min = DatasetParams.longitude_range_min + 180
    theta_max = DatasetParams.longitude_range_max + 180
    phi_min = DatasetParams.latitude_range_min + 90
    phi_max = DatasetParams.latitude_range_max + 90
    delta = (theta_max - theta_min) / 20
    theta_list = np.arange(start=theta_min, stop=theta_max, step=delta).tolist()
    theta_list.append(theta_max)
    phi_list = []
    phi = phi_min
    while phi <= phi_max:
        phi_list.append(phi)
        phi = phi - delta / np.sin(phi)
    phi_list.append(phi_max)
    theta_list = [each_theta - 180 for each_theta in theta_list]  # length=21
    phi_list = [each_phi - 90 for each_phi in phi_list]  # length=38
    grid_list = list(itertools.product(theta_list, phi_list))  # length=798
    grid_dict_2D = np.array(grid_list).reshape(len(theta_list), len(phi_list), 2)
    # grid_dict = dict.fromkeys(grid_list, 0)
    # df_sample_trips =
    # for row_index in range(len(df_sample_trips)):
    #     dropoff_lon = df_sample_trips.loc[row_index, "dropoff_longitude"]
    #     dropoff_lat = df_sample_trips.loc[index, "dropoff_latitude"]
    # key = (-74.0, dropoff_lat)
    # grid_dict[key]
    print(grid_list)
    print(grid_dict_2D[0])
    # return lat, lon


if __name__ == "__main__":
    # sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
    #                                        lon_max=DatasetParams.longitude_range_max,
    #                                        lat_min=DatasetParams.latitude_range_min,
    #                                        lat_max=DatasetParams.latitude_range_max)
    # calc_dist_between_two_points(start_lat=start_lat,
    #                              start_lon=start_lon,
    #                              end_lat=end_lat,
    #                              end_lon=end_lon,
    #                              dist_correction_factor=dist_correction_factor)
    sample_random_chargers(lon_min=-74.0098531455,
                           lon_max=-73.77671585449998,
                           lat_min=40.633872634999996,
                           lat_max=40.795812)
