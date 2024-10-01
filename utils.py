import numpy as np
from haversine import haversine, haversine_vector, Unit
from sim_metadata import SimMetaData, DatasetParams, DistFunc
import itertools


def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon, dist_func, dist_correction_factor=1):
    if np.array(start_lat).size != np.array(end_lat).size:
        if np.array(start_lat).size == 1:
            start_lat_list = np.ones(len(end_lat)) * start_lat
            start_lon_list = np.ones(len(end_lon)) * start_lon
            end_lat_list = end_lat
            end_lon_list = end_lon
        elif np.array(end_lat).size == 1:
            start_lat_list = start_lat
            start_lon_list = start_lon
            end_lat_list = np.ones(len(start_lat)) * end_lat
            end_lon_list = np.ones(len(start_lon)) * end_lon
        else:
            raise ValueError("The number of start locations and end locations do not match.")
        if dist_func == DistFunc.HAVERSINE.value:
            start = list(zip(start_lat_list, start_lon_list))
            end = list(zip(end_lat_list, end_lon_list))
            distance = haversine_vector(start, end, unit=Unit.MILES)
        elif dist_func == DistFunc.MANHATTAN.value:
            start1 = list(zip(start_lat_list, start_lon_list))
            end1 = list(zip(start_lat_list, end_lon_list))
            start2 = list(zip(start_lat_list, start_lon_list))
            end2 = list(zip(end_lat_list, start_lon_list))
            distance = haversine_vector(start1, end1, unit=Unit.MILES) + haversine_vector(start2, end2, unit=Unit.MILES)
        else:
            raise ValueError("The number of start locations and end locations do not match.")
    else:
        if np.array(start_lat).size == 1:
            if dist_func == DistFunc.HAVERSINE.value:
                start = (start_lat, start_lon)
                end = (end_lat, end_lon)
                distance = haversine(start, end, unit=Unit.MILES)
            elif dist_func == DistFunc.MANHATTAN.value:
                start1 = (start_lat, start_lon)
                end1 = (start_lat, end_lon)
                start2 = (start_lat, start_lon)
                end2 = (end_lat, start_lon)
                distance = haversine(start1, end1, unit=Unit.MILES) + haversine(start2, end2, unit=Unit.MILES)
            else:
                raise ValueError("The number of start locations and end locations do not match.")
        else:
            if dist_func == DistFunc.HAVERSINE.value:
                start = list(zip(start_lat, start_lon))
                end = list(zip(end_lat, end_lon))
                distance = haversine_vector(start, end, unit=Unit.MILES)
            elif dist_func == DistFunc.MANHATTAN.value:
                start1 = list(zip(start_lat, start_lon))
                end1 = list(zip(start_lat, end_lon))
                start2 = list(zip(start_lat, start_lon))
                end2 = list(zip(end_lat, start_lon))
                distance = (
                        haversine_vector(start1, end1, unit=Unit.MILES)
                        + haversine_vector(start2, end2, unit=Unit.MILES)
                )
            else:
                raise ValueError("The number of start locations and end locations do not match.")
    return dist_correction_factor * distance


def sample_unif_points_on_sphere(lon_min, lon_max, lat_min, lat_max, size=None):
    theta_min = lon_min + 180
    theta_max = lon_max + 180
    phi_min = lat_min + 90
    phi_max = lat_max + 90
    # theta is longitude; phi is latitude
    unif_lon = SimMetaData.random_seed_gen.uniform(theta_min / 360, theta_max / 360, size=size)  # radians
    unif_lat = SimMetaData.random_seed_gen.uniform(min((np.cos(phi_min * np.pi / 180) + 1) / 2,
                                                       (np.cos(phi_max * np.pi / 180) + 1) / 2),
                                                   max((np.cos(phi_min * np.pi / 180) + 1) / 2,
                                                       (np.cos(phi_max * np.pi / 180) + 1) / 2),
                                                   size=size
                                                   )  # radians
    lon = unif_lon * 360 - 180  # degrees
    lat = 180 / np.pi * np.arccos(2 * unif_lat - 1) - 90  # degrees
    return lat, lon


if __name__ == "__main__":
    lat1 = 40.7128  # Latitude of point 1
    lon1 = -74.0060  # Longitude of point 1
    lat2 = 40.0522  # Latitude of point 2
    lon2 = -74.2437  # Longitude of point 2
    dist_correction_factor = 1

    sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                 lon_max=DatasetParams.longitude_range_max,
                                 lat_min=DatasetParams.latitude_range_min,
                                 lat_max=DatasetParams.latitude_range_max)

    dist = calc_dist_between_two_points(start_lat=lat1,
                                        start_lon=lon1,
                                        end_lat=lat2,
                                        end_lon=lon2,
                                        dist_correction_factor=dist_correction_factor,
                                        dist_func=DistFunc.MANHATTAN.value)
    print(dist)
