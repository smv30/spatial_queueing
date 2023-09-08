import numpy as np
from haversine import haversine, haversine_vector, Unit
from spatial_queueing import real_life_data_input


def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon):
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
    return real_life_data_input.slope * haversine_distance

if __name__ == "__main__":
    start_lat = 76.345678
    start_lon = 55.765
    end_lat = [55, 45]
    end_lon = [45, 55]
    calc_dist_between_two_points(start_lat=start_lat,
                                 start_lon=start_lon,
                                 end_lat=end_lat,
                                 end_lon=end_lon)
