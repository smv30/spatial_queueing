def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon):
    return ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5
