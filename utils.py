def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon):
    return ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5


if __name__ == "__main__":
    dist = calc_dist_between_two_points(start_lat=1,
                                        start_lon=4,
                                        end_lat=5,
                                        end_lon=8)
    print(dist)
