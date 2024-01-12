from sim_metadata import TripState, SimMetaData, DatasetParams
from spatial_queueing.utils import sample_unif_points_on_sphere


class Trip(object):

    def __init__(self,
                 env,
                 trip_id,
                 arrival_time_min,
                 state,
                 random=True,
                 start_lat=None,
                 start_lon=None,
                 end_lat=None,
                 end_lon=None,
                 trip_time_min=None
                 ):
        if random:
            start_lat, start_lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                                lon_max=DatasetParams.longitude_range_max,
                                                                lat_min=DatasetParams.latitude_range_min,
                                                                lat_max=DatasetParams.latitude_range_max)
            end_lat, end_lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                            lon_max=DatasetParams.longitude_range_max,
                                                            lat_min=DatasetParams.latitude_range_min,
                                                            lat_max=DatasetParams.latitude_range_max)
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.arrival_time_min = arrival_time_min
        self.trip_id = trip_id
        self.env = env
        self.state = state
        self.pickup_time_min = 0
        self.trip_time_min = trip_time_min

    def update_trip_state(self, renege_time_min):

        yield self.env.timeout(renege_time_min)

        if self.state == TripState.WAITING.value:
            self.state = TripState.RENEGED
            if not SimMetaData.quiet_sim:
                print(f"Trip {self.trip_id} reneged after waiting for {renege_time_min} time")

    def to_dict(self):
        return {
            "trip_id": self.trip_id,
            "arrival_time": self.arrival_time_min,
            "start_lat": self.start_lat,
            "start_lon": self.start_lon,
            "end_lat": self.end_lat,
            "end_lon": self.end_lon,
            "state": self.state
        }
