import numpy as np
from sim_metadata import TripState, SimMetaData


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
                 ):
        if random:
            start_lat = np.random.uniform(0, SimMetaData.max_lat)
            start_lon = np.random.uniform(0, SimMetaData.max_lon)
            end_lat = np.random.uniform(0, SimMetaData.max_lat)
            end_lon = np.random.uniform(0, SimMetaData.max_lon)
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.arrival_time_min = arrival_time_min
        self.trip_id = trip_id
        self.env = env
        self.state = state

    def calc_trip_time(self):
        trip_dist_mi = ((self.start_lat - self.end_lat) ** 2 + (self.start_lon - self.end_lon) ** 2) ** 0.5
        return trip_dist_mi / SimMetaData.avg_vel_mph * 60

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



