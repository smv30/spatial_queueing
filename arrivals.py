import numpy as np
from sim_metadata import TripState, SimMetaData


class Trip(object):

    def __init__(self,
                 env,
                 trip_id,
                 arrival_time,
                 state,
                 random=True,
                 start_lat=None,
                 start_lon=None,
                 end_lat=None,
                 end_lon=None,
                 ):
        if random:
            start_lat = np.random.uniform(0, 1)
            start_lon = np.random.uniform(0, 1)
            end_lat = np.random.uniform(0, 1)
            end_lon = np.random.uniform(0, 1)
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.arrival_time = arrival_time
        self.trip_id = trip_id
        self.env = env
        self.state = state

    def calc_trip_time(self):
        trip_dist = ((self.start_lat - self.end_lat) ** 2 + (self.start_lon - self.end_lon) ** 2) ** 0.5
        return trip_dist / SimMetaData.avg_vel

    def update_trip_state(self, renege_time):

        yield self.env.timeout(renege_time)

        if self.state == TripState.WAITING.value:
            self.state = TripState.RENEGED
            print(f"Trip {self.trip_id} reneged after waiting for {renege_time} time")

    def to_dict(self):
        return {
            "trip_id": self.trip_id,
            "arrival_time": self.arrival_time,
            "start_lat": self.start_lat,
            "start_lon": self.start_lon,
            "end_lat": self.end_lat,
            "end_lon": self.end_lon,
            "state": self.state
        }



