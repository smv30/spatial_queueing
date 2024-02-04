from sim_metadata import ChargerState, CarState, ChargingAlgoParams, DatasetParams
from utils import sample_unif_points_on_sphere


class SuperCharger:

    def __init__(self,
                 idx,
                 n_posts,
                 env,
                 df_arrival_sequence,
                 random=False,
                 lat=None,
                 lon=None):
        self.idx = idx
        self.n_posts = n_posts
        # set the charger locations to be close to most drop-off locations
        # 1) divide the area -> output a list of lats % lons; 2) count the number of trips having drop-off locations
        # within each lat & lon; 3) divide each number by the total number of trips to get the probability that the
        # trip drop-off within each lat & lon; 4) sample the charger locations based on the probability distribution
        if random:
            lat, lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                    lon_max=DatasetParams.longitude_range_max,
                                                    lat_min=DatasetParams.latitude_range_min,
                                                    lat_max=DatasetParams.latitude_range_max)
        else:
            sample_trip = df_arrival_sequence.sample(1)
            lon = sample_trip["dropoff_longitude"].values[0]
            lat = sample_trip["dropoff_latitude"].values[0]
        self.lat = lat
        self.lon = lon
        self.occupancy = 0
        self.n_cars_waiting = 0
        self.state = ChargerState.AVAILABLE.value
        self.queue_list = []
        self.car_tracker = None
        self.env = env

    def queueing_at_charger(self, car_id, end_soc):
        # Add the input car and its end SOC (if provided) to the charger queue
        if car_id is not None:
            self.queue_list.append([car_id, end_soc])
        # If a car arrives and finds an empty charger, it will be added to the list and removed immediately
        while (self.state == ChargerState.AVAILABLE.value) and (len(self.queue_list) != 0):
            # The first car of the line starts charging (call car_charging function)
            car = self.car_tracker[self.queue_list[0][0]]
            # time out inside car_charging function only affects the function itself
            # occupancy increases at the same time charging happens
            if car.state != CarState.WAITING_FOR_CHARGER.value:
                raise ValueError(f"Car {self.id} is not at the charger to be charged")
            car.prev_charging_process = self.env.process(car.car_charging(self.idx, self.queue_list[0][1]))
            # Increase the occupancy of the charger by one
            self.occupancy += 1
            # If all posts are busy, then set charger state to BUSY
            if self.n_posts == self.occupancy and not ChargingAlgoParams.infinite_chargers:
                self.state = ChargerState.BUSY.value
            # Remove the first car of the line from the queue
            del self.queue_list[0]

    def to_dict(self):
        return {
            "idx": self.idx,
            "lat": self.lat,
            "lon": self.lon,
            "n_posts": self.n_posts,
            "state": self.state,
            "n_available_posts": max(self.n_posts - len(self.queue_list), 0)
        }
