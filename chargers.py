from sim_metadata import SimMetaData, ChargerState, CarState, ChargingAlgoParams


class SuperCharger:

    def __init__(self,
                 idx,
                 n_posts,
                 env,
                 random=True,
                 lat=None,
                 lon=None):
        self.idx = idx
        self.n_posts = n_posts
        if random:
            lat = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lat)
            lon = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lon)
        self.lat = lat
        self.lon = lon
        self.occupancy = 0
        self.n_cars_waiting = 0
        self.state = ChargerState.AVAILABLE.value
        self.queue_list = []
        self.car_tracker = None
        self.env = env

    def queueing_at_charger(self, car_id, end_soc):
        # Add input car and end SOC (if provided) to the charger queue
        if car_id is not None:
            self.queue_list.append([car_id, end_soc])
        # For the cars no need to wait, they still will be in the list and be removed
        while (self.state == ChargerState.AVAILABLE.value) and (len(self.queue_list) != 0):
            # Head of the line car starts charging (call car_charging function)
            car = self.car_tracker[self.queue_list[0][0]]
            # time out inside car_charging function only affects the function itself
            # occupancy increases at the same time charging happens
            if car.state != CarState.WAITING_FOR_CHARGER.value:
                raise ValueError(f"Car {self.id} is not at the charger to be charged")
            car.prev_charging_process = self.env.process(car.car_charging(self.idx, self.queue_list[0][1]))
            # Increase the occupancy of the charger by one
            self.occupancy += 1
            # If all posts are busy, then set charger state equal to busy
            if self.n_posts == self.occupancy and not ChargingAlgoParams.infinite_chargers:
                self.state = ChargerState.BUSY.value
            # Remove the head of the line car from the queue
            del self.queue_list[0]

    def to_dict(self):
        return {
            "idx": self.idx,
            "lat": self.lat,
            "lon": self.lon,
            "n_posts": self.n_posts
        }
