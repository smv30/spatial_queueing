from sim_metadata import SimMetaData


class SuperCharger:

    def __init__(self,
                 idx,
                 n_posts,
                 state,
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
        self.state = state
        self.occupancy = 0
        self.n_cars_waiting = 0

    def update_occupancy(self, increase=True):
        if increase:
            if self.occupancy < self.n_posts:
                self.occupancy = self.occupancy + 1
            else:
                self.n_cars_waiting = self.n_cars_waiting + 1
        else:
            if self.n_cars_waiting > 0:
                self.n_cars_waiting = self.n_cars_waiting - 1
            else:
                self.occupancy = self.occupancy - 1

    def to_dict(self):
        return {
            "idx": self.idx,
            "lat": self.lat,
            "lon": self.lon,
            "state": self.state,
            "n_posts": self.n_posts
        }
