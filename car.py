from sim_metadata import SimMetaData, CarState, TripState
from chargers import SuperCharger
import simpy
import numpy as np


class Car:
    def __init__(self,
                 car_id,
                 env,
                 list_chargers,
                 random=True,
                 lat=None,
                 lon=None,
                 soc=None,
                 state=None,
                 ):
        if random:
            lat = np.random.uniform(0, SimMetaData.max_lat)
            lon = np.random.uniform(0, SimMetaData.max_lon)
            soc = np.random.uniform(0.5, 1)
            state = CarState.IDLE.value
        self.id = car_id
        self.lat = lat
        self.lon = lon
        self.state = state
        self.soc = soc
        self.env = env
        self.list_chargers = list_chargers

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "soc": self.soc,
            "state": self.state
        }

    def run_trip(self, trip, end_soc=None, charger_idx=None):
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is currently not idle to be matched")
        trip.state = TripState.MATCHED
        pickup_dist_mi = ((trip.start_lat - self.lat) ** 2 + (trip.start_lon - self.lon) ** 2) ** 0.5
        pickup_time_min = pickup_dist_mi / SimMetaData.avg_vel_mph * 60
        self.state = CarState.DRIVING_WITHOUT_PASSENGER.value
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} picking up Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(pickup_time_min)

        trip_dist_mi = ((trip.start_lat - trip.end_lat) ** 2 + (trip.start_lon - trip.end_lon) ** 2) ** 0.5
        trip_time_min = trip_dist_mi / SimMetaData.avg_vel_mph * 60
        self.lat = trip.start_lat
        self.lon = trip.start_lon
        self.state = CarState.DRIVING_WITH_PASSENGER.value
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} driving with Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(trip_time_min)

        consumption = SimMetaData.consumption_kwhpmi * (pickup_dist_mi + trip_dist_mi)
        delta_soc = consumption / SimMetaData.pack_size_kwh
        self.lat = trip.end_lat
        self.lon = trip.end_lon
        self.state = CarState.IDLE.value
        self.soc = self.soc - delta_soc
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} finished trip with an SOC equal to {self.soc} at time {self.env.now}")
        if self.soc < 0:
            raise ValueError("SOC cannot be less than 0")
        if end_soc:
            self.env.process(self.run_charge(end_soc, charger_idx))

    def run_charge(self, end_soc, charger_idx):
        charger = self.list_chargers[charger_idx]
        charger_lat = charger.lat
        charger_lon = charger.lon
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is not idle to be sent to charge")

        dist_to_charger_mi = ((self.lat - charger_lat) ** 2 + (self.lon - charger_lon) ** 2) ** 0.5
        drive_time_min = dist_to_charger_mi / SimMetaData.avg_vel_mph * 60
        self.state = CarState.DRIVING_TO_CHARGER.value
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} driving to charger {charger_idx} at time {self.env.now} with initial soc {self.soc}")
        yield self.env.timeout(drive_time_min)

        consumption_kwh = dist_to_charger_mi * SimMetaData.consumption_kwhpmi
        charge_kwh = (end_soc - self.soc) * SimMetaData.pack_size_kwh
        charge_time_min = (charge_kwh + consumption_kwh) / SimMetaData.charge_rate_kw * 60
        self.lat = charger_lat
        self.lon = charger_lon
        self.state = CarState.CHARGING.value
        charger.update_occupancy(increase=True)
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} starting to charge at charger {charger_idx} at time {self.env.now}")
        yield self.env.timeout(charge_time_min)

        charger.update_occupancy(increase=False)
        self.state = CarState.IDLE.value

        self.soc = end_soc
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} finished charging at charger {charger_idx}, gained {charge_kwh} kwh in {charge_time_min} mins")
        if self.soc < 0 or self.soc > 1:
            raise ValueError("SOC must be between 0 and 1")


if __name__ == "__main__":
    env = simpy.Environment()
    n_chargers = SimMetaData.n_charger_loc
    n_posts = SimMetaData.n_posts
    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts)
        list_chargers.append(charger)

    car = Car(car_id=0,
              lat=0,
              lon=1,
              state=CarState.IDLE.value,
              soc=0.5,
              env=env,
              list_chargers=list_chargers
              )
    env.process(car.run_charge(5, 5))
    env.run()
