from sim_metadata import SimMetaData, CarState
import simpy
import numpy as np


class Car:
    def __init__(self,
                 car_id,
                 env,
                 random=True,
                 lat=None,
                 lon=None,
                 soc=None,
                 state=None,
                 ):
        if random:
            lat = np.random.uniform(0, 1)
            lon = np.random.uniform(0, 1)
            soc = np.random.uniform(0.5, 1)
            state = CarState.IDLE.value
        self.id = car_id
        self.lat = lat
        self.lon = lon
        self.state = state
        self.soc = soc
        self.env = env

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "soc": self.soc,
            "state": self.state
        }

    def run_trip(self, trip):
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is currently not idle to be matched")
        pickup_dist = ((trip.start_lat - self.lat) ** 2 + (trip.start_lon - self.lon) ** 2) ** 0.5
        pickup_time = pickup_dist / SimMetaData.avg_vel
        self.state = CarState.DRIVING_WITHOUT_PASSENGER.value
        print(f"Car {self.id} picking up Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(pickup_time)

        trip_dist = ((trip.start_lat - trip.end_lat) ** 2 + (trip.start_lon - trip.end_lon) ** 2) ** 0.5
        trip_time = trip_dist / SimMetaData.avg_vel
        self.lat = trip.start_lat
        self.lon = trip.start_lon
        self.state = CarState.DRIVING_WITH_PASSENGER.value
        print(f"Car {self.id} driving with Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(trip_time)

        consumption = SimMetaData.consumption_kwhpmi * (pickup_dist + trip_dist)
        delta_soc = consumption / SimMetaData.pack_size_kwh
        self.lat = trip.end_lat
        self.lon = trip.end_lon
        self.state = CarState.IDLE.value
        self.soc = self.soc - delta_soc
        print(f"Car {self.id} finished trip with an SOC equal to {self.soc} at time {self.env.now}")
        if self.soc < 0:
            raise ValueError("SOC cannot be less than 0")

    def run_charge(self, charge_kwh, charger_lat, charger_lon):
        dist_to_charger = ((self.lat - charger_lat) ** 2 + (self.lon - charger_lon) ** 2) ** 0.5
        drive_time = dist_to_charger / SimMetaData.avg_vel
        self.state = CarState.DRIVING_TO_CHARGER.value
        yield self.env.timeout(drive_time)

        charge_time = charge_kwh / SimMetaData.charge_rate_kw * 60
        self.lat = charger_lat
        self.lon = charger_lon
        self.state = CarState.CHARGING.value
        yield self.env.timeout(charge_time)

        self.state = CarState.IDLE.value

        consumption = dist_to_charger * SimMetaData.consumption_kwhpmi
        delta_soc = (consumption - charge_kwh) / SimMetaData.pack_size_kwh
        self.soc = self.soc - delta_soc

        if self.soc < 0 or self.soc > 1:
            raise ValueError("SOC must be between 0 and 1")


if __name__ == "__main__":
    env = simpy.Environment()
    car = Car(car_id=0,
              lat=0,
              lon=1,
              state=CarState.IDLE.value,
              soc=0.5,
              env=env
              )
    env.process(car.run_charge(5, 5, 5))
    env.run()
    for i in range(100):
        if env.now == i:
            print(car.state)
