from sim_metadata import SimMetaData, CarState, TripState, ChargingAlgoParams
from chargers import SuperCharger
from arrivals import Trip
from utils import calc_dist_between_two_points
import simpy


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
            lat = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lat)
            lon = SimMetaData.random_seed_gen.uniform(0, SimMetaData.max_lon)
            soc = SimMetaData.random_seed_gen.uniform(0.7, 0.9)
            state = CarState.IDLE.value
        self.id = car_id
        self.lat = lat
        self.lon = lon
        self.state = state
        self.soc = soc
        self.env = env
        self.list_chargers = list_chargers
        self.state_start_time = 0
        self.prev_charging_process = None
        self.n_of_charging_stops = 0
        self.total_drive_to_charge_time = 0

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "soc": self.soc,
            "state": self.state,
            "state_start_time": self.state_start_time
        }

    def run_trip(self, trip, end_soc=None, charger_idx=None):
        # If the car is going to charge or charging, interrupt that process and update the SOC
        if self.state in (CarState.DRIVING_TO_CHARGER.value, CarState.CHARGING.value):
            self.prev_charging_process.interrupt()
            if not SimMetaData.quiet_sim:
                print(f"Car {self.id} was interrupted while charging at time {self.env.now}")
            time_spent_in_this_state_min = self.env.now - self.state_start_time
            if self.state == CarState.DRIVING_TO_CHARGER.value:
                consumption_kwh = (
                        time_spent_in_this_state_min * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / 60
                )
                delta_soc = consumption_kwh / SimMetaData.pack_size_kwh
                self.soc = self.soc - delta_soc
            else:
                charging_kwh = time_spent_in_this_state_min * SimMetaData.charge_rate_kw / 60
                delta_soc = charging_kwh / SimMetaData.pack_size_kwh
                self.soc = self.soc + delta_soc
            self.state = CarState.IDLE.value
            self.state_start_time = self.env.now

        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is currently finishing a trip and cannot be matched")
        trip.state = TripState.MATCHED
        pickup_dist_mi = calc_dist_between_two_points(start_lat=trip.start_lat,
                                                      start_lon=trip.start_lon,
                                                      end_lat=self.lat,
                                                      end_lon=self.lon)
        pickup_time_min = pickup_dist_mi / SimMetaData.avg_vel_mph * 60
        trip.pickup_time_min = pickup_time_min
        self.state = CarState.DRIVING_WITHOUT_PASSENGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} picking up Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(pickup_time_min)

        trip_time_min = trip.calc_trip_time()
        trip_dist_mi = trip_time_min / 60 * SimMetaData.avg_vel_mph
        self.lat = trip.start_lat
        self.lon = trip.start_lon
        self.state = CarState.DRIVING_WITH_PASSENGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} driving with Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(trip_time_min)

        consumption = SimMetaData.consumption_kwhpmi * (pickup_dist_mi + trip_dist_mi)
        delta_soc = consumption / SimMetaData.pack_size_kwh
        self.lat = trip.end_lat
        self.lon = trip.end_lon
        self.state = CarState.IDLE.value
        self.state_start_time = self.env.now
        self.soc = self.soc - delta_soc
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} finished trip with an SOC equal to {self.soc} at time {self.env.now}")
        if self.soc < 0:
            raise ValueError("SOC cannot be less than 0")
        if ChargingAlgoParams.send_all_idle_cars_to_charge:
            self.prev_charging_process = self.env.process(self.run_charge(1, charger_idx))
        elif end_soc:
            self.prev_charging_process = self.env.process(self.run_charge(end_soc, charger_idx))

    def run_charge(self, end_soc, charger_idx):
        charger = self.list_chargers[charger_idx]
        charger_lat = charger.lat
        charger_lon = charger.lon
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is not idle to be sent to charge")

        if not ChargingAlgoParams.infinite_chargers:
            dist_to_charger_mi = ((self.lat - charger_lat) ** 2 + (self.lon - charger_lon) ** 2) ** 0.5
        else:
            dist_to_charger_mi = 0.001

        drive_time_min = dist_to_charger_mi / SimMetaData.avg_vel_mph * 60
        self.state = CarState.DRIVING_TO_CHARGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} driving to charger {charger_idx} at time {self.env.now} with initial soc {self.soc}")
        try:
            yield self.env.timeout(drive_time_min)
        except:
            return 0

        consumption_kwh = dist_to_charger_mi * SimMetaData.consumption_kwhpmi
        charge_kwh = (end_soc - self.soc) * SimMetaData.pack_size_kwh
        charge_time_min = (charge_kwh + consumption_kwh) / SimMetaData.charge_rate_kw * 60
        if not ChargingAlgoParams.infinite_chargers:
            self.lat = charger_lat
            self.lon = charger_lon
        self.state = CarState.CHARGING.value
        self.state_start_time = self.env.now
        self.soc = self.soc - consumption_kwh / SimMetaData.pack_size_kwh
        charger.update_occupancy(increase=True)
        self.n_of_charging_stops += 1
        self.total_drive_to_charge_time += drive_time_min
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} starting to charge at charger {charger_idx} at time {self.env.now}")
        try:
            yield self.env.timeout(charge_time_min)
        except:
            return 0

        charger.update_occupancy(increase=False)
        self.state = CarState.IDLE.value
        self.state_start_time = self.env.now

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
    try:
        car.prev_charging_process = env.process(car.run_charge(1, 1))
    except simpy.Interrupt:
        print("interrupted")
    trip = Trip(env, 0, 1, TripState.WAITING.value)
    env.process(car.run_trip(trip))
    env.run()
