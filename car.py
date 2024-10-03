from sim_metadata import SimMetaData, CarState, TripState, ChargingAlgoParams, ChargerState, DatasetParams, \
    Initialize, DistFunc
from chargers import SuperCharger
from arrivals import Trip
from utils import calc_dist_between_two_points, sample_unif_points_on_sphere
import simpy


class Car:
    def __init__(self,
                 car_id,
                 env,
                 list_chargers,
                 df_arrival_sequence,
                 initialize_car=Initialize.RANDOM_PICKUP.value,
                 lat=None,
                 lon=None,
                 ):
        if initialize_car == Initialize.RANDOM_UNIFORM.value:
            self.lat, self.lon = sample_unif_points_on_sphere(lon_min=DatasetParams.longitude_range_min,
                                                              lon_max=DatasetParams.longitude_range_max,
                                                              lat_min=DatasetParams.latitude_range_min,
                                                              lat_max=DatasetParams.latitude_range_max)
        elif initialize_car == Initialize.RANDOM_PICKUP.value:
            sample_trip = df_arrival_sequence.sample(1)
            self.lon = sample_trip["pickup_longitude"].values[0]
            self.lat = sample_trip["pickup_latitude"].values[0]
        elif initialize_car == Initialize.EQUAL_TO_INPUT.value:
            self.lat = lat
            self.lon = lon
        else:
            raise ValueError("No such command for initialization cars exists")
        soc = SimMetaData.random_seed_gen.uniform(0.7, 0.9)
        state = CarState.IDLE.value
        self.id = car_id
        self.state = state
        self.soc = soc
        self.env = env
        self.list_chargers = list_chargers
        self.state_start_time = 0
        self.prev_charging_process = None
        self.n_of_charging_stops = 0
        self.total_drive_to_charge_time = 0
        self.charging_at_idx = None
        self.end_soc_post_charging = None
        self.list_drive_to_charger_time_min = []
        self.list_n_available_chargers = []
        self.list_n_available_posts = []
        self.list_n_cars_driving_to_charger = []

    def to_dict(self):
        if self.charging_at_idx is not None:
            charging_at_lat = self.list_chargers[self.charging_at_idx].lat
            charging_at_lon = self.list_chargers[self.charging_at_idx].lon
        else:
            charging_at_lat = None
            charging_at_lon = None
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "soc": self.soc,
            "state": self.state,
            "state_start_time": self.state_start_time,
            "charging_at_lat": charging_at_lat,
            "charging_at_lon": charging_at_lon
        }

    def run_trip(self, trip, dist_correction_factor, dist_func):
        # If the car is driving to charger or charging or waiting for charger, interrupt that process
        prev_state = self.state
        if self.state in (
                CarState.DRIVING_TO_CHARGER.value, CarState.CHARGING.value, CarState.WAITING_FOR_CHARGER.value):
            self.interrupt_charging(
                charger_idx=self.charging_at_idx,
                end_soc=self.end_soc_post_charging,
                dist_correction_factor=dist_correction_factor,
                dist_func=dist_func
            )
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is currently finishing a trip and cannot be matched")
        trip.state = TripState.MATCHED.value
        pickup_dist_mi = calc_dist_between_two_points(start_lat=self.lat,
                                                      start_lon=self.lon,
                                                      end_lat=trip.start_lat,
                                                      end_lon=trip.start_lon,
                                                      dist_correction_factor=dist_correction_factor,
                                                      dist_func=dist_func)
        pickup_time_min = pickup_dist_mi / SimMetaData.avg_vel_mph * 60
        trip.pickup_time_min = pickup_time_min
        self.state = CarState.DRIVING_WITHOUT_PASSENGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} picking up Trip {trip.trip_id} at time {self.env.now}")
        yield self.env.timeout(pickup_time_min)

        trip_time_min = trip.trip_time_min
        trip_dist_mi = trip.trip_distance_mi
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
            print(self.env.now)
            print(prev_state)
            raise ValueError("SOC cannot be less than 0")

    def interrupt_charging(self, charger_idx, end_soc, dist_correction_factor, dist_func):
        charger = self.list_chargers[charger_idx]
        # interrupt the process
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} was interrupted while charging at time {self.env.now}")
        time_spent_in_this_state_min = self.env.now - self.state_start_time
        # if the car is driving to the charger:
        #     update soc
        #     update car state to idle
        if self.state == CarState.DRIVING_TO_CHARGER.value:
            self.prev_charging_process.interrupt()
            consumption_kwh = (
                    time_spent_in_this_state_min * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / 60
            )
            delta_soc = consumption_kwh / SimMetaData.pack_size_kwh
            self.soc = self.soc - delta_soc
            driving_distance_mi = time_spent_in_this_state_min * SimMetaData.avg_vel_mph / 60
            total_distance_mi = max(
                0.01,
                calc_dist_between_two_points(
                    start_lat=self.lat,
                    start_lon=self.lon,
                    end_lat=charger.lat,
                    end_lon=charger.lon,
                    dist_correction_factor=dist_correction_factor,
                    dist_func=dist_func
                )
            )
            self.lat = self.lat + (charger.lat - self.lat) * driving_distance_mi / total_distance_mi
            self.lon = self.lon + (charger.lon - self.lon) * driving_distance_mi / total_distance_mi
            self.state = CarState.IDLE.value
            self.state_start_time = self.env.now
            self.charging_at_idx = None
            charger.n_cars_driving_to_charger -= 1
        # else if the car is waiting at the charger:
        #     update car state to idle
        #     use charger object with charger_idx to get queueing_list, then remove the car from the list
        elif self.state == CarState.WAITING_FOR_CHARGER.value:
            self.state = CarState.IDLE.value
            self.state_start_time = self.env.now
            self.charging_at_idx = None
            if [self.id, end_soc] in charger.queue_list:
                charger.queue_list.remove([self.id, end_soc])
            else:
                raise ValueError(f"Car {self.id} is not waiting at charger {charger_idx} to be removed")
        # else if the car is currently charging:
        #       update soc
        #       decrease occupancy by one
        #       update car state to idle
        #       update charger state to available
        #       call queueing_at_charger
        elif self.state == CarState.CHARGING.value:
            self.prev_charging_process.interrupt()
            charging_kwh = time_spent_in_this_state_min * SimMetaData.charge_rate_kw / 60
            delta_soc = charging_kwh / SimMetaData.pack_size_kwh
            self.soc = self.soc + delta_soc
            charger.occupancy -= 1
            self.state = CarState.IDLE.value
            self.state_start_time = self.env.now
            charger.state = ChargerState.AVAILABLE.value
            charger.queueing_at_charger(None, None)
            self.charging_at_idx = None
        else:
            raise ValueError("Charging process is not going on to be interrupted")

    # Logic: drive_to_charger() call queueing_at_charger()
    #        -> queueing_at_charger() call car_charging()
    #        -> car_charging() call queueing_at_charger()
    def drive_to_charger(self, end_soc, charger_idx, dist_correction_factor, dist_func, list_available_chargers):
        # Change the car state to DRIVING_TO_CHARGER
        self.charging_at_idx = charger_idx
        self.end_soc_post_charging = end_soc
        charger = self.list_chargers[charger_idx]
        charger_lat = charger.lat
        charger_lon = charger.lon
        charger.n_cars_driving_to_charger += 1
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is not idle to be sent to charge")

        if not ChargingAlgoParams.infinite_chargers:
            dist_to_charger_mi = calc_dist_between_two_points(start_lat=self.lat,
                                                              start_lon=self.lon,
                                                              end_lat=charger_lat,
                                                              end_lon=charger_lon,
                                                              dist_correction_factor=dist_correction_factor,
                                                              dist_func=dist_func)
        else:
            dist_to_charger_mi = 0.001

        # Add a timeout equal to the drive time to the charger
        drive_time_min = dist_to_charger_mi / SimMetaData.avg_vel_mph * 60
        self.state = CarState.DRIVING_TO_CHARGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} driving to charger {charger_idx} at time {self.env.now} with initial soc {self.soc}")
        try:
            yield self.env.timeout(drive_time_min)
        except:
            return 0

        # Reduce SOC by the amount spent while driving to the charger
        consumption_kwh = dist_to_charger_mi * SimMetaData.consumption_kwhpmi
        self.soc = self.soc - consumption_kwh / SimMetaData.pack_size_kwh

        # Set the car location equal to the charger location
        if not ChargingAlgoParams.infinite_chargers:
            self.lat = charger_lat
            self.lon = charger_lon

        self.n_of_charging_stops += 1
        self.total_drive_to_charge_time += drive_time_min

        self.list_drive_to_charger_time_min.append(drive_time_min)
        self.list_n_available_chargers.append(len(list_available_chargers))
        self.list_n_available_posts.append(sum(list_available_chargers["n_available_posts"]))
        self.list_n_cars_driving_to_charger.append(sum(list_available_chargers["n_cars_driving_to_charger"]))

        # Change the car state to WAITING_FOR_CHARGER
        self.state = CarState.WAITING_FOR_CHARGER.value
        self.state_start_time = self.env.now
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} starting to wait at charger {charger_idx} at time {self.env.now}")

        # Call queueing_at_charger function
        charger.queueing_at_charger(self.id, end_soc)
        charger.n_cars_driving_to_charger -= 1

    def car_charging(self, charger_idx, end_soc):
        self.state = CarState.CHARGING.value
        self.state_start_time = self.env.now

        charger = self.list_chargers[charger_idx]
        charge_kwh = (end_soc - self.soc) * SimMetaData.pack_size_kwh
        charge_time_min = charge_kwh / SimMetaData.charge_rate_kw * 60
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} starting to charge at charger {charger_idx} at time {self.env.now}")
        try:
            yield self.env.timeout(charge_time_min)
        except:
            return 0

        self.soc = end_soc
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} finished charging at charger {charger_idx}, gained {charge_kwh} kwh in {charge_time_min} mins")
        if self.soc < 0 or self.soc > 1:
            raise ValueError("SOC must be between 0 and 1")

        self.state = CarState.IDLE.value
        self.state_start_time = self.env.now

        charger.occupancy -= 1
        charger.state = ChargerState.AVAILABLE.value
        self.charging_at_idx = None
        self.end_soc_post_charging = None

        # Call queueing_at_charger function so that cars waiting in the queue starts charging
        # Call queueing_at_charger twice for each car (every time it arrives & leaves)
        #       first time: add the car to the list and see if it needs to wait
        #       second time: let the cars behind it in the list to be charged
        charger.queueing_at_charger(None, None)


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
              env=env,
              list_chargers=list_chargers,
              df_arrival_sequence=None  # Update this if you want to test this file separately
              )
    try:
        car.prev_charging_process = env.process(car.run_charge(1, 1))
    except simpy.Interrupt:
        print("interrupted")
    trip = Trip(env, 0, 1, TripState.WAITING.value)
    env.process(car.run_trip(trip, 1, DistFunc.MANHATTAN.value))
    env.run()
