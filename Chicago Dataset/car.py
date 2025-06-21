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
        self.prev_relocating_process = None
        self.n_of_charging_stops = 0
        self.total_drive_to_charge_time = 0
        self.charging_at_idx = None
        self.relocating_to_lat = None
        self.relocating_to_lon = None
        self.end_soc_post_charging = None
        self.list_drive_to_charger_time_min = []
        self.list_n_available_chargers = []
        self.list_n_available_posts = []
        self.list_n_cars_driving_to_charger = []
        self.fake_charging = False
        self.is_it_relocating_to_charger = False

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
            "charging_at_lon": charging_at_lon,
            "relocating_to_lat": self.relocating_to_lat,
            "relocating_to_lon": self.relocating_to_lon,
            "fake_charging_bool": self.fake_charging
        }

    def run_trip(self, trip, dist_correction_factor, dist_func, bool_relocate=False, relocate_lat=None, relocate_lon=None):
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
        elif self.state == CarState.RELOCATING.value:
            if self.is_it_relocating_to_charger is True:
                self.interrupt_relocation(
                    dist_correction_factor=dist_correction_factor,
                    dist_func=dist_func,
                    charger_idx=self.charging_at_idx
                )
            else:
                self.interrupt_relocation(
                    dist_correction_factor=dist_correction_factor,
                    dist_func=dist_func,
                        relocating_lat=self.relocating_to_lat,
                        relocating_lon=self.relocating_to_lon)
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
            raise ValueError(f"Curr SoC is {self.soc}. SOC cannot be less than 0. \n" +
                             f"Curr simulation time is {self.env.now} mins. \n" +
                              f"The car was interrupted from state {prev_state}. The curr charging session is fake or not: {self.fake_charging} \n" +
                              f"Total SoC consumed: {delta_soc}. \n" +
                               f"It took {pickup_time_min} mins to pickup the customer. \n" +
                               f"Also, it took {trip_time_min} mins to finish the trip.")
        if bool_relocate is True and self.soc >= 0.2:
            self.prev_relocating_process = self.env.process(self.relocate(dist_correction_factor=dist_correction_factor,
                                                                          dist_func=dist_func,
                                                                          end_lat=relocate_lat,
                                                                          end_lon=relocate_lon
                                                                          ))

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
            if self.fake_charging is False:
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
    def drive_to_charger(self, end_soc, charger_idx, dist_correction_factor, dist_func, list_available_chargers, fake_charging_bool=False):
        # Change the car state to DRIVING_TO_CHARGER
        if fake_charging_bool is True:
            self.fake_charging = True
        else:
            self.fake_charging = False
        self.charging_at_idx = charger_idx
        self.end_soc_post_charging = end_soc
        charger = self.list_chargers[charger_idx]
        charger_lat = charger.lat
        charger_lon = charger.lon
        charger.n_cars_driving_to_charger += 1
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is not idle to be sent to charge. It is currently in state {self.state}")

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
        if self.soc < 0 or self.soc > 1:
            raise ValueError(f"Curr SOC is {self.soc}. Took {consumption_kwh / SimMetaData.pack_size_kwh} SOC to reach the charger. SOC must be between 0 and 1. Fake charging is {self.fake_charging}.")

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
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} starting to charge at charger {charger_idx} at time {self.env.now}")
        if self.soc <= end_soc:
            charger = self.list_chargers[charger_idx]
            charge_kwh = (end_soc - self.soc) * SimMetaData.pack_size_kwh
            charge_time_min = charge_kwh / SimMetaData.charge_rate_kw * 60
            try:
                yield self.env.timeout(charge_time_min)
            except:
                return 0
            if self.fake_charging is False:
                self.soc = end_soc
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} finished charging at charger {charger_idx}, gained {charge_kwh} kwh in {charge_time_min} mins")
        if self.soc < 0 or self.soc > 1:
            raise ValueError(f"Curr SOC is {self.soc}. SOC must be between 0 and 1. Fake charging is {self.fake_charging}.")

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

    def relocate(self, dist_correction_factor, dist_func, end_lat, end_lon):
        # Change the car state to DRIVING_TO_CHARGER
        if self.state != CarState.IDLE.value:
            raise ValueError(f"Car {self.id} is not idle to relocate")

        dist_to_destination = calc_dist_between_two_points(start_lat=self.lat,
                                                           start_lon=self.lon,
                                                           end_lat=end_lat,
                                                           end_lon=end_lon,
                                                           dist_correction_factor=dist_correction_factor,
                                                           dist_func=dist_func)

        # Add a timeout equal to the drive time to the charger
        drive_time_min = dist_to_destination / SimMetaData.avg_vel_mph * 60
        self.state = CarState.RELOCATING.value
        self.state_start_time = self.env.now
        self.relocating_to_lat = end_lat
        self.relocating_to_lon = end_lon
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} relocating to location {[end_lat, end_lon]} at time {self.env.now}. Driving a distance of {dist_to_destination} miles")
        try:
            yield self.env.timeout(drive_time_min)
        except:
            return 0

        # Reduce SOC by the amount spent while driving to the charger
        consumption_kwh = dist_to_destination * SimMetaData.consumption_kwhpmi
        self.soc = self.soc - consumption_kwh / SimMetaData.pack_size_kwh

        # Set the car location equal to the final location
        self.lat = end_lat
        self.lon = end_lon

        # Change the car state to IDLE
        self.state = CarState.IDLE.value
        self.state_start_time = self.env.now
        self.relocating_to_lat = None
        self.relocating_to_lon = None
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} is idling after relocating to {[end_lat, end_lon]} at time {self.env.now}")

    def interrupt_relocation(self, dist_correction_factor, dist_func, relocating_lat=None, relocating_lon=None, charger_idx=None):
        if self.state != CarState.RELOCATING.value:
            raise ValueError("The EV is not relocating to interrupt relocation")
        # interrupt the process
        if not SimMetaData.quiet_sim:
            print(f"Car {self.id} was interrupted while relocating at time {self.env.now}")
        time_spent_in_this_state_min = self.env.now - self.state_start_time
        #  update soc
        #  update car state to idle
        if self.is_it_relocating_to_charger is True:
            self.prev_charging_process.interrupt()
            charger = self.list_chargers[charger_idx]
            relocating_lat = charger.lat
            relocating_lon = charger.lon
            charger.n_cars_driving_to_charger -= 1
            self.charging_at_idx = None
        else:
            self.prev_relocating_process.interrupt()
        self.is_it_relocating_to_charger = False
        consumption_kwh = (
                time_spent_in_this_state_min * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / 60
        )
        delta_soc = consumption_kwh / SimMetaData.pack_size_kwh
        self.soc = self.soc - delta_soc
        driving_distance_mi = time_spent_in_this_state_min * SimMetaData.avg_vel_mph / 60
        relocation_dist_mi = calc_dist_between_two_points(start_lat=self.lat,
                                                          start_lon=self.lon,
                                                          end_lat=relocating_lat,
                                                          end_lon=relocating_lon,
                                                          dist_correction_factor=dist_correction_factor,
                                                          dist_func=dist_func)
        self.lat = self.lat + (relocating_lat - self.lat) * driving_distance_mi / relocation_dist_mi
        self.lon = self.lon + (relocating_lon - self.lon) * driving_distance_mi / relocation_dist_mi
        self.state = CarState.IDLE.value
        self.state_start_time = self.env.now
        self.relocating_to_lat = None
        self.relocating_to_lon = None

if __name__ == "__main__":
    env = simpy.Environment()
    n_chargers = SimMetaData.n_charger_loc
    n_posts = SimMetaData.n_posts
    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts,
                               env=env,
                               df_arrival_sequence=None)
        list_chargers.append(charger)

    car = Car(car_id=0,
              lat=0,
              lon=1,
              env=env,
              list_chargers=list_chargers,
              df_arrival_sequence=None,  # Update this if you want to test this file separately
              )
    try:
        car.prev_charging_process = env.process(car.car_charging(0, 1))
    except simpy.Interrupt:
        print("interrupted")
    trip = Trip(env, 0, 1, TripState.WAITING.value)
    env.process(car.run_trip(trip, 1, DistFunc.MANHATTAN.value))
    env.run()
