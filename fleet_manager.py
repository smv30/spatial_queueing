import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgo
from spatial_queueing.real_life_data_input import sampleDF
from utils import calc_dist_between_two_points
from data_logging import DataLogging
from dateutil import parser


# FleetManager run the functions every minute
class FleetManager:
    def __init__(self,
                 arrival_rate_pmin,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time_min,
                 list_chargers,
                 matching_algo,
                 d):
        self.arrival_rate_pmin = arrival_rate_pmin
        self.env = env
        self.n_arrivals = 0
        self.car_tracker = car_tracker
        self.n_cars = n_cars
        self.renege_time_min = renege_time_min
        self.list_chargers = list_chargers
        self.data_logging = DataLogging()
        self.list_trips = []
        self.matching_algo = matching_algo
        self.df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        self.d = d

    def match_trips(self):
        time_to_go_for_data_logging = 0
        # add a counter to record the current number (index) of the trip
        counter = 0
        while True:    # everything inside runs every arrival
            df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
            # inter_arrival_time should be changed to the subtraction of the current trip and the past trip
            # inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / self.arrival_rate_pmin)
            if counter == 0:
                inter_arrival_time_min = 0
            elif counter == len(sampleDF):
                break
            else:
                # TO-DO: rename sampleDF to trip_data
                previous_pickup_datetime = parser.parse(sampleDF["pickup_datetime"].iloc[counter - 1])
                current_pickup_datetime = parser.parse(sampleDF["pickup_datetime"].iloc[counter])
                inter_arrival_time_sec = current_pickup_datetime - previous_pickup_datetime
                inter_arrival_time_min = int(inter_arrival_time_sec.total_seconds() / 60) + 1
            if SimMetaData.save_results:
                if time_to_go_for_data_logging <= 0:
                    list_soc = [self.car_tracker[car].soc for car in range(min(self.n_cars, 20))]
                    n_cars_idle = sum(df_car_tracker["state"] == CarState.IDLE.value)
                    n_cars_charging = sum(df_car_tracker["state"] == CarState.CHARGING.value)
                    n_cars_driving_to_charger = sum(df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
                    n_cars_driving_without_passenger = sum(
                        df_car_tracker["state"] == CarState.DRIVING_WITHOUT_PASSENGER.value
                    )
                    n_cars_driving_with_passenger = sum(
                        df_car_tracker["state"] == CarState.DRIVING_WITH_PASSENGER.value
                    )
                    n_cars_waiting_for_charger = sum(
                        df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value
                    )
                    avg_soc = np.mean(df_car_tracker["soc"])
                    stdev_soc = np.std(df_car_tracker["soc"])
                    self.data_logging.update_data(curr_list_soc=list_soc,
                                                  n_cars_idle=n_cars_idle,
                                                  n_cars_charging=n_cars_charging,
                                                  n_cars_driving_to_charger=n_cars_driving_to_charger,
                                                  n_cars_driving_without_passenger=n_cars_driving_without_passenger,
                                                  n_cars_driving_with_passenger=n_cars_driving_with_passenger,
                                                  n_cars_waiting_for_charger=n_cars_waiting_for_charger,
                                                  time_of_logging=self.env.now,
                                                  avg_soc=avg_soc,
                                                  stdev_soc=stdev_soc
                                                  )
                    time_to_go_for_data_logging = SimMetaData.freq_of_data_logging_min - inter_arrival_time_min
                else:
                    time_to_go_for_data_logging = time_to_go_for_data_logging - inter_arrival_time_min
            if ChargingAlgoParams.send_all_idle_cars_to_charge:
                list_cars = df_car_tracker[df_car_tracker["state"] == CarState.IDLE.value]["id"]
                for car_id in list_cars:
                    car = self.car_tracker[car_id]
                    if car.soc <= 0.95:
                        list_dist_to_charger = calc_dist_between_two_points(start_lat=car.lat,
                                                                            start_lon=car.lon,
                                                                            end_lat=self.df_list_charger["lat"],
                                                                            end_lon=self.df_list_charger["lon"]
                                                                            )
                        closest_charger_idx = np.argmin(list_dist_to_charger)
                        car.prev_charging_process = self.env.process(car.drive_to_charger(1, closest_charger_idx))
            curr_time_min = self.env.now
            # change the start & end latitude & longitude
            pickup_datetime = parser.parse(sampleDF["pickup_datetime"].iloc[counter])
            dropoff_datetime = parser.parse(sampleDF["dropoff_datetime"].iloc[counter])
            trip_time_sec = dropoff_datetime - pickup_datetime
            trip_time_min = int(trip_time_sec.total_seconds() / 60)
            trip = Trip(env=self.env,
                        arrival_time_min=curr_time_min,
                        trip_id=self.n_arrivals,
                        state=TripState.WAITING.value,
                        random=False,
                        start_lon=sampleDF["pickup_longitude"].iloc[counter],
                        start_lat=sampleDF["pickup_latitude"].iloc[counter],
                        end_lon=sampleDF["dropoff_longitude"].iloc[counter],
                        end_lat=sampleDF["dropoff_latitude"].iloc[counter],
                        trip_time_min=trip_time_min)
            self.list_trips.append(trip)

            if self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle(trip, df_car_tracker)
            elif self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle_or_charging(trip, df_car_tracker)
            else:
                raise ValueError(f"Matching algorithm {self.matching_algo} does not exist")

            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                service_time_min = pickup_time_min + trip.calc_trip_time()
                end_soc, closest_supercharger_idx = self.charge_just_after_trip_end(matched_car, service_time_min)
                self.env.process(matched_car.run_trip(trip, end_soc, closest_supercharger_idx))
            self.env.process(trip.update_trip_state(renege_time_min=self.renege_time_min))

            yield self.env.timeout(inter_arrival_time_min)
            self.n_arrivals = self.n_arrivals + 1
            counter += 1

    def power_of_d_closest_idle(self, trip, df_car_tracker):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)

        df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
                                             start_lon=df_car_tracker["lon"],
                                             end_lat=trip.start_lat,
                                             end_lon=trip.end_lon)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.trip_time_min
        df_car_tracker["delta_soc"] = (
                                              (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
                                              * SimMetaData.avg_vel_mph
                                              * SimMetaData.consumption_kwhpmi
                                      ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = [min(calc_dist_between_two_points(
            start_lat=self.car_tracker[car].lat,
            start_lon=self.car_tracker[car].lon,
            end_lat=df_list_charger["lat"],
            end_lon=df_list_charger["lon"],
        )
        ) * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh for car in range(self.n_cars)
                                             ]
        enough_soc_mask = (
                df_car_tracker["soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )

        available_cars = df_car_tracker[idle_cars_mask & enough_soc_mask].sort_values("pickup_time_min")
        if len(available_cars) > 0:
            available_cars_of_interest = available_cars.iloc[0:self.d]
            car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
            return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def power_of_d_closest_idle_or_charging(self, trip, df_car_tracker):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
        driving_to_charger_mask = (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
        charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
        waiting_for_charger_mask = (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value)
        df_car_tracker["curr_soc"] = np.copy(df_car_tracker["soc"])
        df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "curr_soc"] = (
                (df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "state_start_time"]
                 - self.env.now)
                / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
                + df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "soc"]
        )
        df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "curr_soc"] = (
                (self.env.now - df_car_tracker.loc[
                    df_car_tracker["state"] == CarState.CHARGING.value, "state_start_time"])
                / 60 * SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh
                + df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "soc"]
        )
        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
                                             start_lon=df_car_tracker["lon"],
                                             end_lat=trip.start_lat,
                                             end_lon=trip.start_lon)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.calc_trip_time()
        df_car_tracker["delta_soc"] = (
                                              (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
                                              * SimMetaData.avg_vel_mph
                                              * SimMetaData.consumption_kwhpmi
                                      ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = 0.1
        enough_soc_mask = (
                df_car_tracker["curr_soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )

        available_cars = df_car_tracker[
            (idle_cars_mask | driving_to_charger_mask | charging_mask | waiting_for_charger_mask) & enough_soc_mask
            ].sort_values("pickup_time_min")
        if len(available_cars) > 0:
            available_cars_of_interest = available_cars.iloc[0:self.d]
            car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
            return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def charge_just_after_trip_end(self, car, service_time_min):
        trip_end_soc = (
                car.soc
                - service_time_min / 60
                * SimMetaData.avg_vel_mph
                * SimMetaData.consumption_kwhpmi
                / SimMetaData.pack_size_kwh
        )
        end_soc = ChargingAlgoParams.higher_soc_threshold
        dist_to_superchargers = calc_dist_between_two_points(
            start_lat=car.lat,
            start_lon=car.lon,
            end_lat=self.df_list_charger["lat"],
            end_lon=self.df_list_charger["lon"],
        )
        closest_supercharger_idx = np.argmin(dist_to_superchargers)
        if ChargingAlgoParams.send_all_idle_cars_to_charge:
            return 1, closest_supercharger_idx
        elif trip_end_soc < ChargingAlgoParams.lower_soc_threshold:
            return end_soc, closest_supercharger_idx
        else:
            return None, None
