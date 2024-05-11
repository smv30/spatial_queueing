import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgo, ChargerState, Dataset, \
    AdaptivePowerOfDParams, ChargingAlgo, AvailableCarsForMatching, PickupThresholdType, PickupThresholdMatchingParams
from utils import calc_dist_between_two_points
from data_logging import DataLogging


# FleetManager run the functions every minute
class FleetManager:
    def __init__(self,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time_min,
                 list_chargers,
                 trip_data,
                 matching_algo,
                 available_cars_for_matching,
                 charging_algo,
                 dist_correction_factor,
                 pickup_threshold_type,
                 dist_func,
                 d):
        self.env = env
        self.n_arrivals = 0
        self.car_tracker = car_tracker
        self.n_cars = n_cars
        self.renege_time_min = renege_time_min
        self.list_chargers = list_chargers
        self.data_logging = DataLogging()
        self.list_trips = []
        self.trip_data = trip_data
        self.matching_algo = matching_algo
        self.available_cars_for_matching = available_cars_for_matching
        self.charging_algo = charging_algo
        self.dist_correction_factor = dist_correction_factor
        self.df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        self.d = d
        self.dist_func = dist_func
        self.pickup_threshold_type = pickup_threshold_type

    def match_trips(self):
        time_to_go_for_data_logging = 0
        # add a counter to record the current number (index) of the trip
        counter = 0
        n_served_trips_before_updating_d = 0
        n_cars_idle_full_soc_average = 0
        charge_threshold = 0.95
        while True:  # everything inside runs every arrival
            df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
            self.df_list_charger = pd.DataFrame([
                self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
            ])
            if counter == 0:
                inter_arrival_time_min = 0
            elif counter == len(self.trip_data):
                break
            else:
                previous_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter - 1]
                current_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter]
                inter_arrival_time_datetime = current_arrival_datetime - previous_arrival_datetime
                inter_arrival_time_min = inter_arrival_time_datetime.total_seconds() / 60.0
            if (self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS.value or
                    self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value):
                list_cars = df_car_tracker[
                    df_car_tracker["state"] == CarState.IDLE.value].sort_values(by="soc", ascending=True)["id"]
                available_charger_mask = (self.df_list_charger["n_available_posts"]
                                          - ChargingAlgoParams.n_cars_driving_to_charger_discounter
                                          * self.df_list_charger["n_cars_driving_to_charger"] > 0
                                          )
                list_available_chargers = self.df_list_charger[available_charger_mask]
                for car_id in list_cars:
                    list_posts_available = np.copy(self.df_list_charger["n_available_posts"])
                    car = self.car_tracker[car_id]
                    if car.soc <= charge_threshold:
                        closest_available_charger_idx = self.closest_available_charger(car, list_available_chargers)
                        if closest_available_charger_idx is not None:
                            car.prev_charging_process = self.env.process(
                                car.drive_to_charger(
                                    end_soc=1,
                                    charger_idx=closest_available_charger_idx,
                                    dist_correction_factor=self.dist_correction_factor,
                                    dist_func=self.dist_func
                                ))
                            list_posts_available[closest_available_charger_idx] -= 1
                            if list_posts_available[closest_available_charger_idx] <= 0:
                                list_available_chargers = list_available_chargers[
                                    list_available_chargers['idx'] != closest_available_charger_idx]
            else:
                raise ValueError(f"Charging algorithm {self.charging_algo} does not exist")
            curr_time_min = self.env.now
            pickup_datetime = self.trip_data["pickup_datetime"].iloc[counter]
            dropoff_datetime = self.trip_data["dropoff_datetime"].iloc[counter]
            trip_time_sec = dropoff_datetime - pickup_datetime
            trip_time_min = int(trip_time_sec.total_seconds() / 60.0)
            trip = Trip(env=self.env,
                        arrival_time_min=curr_time_min,
                        trip_id=self.n_arrivals,
                        state=TripState.WAITING.value,
                        trip_distance_mi=self.trip_data["trip_distance"].iloc[counter],
                        start_lon=self.trip_data["pickup_longitude"].iloc[counter],
                        start_lat=self.trip_data["pickup_latitude"].iloc[counter],
                        end_lon=self.trip_data["dropoff_longitude"].iloc[counter],
                        end_lat=self.trip_data["dropoff_latitude"].iloc[counter],
                        trip_time_min=trip_time_min)
            self.list_trips.append(trip)

            car_id, pickup_time_min, n_cars_available = self.matching_algorithms(trip=trip, df_car_tracker=df_car_tracker)

            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                self.env.process(matched_car.run_trip(trip, self.dist_correction_factor, self.dist_func))
                n_served_trips_before_updating_d += 1
            self.env.process(trip.update_trip_state(renege_time_min=self.renege_time_min))

            yield self.env.timeout(inter_arrival_time_min)
            self.n_arrivals = self.n_arrivals + 1
            counter += 1

            if AdaptivePowerOfDParams.adaptive_d is True:
                n_cars_idle_full_soc = len(df_car_tracker[(df_car_tracker["state"] == CarState.IDLE.value) &
                                                          (df_car_tracker["soc"] >= 0.95)])
                n_cars_idle_full_soc_average = (
                        (n_cars_idle_full_soc_average * (counter % AdaptivePowerOfDParams.n_trips_before_updating_d)
                         + n_cars_idle_full_soc) / (counter % AdaptivePowerOfDParams.n_trips_before_updating_d + 1)
                )
                if counter % AdaptivePowerOfDParams.n_trips_before_updating_d == 0:
                    if n_served_trips_before_updating_d < 0.95 * AdaptivePowerOfDParams.n_trips_before_updating_d:
                        if (
                                n_cars_idle_full_soc_average >
                                AdaptivePowerOfDParams.threshold_percent_of_cars_idling * self.n_cars
                        ):
                            self.d += 1
                        elif n_cars_idle_full_soc_average == 0:
                            if self.d > 1:
                                self.d -= 1
                    n_served_trips_before_updating_d = 0
                    n_cars_idle_full_soc_average = 0

            if self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value:
                curr_time_in_the_day_min = curr_time_min % (24 * 60)
                if (
                        (ChargingAlgoParams.start_of_the_night * 60)
                        <= curr_time_in_the_day_min
                        <= (ChargingAlgoParams.end_of_the_night * 60)
                ):
                    charge_threshold = 0.95
                else:
                    charge_threshold = 0.2

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
                    d = self.d
                    self.data_logging.update_data(curr_list_soc=list_soc,
                                                  n_cars_idle=n_cars_idle,
                                                  n_cars_charging=n_cars_charging,
                                                  n_cars_driving_to_charger=n_cars_driving_to_charger,
                                                  n_cars_driving_without_passenger=n_cars_driving_without_passenger,
                                                  n_cars_driving_with_passenger=n_cars_driving_with_passenger,
                                                  n_cars_waiting_for_charger=n_cars_waiting_for_charger,
                                                  time_of_logging=self.env.now,
                                                  avg_soc=avg_soc,
                                                  stdev_soc=stdev_soc,
                                                  d=d,
                                                  charge_threshold=charge_threshold,
                                                  n_cars_available=n_cars_available,
                                                  pickup_time_min=pickup_time_min
                                                  )
                    time_to_go_for_data_logging = SimMetaData.freq_of_data_logging_min - inter_arrival_time_min
                else:
                    time_to_go_for_data_logging = time_to_go_for_data_logging - inter_arrival_time_min

    def matching_algorithms(self, trip, df_car_tracker, drive_to_charger_update=True):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
        charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
        waiting_for_charger_mask = (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value)
        driving_to_charger_mask = (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
        df_car_tracker["curr_soc"] = np.copy(df_car_tracker["soc"])
        df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "curr_soc"] = (
                (self.env.now - df_car_tracker.loc[
                    df_car_tracker["state"] == CarState.CHARGING.value, "state_start_time"])
                / 60 * SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh
                + df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "soc"]
        )
        df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "curr_soc"] = \
            (
                    df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "soc"]
                    - (self.env.now
                       - df_car_tracker.loc[
                           df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "state_start_time"]
                       )
                    / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
            )
        df_car_tracker["curr_lat"] = np.copy(df_car_tracker["lat"])
        df_car_tracker["curr_lon"] = np.copy(df_car_tracker["lon"])
        if (drive_to_charger_update is True
                and self.available_cars_for_matching == AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value
                and sum(df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value) > 1
        ):
            filtered_car_tracker = df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value]
            dist_to_charger = calc_dist_between_two_points(
                start_lat=filtered_car_tracker["lat"],
                start_lon=filtered_car_tracker["lon"],
                end_lat=filtered_car_tracker["charging_at_lat"],
                end_lon=filtered_car_tracker["charging_at_lon"],
                dist_correction_factor=self.dist_correction_factor,
                dist_func=self.dist_func
            )
            filtered_car_tracker = filtered_car_tracker.loc[dist_to_charger > 0]
            dist_traveled = (self.env.now - filtered_car_tracker["state_start_time"]) / 60 * SimMetaData.avg_vel_mph

            filtered_car_tracker["curr_lat"] = (filtered_car_tracker["lat"] +
                                                (filtered_car_tracker["charging_at_lat"] - filtered_car_tracker["lat"])
                                                * dist_traveled / dist_to_charger[dist_to_charger > 0]
                                                )
            filtered_car_tracker["curr_lon"] = (filtered_car_tracker["lon"] +
                                                (filtered_car_tracker["charging_at_lon"] - filtered_car_tracker["lon"])
                                                * dist_traveled / dist_to_charger[dist_to_charger > 0]
                                                )
        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
                                             start_lon=df_car_tracker["lon"],
                                             end_lat=trip.start_lat,
                                             end_lon=trip.start_lon,
                                             dist_correction_factor=self.dist_correction_factor,
                                             dist_func=self.dist_func)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_distance_mi = trip.trip_distance_mi
        df_car_tracker["delta_soc"] = ((df_car_tracker[
                                            "pickup_time_min"] / 60 * SimMetaData.avg_vel_mph + trip_distance_mi)
                                       * SimMetaData.consumption_kwhpmi
                                       ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = (min(calc_dist_between_two_points(
            start_lat=trip.end_lat,
            start_lon=trip.end_lon,
            end_lat=self.df_list_charger["lat"],
            end_lon=self.df_list_charger["lon"],
            dist_correction_factor=self.dist_correction_factor,
            dist_func=self.dist_func
        )) * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
                                             * ChargingAlgoParams.safety_factor_to_reach_closest_charger)
        enough_soc_mask = (
                df_car_tracker["curr_soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )
        n_available_cars = len(df_car_tracker[
            (idle_cars_mask | charging_mask | waiting_for_charger_mask) & enough_soc_mask
            ].sort_values(by=["pickup_time_min", "soc"], ascending=[True, False]))

        if self.matching_algo == MatchingAlgo.POWER_OF_D.value:
            if self.available_cars_for_matching == AvailableCarsForMatching.ONLY_IDLE.value:
                cars_of_interest = df_car_tracker[idle_cars_mask].sort_values(by=["pickup_time_min", "curr_soc"],
                                                                              ascending=[True, False])
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_CHARGING.value:
                cars_of_interest = df_car_tracker[
                    (idle_cars_mask | charging_mask | waiting_for_charger_mask)
                ].sort_values(by=["pickup_time_min", "curr_soc"], ascending=[True, False])
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value:
                cars_of_interest = df_car_tracker[
                    (idle_cars_mask | charging_mask | waiting_for_charger_mask | driving_to_charger_mask)
                ].sort_values(by=["pickup_time_min", "curr_soc"], ascending=[True, False])
            else:
                raise ValueError("Such an input for the available cars for matching is invalid")
            if len(cars_of_interest) == 0:
                return None, None, n_available_cars
            d_closest_cars = cars_of_interest.iloc[0:self.d]
            possible_car_to_dispatch = d_closest_cars.sort_values("curr_soc", ascending=False).iloc[0]
            if (possible_car_to_dispatch["curr_soc"]
                    - possible_car_to_dispatch["delta_soc"]
                    - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
            ):
                if self.pickup_threshold_type == PickupThresholdType.PERCENT_THRESHOLD.value:
                    if (
                            possible_car_to_dispatch["pickup_time_min"]
                            <= PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min
                    ):
                        return (int(possible_car_to_dispatch["id"]),
                                possible_car_to_dispatch["pickup_time_min"],
                                n_available_cars
                                )
                elif self.pickup_threshold_type == PickupThresholdType.CONSTANT_THRESHOLD.value:
                    if possible_car_to_dispatch["pickup_time_min"] <= PickupThresholdMatchingParams.threshold_min:
                        return (
                            int(possible_car_to_dispatch["id"]),
                            possible_car_to_dispatch["pickup_time_min"],
                            n_available_cars
                        )
                elif self.pickup_threshold_type == PickupThresholdType.BOTH_PERCENT_AND_CONSTANT.value:
                    if (
                            possible_car_to_dispatch["pickup_time_min"]
                            <= min(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return (
                            int(possible_car_to_dispatch["id"]),
                            possible_car_to_dispatch["pickup_time_min"],
                            n_available_cars
                        )
                elif self.pickup_threshold_type == PickupThresholdType.EITHER_PERCENT_OR_CONSTANT.value:
                    if (
                            possible_car_to_dispatch["pickup_time_min"]
                            <= max(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return (
                            int(possible_car_to_dispatch["id"]),
                            possible_car_to_dispatch["pickup_time_min"],
                            n_available_cars
                        )
                elif self.pickup_threshold_type == PickupThresholdType.NO_THRESHOLD.value:
                    return (
                        int(possible_car_to_dispatch["id"]),
                        possible_car_to_dispatch["pickup_time_min"],
                        n_available_cars
                    )
                else:
                    raise ValueError("No such thresholding scheme exists")
            return None, None, n_available_cars
        elif self.matching_algo == MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value:
            enough_soc_mask = (
                    df_car_tracker["curr_soc"]
                    - df_car_tracker["delta_soc"]
                    - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
            )
            if self.available_cars_for_matching == AvailableCarsForMatching.ONLY_IDLE.value:
                available_cars = df_car_tracker[
                    idle_cars_mask & enough_soc_mask
                    ].sort_values(by=["pickup_time_min", "soc"], ascending=[True, False])
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_CHARGING.value:
                available_cars = df_car_tracker[
                    (idle_cars_mask | charging_mask | waiting_for_charger_mask) & enough_soc_mask
                    ].sort_values(by=["pickup_time_min", "soc"], ascending=[True, False])
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value:
                available_cars = df_car_tracker[
                    (idle_cars_mask | charging_mask | waiting_for_charger_mask | driving_to_charger_mask)
                    & enough_soc_mask].sort_values(by=["pickup_time_min", "soc"], ascending=[True, False])
            else:
                raise ValueError("Such an input for the available cars for matching is invalid")
            if len(available_cars) > 0:
                car_to_dispatch = available_cars.iloc[0]
                if self.pickup_threshold_type == PickupThresholdType.PERCENT_THRESHOLD.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min
                    ):
                        return car_to_dispatch["id"], car_to_dispatch["pickup_time_min"], n_available_cars
                elif self.pickup_threshold_type == PickupThresholdType.CONSTANT_THRESHOLD.value:
                    if car_to_dispatch["pickup_time_min"] <= PickupThresholdMatchingParams.threshold_min:
                        return car_to_dispatch["id"], car_to_dispatch["pickup_time_min"], n_available_cars
                elif self.pickup_threshold_type == PickupThresholdType.BOTH_PERCENT_AND_CONSTANT.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= min(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return car_to_dispatch["id"], car_to_dispatch["pickup_time_min"], n_available_cars
                elif self.pickup_threshold_type == PickupThresholdType.EITHER_PERCENT_OR_CONSTANT.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= max(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return car_to_dispatch["id"], car_to_dispatch["pickup_time_min"], n_available_cars
                elif self.pickup_threshold_type == PickupThresholdType.NO_THRESHOLD.value:
                    return car_to_dispatch["id"], car_to_dispatch["pickup_time_min"], n_available_cars
                else:
                    raise ValueError("No such thresholding scheme exists")
            return None, None, n_available_cars
        else:
            raise ValueError(f"Matching algorithm {self.matching_algo} does not exist")

    def closest_available_charger(self, car, list_available_chargers):
        if len(list_available_chargers["lat"]) == 0:
            return None
        dist_to_superchargers = calc_dist_between_two_points(
            start_lat=car.lat,
            start_lon=car.lon,
            end_lat=list_available_chargers["lat"],
            end_lon=list_available_chargers["lon"],
            dist_correction_factor=self.dist_correction_factor,
            dist_func=self.dist_func
        )
        argmin_idx = np.argmin(dist_to_superchargers)
        closest_supercharger_idx = list_available_chargers["idx"].iloc[argmin_idx]
        return closest_supercharger_idx
