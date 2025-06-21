import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgo, ChargerState, ChargingAlgo
from utils import calc_dist_between_two_points, bin_numbers
from data_logging import DataLogging
from tqdm import tqdm


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
                 charging_algo,
                 d,
                 df_trip_data):
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
        self.charging_algo = charging_algo
        self.df_trip_data = df_trip_data

    def match_trips(self):
        time_to_go_for_data_logging = 0
        for _ in tqdm(range(len(self.df_trip_data)), desc="Simulating Arrivals"):  # everything inside runs every arrival
            df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
            self.df_list_charger = pd.DataFrame([
                self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
            ])
            if SimMetaData.random_data is True:
                inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / self.arrival_rate_pmin)
            else:
                inter_arrival_time_min = (self.df_trip_data["arrival_time"][self.n_arrivals + 1] -
                                          self.df_trip_data["arrival_time"][self.n_arrivals])
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
                    n_trips_till_now = len(self.list_trips)
                    n_trips_fulfilled_till_now = sum(int(self.list_trips[trip].state == TripState.MATCHED) for trip in range(len(self.list_trips)))
                    avg_soc = np.mean(df_car_tracker["soc"])
                    stdev_soc = np.std(df_car_tracker["soc"])
                    soc_dist = bin_numbers(df=df_car_tracker["soc"],
                                           bins=np.arange(1, 106, 5) / 100,
                                           bin_names=np.arange(1, 101, 5).astype(str))
                    soc_dist["time"] = self.env.now
                    self.data_logging.update_soc_dist(df_soc=soc_dist)
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
                                                  n_trips_till_now=n_trips_till_now,
                                                  n_trips_fulfilled_till_now=n_trips_fulfilled_till_now
                                                  )
                    time_to_go_for_data_logging = SimMetaData.freq_of_data_logging_min - inter_arrival_time_min
                else:
                    time_to_go_for_data_logging = time_to_go_for_data_logging - inter_arrival_time_min

            curr_time_min = self.env.now
            if SimMetaData.random_data is True:
                trip = Trip(env=self.env,
                            arrival_time_min=curr_time_min,
                            trip_id=self.n_arrivals,
                            state=TripState.WAITING.value)
            else:
                trip = Trip(env=self.env,
                            arrival_time_min=curr_time_min,
                            trip_id=self.n_arrivals,
                            state=TripState.WAITING.value,
                            random=False,
                            start_lat=self.df_trip_data["start_lat"][self.n_arrivals + 1],
                            start_lon=self.df_trip_data["start_lon"][self.n_arrivals + 1],
                            end_lat=self.df_trip_data["end_lat"][self.n_arrivals + 1],
                            end_lon=self.df_trip_data["end_lon"][self.n_arrivals + 1])
            self.list_trips.append(trip)

            if self.charging_algo == ChargingAlgo.CHARGE_AFTER_TRIP_END.value:
                sorted_list_cars_by_soc = df_car_tracker[df_car_tracker["state"] == CarState.IDLE.value].sort_values(
                            by="soc", ascending=True)["id"][0:4]  # Hardcoded as it slows down if taken large values
                for car_id in sorted_list_cars_by_soc:
                    car = self.car_tracker[car_id]
                    if car.soc <= ChargingAlgoParams.charging_soc_threshold:
                        closest_charger_idx = self.closest_available_charger(car)
                        if closest_charger_idx is not None:
                            car.prev_charging_process = self.env.process(car.drive_to_charger(1, closest_charger_idx))
            else:
                raise ValueError(f"Charging algorithm {self.charging_algo} does not exist")

            if self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle(trip, df_car_tracker)
            elif self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle_or_charging(trip, df_car_tracker)
            elif self.matching_algo == MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value:
                car_id, pickup_time_min = self.closest_available_dispatch(trip, df_car_tracker)
            else:
                raise ValueError(f"Matching algorithm {self.matching_algo} does not exist")

            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                self.env.process(matched_car.run_trip(trip))
            self.env.process(trip.update_trip_state(renege_time_min=self.renege_time_min))

            yield self.env.timeout(inter_arrival_time_min)
            self.n_arrivals = self.n_arrivals + 1
            if SimMetaData.random_data is False:
                if self.n_arrivals >= len(self.df_trip_data) - 1:
                    break

    def power_of_d_closest_idle(self, trip, df_car_tracker):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)

        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
                                             start_lon=df_car_tracker["lon"],
                                             end_lat=trip.start_lat,
                                             end_lon=trip.end_lon)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.calc_trip_time()
        df_car_tracker["delta_soc"] = (
                                              (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
                                              * SimMetaData.avg_vel_mph
                                              * SimMetaData.consumption_kwhpmi
                                      ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = [min(calc_dist_between_two_points(
            start_lat=self.car_tracker[car].lat,
            start_lon=self.car_tracker[car].lon,
            end_lat=self.df_list_charger["lat"],
            end_lon=self.df_list_charger["lon"],
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
        # Change the position of the car too!
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
        soc_to_reach_closest_supercharger = 0.15
        cars_of_interest = df_car_tracker[
            (idle_cars_mask | charging_mask | waiting_for_charger_mask)
            ].sort_values("pickup_time_min")
        if len(cars_of_interest) == 0:
            return None, None
        d_closest_cars = cars_of_interest.iloc[0:self.d]
        possible_car_to_dispatch = d_closest_cars.sort_values("soc", ascending=False).iloc[0]
        if (possible_car_to_dispatch["curr_soc"]
                - possible_car_to_dispatch["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
           ):
            return int(possible_car_to_dispatch["id"]), possible_car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def closest_available_dispatch(self, trip, df_car_tracker):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
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
        soc_to_reach_closest_supercharger = 0.02
        enough_soc_mask = (
                df_car_tracker["curr_soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )

        available_cars = df_car_tracker[
            (idle_cars_mask | charging_mask | waiting_for_charger_mask) & enough_soc_mask
            ].sort_values("pickup_time_min")
        if len(available_cars) > 0:
            available_cars_of_interest = available_cars.iloc[0:1]
            car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
            return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def closest_available_charger(self, car):
        if len(self.df_list_charger[self.df_list_charger["state"] == ChargerState.AVAILABLE.value]["lat"]) == 0:
            return None
        dist_to_superchargers = calc_dist_between_two_points(
            start_lat=car.lat,
            start_lon=car.lon,
            end_lat=self.df_list_charger[
                self.df_list_charger["state"] == ChargerState.AVAILABLE.value
                                        ]["lat"],
            end_lon=self.df_list_charger[
                self.df_list_charger["state"] == ChargerState.AVAILABLE.value
                                        ]["lon"],
        )
        argmin_idx = np.argmin(dist_to_superchargers)
        closest_supercharger_idx = self.df_list_charger[
            self.df_list_charger["state"] == ChargerState.AVAILABLE.value
                                                        ]["idx"].iloc[argmin_idx]
        return closest_supercharger_idx
