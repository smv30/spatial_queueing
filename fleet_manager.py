import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgoParams, MatchingAlgo
from utils import calc_dist_between_two_points
from data_logging import SocLogging


class FleetManager:
    def __init__(self,
                 arrival_rate_pmin,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time_min,
                 list_chargers,
                 matching_algo):
        self.arrival_rate_pmin = arrival_rate_pmin
        self.env = env
        self.n_arrivals = 0
        self.car_tracker = car_tracker
        self.n_cars = n_cars
        self.renege_time_min = renege_time_min
        self.list_chargers = list_chargers
        self.soc_logging = SocLogging()
        self.list_trips = []
        self.matching_algo = matching_algo

    def match_trips(self):
        while True:
            if ChargingAlgoParams.send_all_idle_cars_to_charge:
                for car in self.car_tracker:
                    if car.state == CarState.IDLE.value:
                        list_dist_to_charger = [
                            calc_dist_between_two_points(start_lat=car.lat,
                                                         start_lon=car.lon,
                                                         end_lat=self.list_chargers[charger].lat,
                                                         end_lon=self.list_chargers[charger].lon)
                            for charger in range(len(self.list_chargers))
                        ]
                        closest_charger_idx = np.argmax(list_dist_to_charger)
                        car.prev_charging_process = self.env.process(car.run_charge(1, closest_charger_idx))
            curr_time_min = self.env.now
            trip = Trip(env=self.env,
                        arrival_time_min=curr_time_min,
                        trip_id=self.n_arrivals,
                        state=TripState.WAITING.value)
            self.list_trips.append(trip)

            if self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle(trip)
            elif self.matching_algo == MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING.value:
                car_id, pickup_time_min = self.power_of_d_closest_idle_or_charging(trip)
            else:
                raise ValueError("No such matching algorithm exists")

            inter_arrival_time_min = SimMetaData.random_seed_gen.exponential(1 / self.arrival_rate_pmin)
            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                service_time_min = pickup_time_min + trip.calc_trip_time()
                end_soc, closest_supercharger_idx = self.charge_just_after_trip_end(matched_car, service_time_min)
                self.env.process(matched_car.run_trip(trip, end_soc, closest_supercharger_idx))
            self.env.process(trip.update_trip_state(renege_time_min=self.renege_time_min))

            yield self.env.timeout(inter_arrival_time_min)
            self.n_arrivals = self.n_arrivals + 1
            list_soc = [self.car_tracker[car].soc for car in range(self.n_cars)]
            self.soc_logging.update_list_of_soc(curr_list_soc=list_soc)

    def power_of_d_closest_idle(self, trip):
        df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
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
        trip_time_min = trip.calc_trip_time()
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
            d = MatchingAlgoParams.d
            available_cars_of_interest = available_cars.iloc[0:d]
            car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
            return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def power_of_d_closest_idle_or_charging(self, trip):
        df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
        driving_to_charger_mask = (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
        charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
        df_car_tracker["curr_soc"] = df_car_tracker["soc"]
        df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "curr_soc"] = (
            (df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "state_start_time"]
             - self.env.now)
            / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
            + df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "soc"]
        )
        df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "curr_soc"] = (
            (self.env.now - df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "state_start_time"])
            / 60 * SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh
            + df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "soc"]
        )

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
        trip_time_min = trip.calc_trip_time()
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
                df_car_tracker["curr_soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )

        available_cars = df_car_tracker[
            (idle_cars_mask | driving_to_charger_mask | charging_mask) & enough_soc_mask
        ].sort_values("pickup_time_min")
        if len(available_cars) > 0:
            d = MatchingAlgoParams.d
            available_cars_of_interest = available_cars.iloc[0:d]
            car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
            return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
        else:
            return None, None

    def charge_just_after_trip_end(self, car, service_time_min):
        df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        trip_end_soc = (
                    car.soc
                    - service_time_min / 60
                    * SimMetaData.avg_vel_mph
                    * SimMetaData.consumption_kwhpmi
                    / SimMetaData.pack_size_kwh
            )
        if trip_end_soc < ChargingAlgoParams.lower_soc_threshold:
            end_soc = ChargingAlgoParams.higher_soc_threshold
            dist_to_superchargers = calc_dist_between_two_points(
                                        start_lat=car.lat,
                                        start_lon=car.lon,
                                        end_lat=df_list_charger["lat"],
                                        end_lon=df_list_charger["lon"],
                                    )
            closest_supercharger_idx = np.argmin(dist_to_superchargers)
        else:
            end_soc = None
            closest_supercharger_idx = None

        return end_soc, closest_supercharger_idx
