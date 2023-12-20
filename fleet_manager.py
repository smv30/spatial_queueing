import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgo, ChargerState, Dataset
from utils import calc_dist_between_two_points
from data_logging import DataLogging


# FleetManager run the functions every minute
class FleetManager:
    def __init__(self,
                 arrival_rate_pmin,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time_min,
                 list_chargers,
                 dataset,
                 trip_data,
                 matching_algo,
                 correction_factor,
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
        self.dataset = dataset
        self.trip_data = trip_data
        self.matching_algo = matching_algo
        self.correction_factor = correction_factor
        self.df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        self.d = d

    def match_trips(self):
        time_to_go_for_data_logging = 0
        # add a counter to record the current number (index) of the trip
        counter = 0
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
                if self.dataset == Dataset.RANDOMLYGENERATED.value:
                    previous_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter - 1]
                    current_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter]
                    inter_arrival_time_min = current_arrival_datetime - previous_arrival_datetime
                elif self.dataset == Dataset.NYTAXI.value:
                    previous_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter - 1]
                    current_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter]
                    inter_arrival_time_sec = current_arrival_datetime - previous_arrival_datetime
                    inter_arrival_time_min = inter_arrival_time_sec.total_seconds() / 60.0
                else:
                    raise ValueError("No such dataset exists")
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
                list_available_chargers = self.df_list_charger[self.df_list_charger["state"] == ChargerState.AVAILABLE.value]
                for car_id in list_cars:
                    list_posts_available = np.copy(self.df_list_charger["n_available_posts"])
                    car = self.car_tracker[car_id]
                    if car.soc <= 0.95:
                        closet_available_charger_idx = self.closet_available_charger(car, list_available_chargers)
                        if closet_available_charger_idx is not None:
                            car.prev_charging_process = self.env.process(
                                car.drive_to_charger(1, closet_available_charger_idx, self.correction_factor))
                            list_posts_available[closet_available_charger_idx] -= 1
                            if list_posts_available[closet_available_charger_idx] <= 0:
                                del list_available_chargers[closet_available_charger_idx]
            curr_time_min = self.env.now
            if self.dataset == Dataset.RANDOMLYGENERATED.value:
                trip_dist_mi = calc_dist_between_two_points(start_lat=self.trip_data["start_lat"].iloc[counter],
                                                            start_lon=self.trip_data["start_lon"].iloc[counter],
                                                            end_lat=self.trip_data["end_lat"].iloc[counter],
                                                            end_lon=self.trip_data["end_lon"].iloc[counter])
                trip_time_min = trip_dist_mi / SimMetaData.avg_vel_mph * 60
                trip = Trip(env=self.env,
                            arrival_time_min=curr_time_min,
                            trip_id=self.n_arrivals,
                            state=TripState.WAITING.value,
                            start_lon=self.trip_data["start_lon"].iloc[counter],
                            start_lat=self.trip_data["start_lat"].iloc[counter],
                            end_lon=self.trip_data["end_lon"].iloc[counter],
                            end_lat=self.trip_data["end_lat"].iloc[counter],
                            trip_time_min=trip_time_min)
            elif self.dataset == Dataset.NYTAXI.value:
                # change the start & end latitude & longitude
                pickup_datetime = self.trip_data["pickup_datetime"].iloc[counter]
                dropoff_datetime = self.trip_data["dropoff_datetime"].iloc[counter]
                trip_time_sec = dropoff_datetime - pickup_datetime
                trip_time_min = int(trip_time_sec.total_seconds() / 60)
                trip = Trip(env=self.env,
                            arrival_time_min=curr_time_min,
                            trip_id=self.n_arrivals,
                            state=TripState.WAITING.value,
                            # random=False,
                            start_lon=self.trip_data["pickup_longitude"].iloc[counter],
                            start_lat=self.trip_data["pickup_latitude"].iloc[counter],
                            end_lon=self.trip_data["dropoff_longitude"].iloc[counter],
                            end_lat=self.trip_data["dropoff_latitude"].iloc[counter],
                            trip_time_min=trip_time_min)
            else:
                raise ValueError("No such dataset exists")
            self.list_trips.append(trip)

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
                # service_time_min = pickup_time_min + trip_time_min
                # end_soc, closest_supercharger_idx = self.charge_just_after_trip_end(matched_car, service_time_min, trip)
                self.env.process(matched_car.run_trip(trip, self.correction_factor))
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
                                             end_lon=trip.start_lon,
                                             correction_factor=self.correction_factor)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.trip_time_min
        df_car_tracker["delta_soc"] = (
                                              (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
                                              * SimMetaData.avg_vel_mph
                                              * SimMetaData.consumption_kwhpmi
                                      ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = ChargingAlgoParams.safety_factor_to_reach_closest_charger * [
            min(calc_dist_between_two_points(
                start_lat=trip.end_lat,
                start_lon=trip.end_lon,
                end_lat=df_list_charger["lat"],
                end_lon=df_list_charger["lon"],
                correction_factor=self.correction_factor
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

    # old function
    # def power_of_d_closest_idle_or_charging(self, trip, df_car_tracker):
    #     idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
    #     driving_to_charger_mask = (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
    #     charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
    #     waiting_for_charger_mask = (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value)
    #     df_car_tracker["curr_soc"] = np.copy(df_car_tracker["soc"])
    #     df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "curr_soc"] = (
    #             (df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "state_start_time"]
    #              - self.env.now)
    #             / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
    #             + df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "soc"]
    #     )
    #     df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "curr_soc"] = (
    #             (self.env.now - df_car_tracker.loc[
    #                 df_car_tracker["state"] == CarState.CHARGING.value, "state_start_time"])
    #             / 60 * SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh
    #             + df_car_tracker.loc[df_car_tracker["state"] == CarState.CHARGING.value, "soc"]
    #     )
    #     df_car_tracker["pickup_time_min"] = (
    #             calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
    #                                          start_lon=df_car_tracker["lon"],
    #                                          end_lat=trip.start_lat,
    #                                          end_lon=trip.start_lon,
    #                                          correction_factor=self.correction_factor)
    #             / SimMetaData.avg_vel_mph
    #             * 60
    #     )
    #     trip_time_min = trip.trip_time_min
    #     df_car_tracker["delta_soc"] = (
    #                                           (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
    #                                           * SimMetaData.avg_vel_mph
    #                                           * SimMetaData.consumption_kwhpmi
    #                                   ) / SimMetaData.pack_size_kwh
    #     soc_to_reach_closest_supercharger = 0.1
    #     enough_soc_mask = (
    #             df_car_tracker["curr_soc"]
    #             - df_car_tracker["delta_soc"]
    #             - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
    #     )
    #
    #     available_cars = df_car_tracker[
    #         (idle_cars_mask | driving_to_charger_mask | charging_mask | waiting_for_charger_mask) & enough_soc_mask
    #         ].sort_values("pickup_time_min")
    #     if len(available_cars) > 0:
    #         available_cars_of_interest = available_cars.iloc[0:self.d]
    #         car_to_dispatch = available_cars_of_interest.sort_values("soc", ascending=False).iloc[0]
    #         return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"]
    #     else:
    #         return None, None

    # new function
    def power_of_d_closest_idle_or_charging(self, trip, df_car_tracker):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
        charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
        waiting_for_charger_mask = (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value)
        df_car_tracker["curr_soc"] = np.copy(df_car_tracker["soc"])
        # df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "curr_soc"] = (
        #         (df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "state_start_time"]
        #          - self.env.now)
        #         / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
        #         + df_car_tracker.loc[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value, "soc"]
        # )
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
                                             end_lon=trip.start_lon,
                                             correction_factor=self.correction_factor)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.trip_time_min
        df_car_tracker["delta_soc"] = (
                                              (df_car_tracker["pickup_time_min"] + trip_time_min) / 60
                                              * SimMetaData.avg_vel_mph
                                              * SimMetaData.consumption_kwhpmi
                                      ) / SimMetaData.pack_size_kwh
        soc_to_reach_closest_supercharger = min(calc_dist_between_two_points(
            start_lat=trip.end_lat,
            start_lon=trip.end_lon,
            end_lat=self.df_list_charger["lat"],
            end_lon=self.df_list_charger["lon"],
            correction_factor=self.correction_factor
        )) * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh * ChargingAlgoParams.safety_factor_to_reach_closest_charger
        cars_of_interest = df_car_tracker[
            (idle_cars_mask | charging_mask | waiting_for_charger_mask)
        ].sort_values(by=["pickup_time_min", "curr_soc"], ascending=[True, False])
        if len(cars_of_interest) == 0:
            return None, None
        d_closest_cars = cars_of_interest.iloc[0:self.d]
        possible_car_to_dispatch = d_closest_cars.sort_values("curr_soc", ascending=False).iloc[0]
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
        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["lat"],
                                             start_lon=df_car_tracker["lon"],
                                             end_lat=trip.start_lat,
                                             end_lon=trip.start_lon)
                / SimMetaData.avg_vel_mph
                * 60
        )
        trip_time_min = trip.trip_time_min
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

    # def charge_just_after_trip_end(self, car, service_time_min, trip):
    #     trip_end_soc = (
    #             car.soc
    #             - service_time_min / 60
    #             * SimMetaData.avg_vel_mph
    #             * SimMetaData.consumption_kwhpmi
    #             / SimMetaData.pack_size_kwh
    #     )
    #     end_soc = ChargingAlgoParams.higher_soc_threshold
    #     dist_to_superchargers = calc_dist_between_two_points(
    #         start_lat=trip.end_lat,
    #         start_lon=trip.end_lon,
    #         end_lat=self.df_list_charger["lat"],
    #         end_lon=self.df_list_charger["lon"],
    #         correction_factor=self.correction_factor
    #     )
    #     closest_supercharger_idx = np.argmin(dist_to_superchargers)
    #     if ChargingAlgoParams.send_all_idle_cars_to_charge:
    #         return 1, closest_supercharger_idx
    #     elif trip_end_soc < ChargingAlgoParams.lower_soc_threshold:
    #         return end_soc, closest_supercharger_idx
    #     else:
    #         return None, None

    def closet_available_charger(self, car, list_available_chargers):
        if len(list_available_chargers["lat"]) == 0:
            return None
        dist_to_superchargers = calc_dist_between_two_points(
            start_lat=car.lat,
            start_lon=car.lon,
            end_lat=list_available_chargers["lat"],
            end_lon=list_available_chargers["lon"],
            correction_factor=self.correction_factor
        )
        argmin_idx = np.argmin(dist_to_superchargers)
        closest_supercharger_idx = self.df_list_charger[
            self.df_list_charger["state"] == ChargerState.AVAILABLE.value
            ]["idx"].iloc[argmin_idx]
        return closest_supercharger_idx
