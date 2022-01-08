import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams
from utils import calc_dist_between_two_points


class FleetManager:
    def __init__(self,
                 arrival_rate_pmin,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time_min,
                 list_chargers):
        self.arrival_rate_pmin = arrival_rate_pmin
        self.env = env
        self.n_arrivals = 0
        self.car_tracker = car_tracker
        self.n_cars = n_cars
        self.renege_time_min = renege_time_min
        self.list_chargers = list_chargers

    def match_trips(self):
        while True:
            curr_time_min = self.env.now
            trip = Trip(env=self.env,
                        arrival_time_min=curr_time_min,
                        trip_id=self.n_arrivals,
                        state=TripState.WAITING.value)
            car_id, pickup_time_min = self.matching_algorithm(trip)

            inter_arrival_time_min = np.random.exponential(1 / self.arrival_rate_pmin)
            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                service_time_min = pickup_time_min + trip.calc_trip_time()
                end_soc, closest_supercharger_idx = self.charging_algorithm(matched_car, service_time_min)
                self.env.process(matched_car.run_trip(trip, end_soc, closest_supercharger_idx))
            self.env.process(trip.update_trip_state(renege_time_min=self.renege_time_min))

            yield self.env.timeout(inter_arrival_time_min)
            self.n_arrivals = self.n_arrivals + 1

    def matching_algorithm(self, trip):
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
        ) * SimMetaData.consumption_kwhpmi for car in range(self.n_cars)
        ]
        enough_soc_mask = (
                df_car_tracker["soc"]
                - df_car_tracker["delta_soc"]
                - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
        )

        available_cars = df_car_tracker[idle_cars_mask & enough_soc_mask]

        if len(available_cars) == 1:
            return int(available_cars.iloc[0]["id"]), available_cars.iloc[0]["pickup_time_min"]
        elif len(available_cars) > 1:
            available_cars.sort_values("pickup_time_min", inplace=True)
            if available_cars.iloc[0]["soc"] >= available_cars.iloc[1]["soc"]:
                loc = 0
            else:
                loc = 1
            return int(available_cars.iloc[loc]["id"]), available_cars.iloc[loc]["pickup_time_min"]
        else:
            return None, None

    def charging_algorithm(self, car, service_time_min):
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
