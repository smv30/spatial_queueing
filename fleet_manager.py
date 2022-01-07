import numpy as np
import pandas as pd
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData
from utils import calc_dist_between_two_points


class FleetManager:
    def __init__(self,
                 arrival_rate,
                 env,
                 car_tracker,
                 n_cars,
                 renege_time,
                 list_chargers):
        self.arrival_rate = arrival_rate
        self.env = env
        self.n_arrivals = 0
        self.car_tracker = car_tracker
        self.n_cars = n_cars
        self.renege_time = renege_time
        self.list_chargers = list_chargers

    def match_trips(self):
        while True:
            curr_time = self.env.now
            trip = Trip(env=self.env,
                        arrival_time=curr_time,
                        trip_id=self.n_arrivals,
                        state=TripState.WAITING.value)
            car_id, pickup_time = self.matching_algorithm(trip)

            inter_arrival_time = np.random.exponential(1 / self.arrival_rate)
            if car_id is not None:
                matched_car = self.car_tracker[car_id]
                trip.state = TripState.MATCHED
                self.env.process(matched_car.run_trip(trip))
                service_time = pickup_time + trip.calc_trip_time()
                if service_time < inter_arrival_time:
                    charge_kwh, charger_lat, charger_lon = self.charging_algorithm(matched_car)
                    if charge_kwh:
                        self.env.process(matched_car.run_charge(charge_kwh, charger_lat, charger_lon))
            self.env.process(trip.update_trip_state(renege_time=self.renege_time))

            yield self.env.timeout(inter_arrival_time)
            self.n_arrivals = self.n_arrivals + 1

    def matching_algorithm(self, trip):
        df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)

        df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        df_car_tracker["pickup_time"] = (
                (df_car_tracker["lat"] - trip.start_lat) ** 2
                + (df_car_tracker["lon"] - trip.start_lon) ** 2) ** 0.5 / SimMetaData.avg_vel
        trip_time = trip.calc_trip_time()
        df_car_tracker["delta_soc"] = (
                (df_car_tracker["pickup_time"] + trip_time)
                * SimMetaData.avg_vel
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
            return int(available_cars.iloc[0]["id"]), available_cars.iloc[0]["pickup_time"]
        elif len(available_cars) > 1:
            available_cars.sort_values("pickup_time", inplace=True)
            if available_cars.iloc[0]["soc"] >= available_cars.iloc[1]["soc"]:
                loc = 0
            else:
                loc = 1
            return int(available_cars.iloc[loc]["id"]), available_cars.iloc[loc]["pickup_time"]
        else:
            return None, None

    def charging_algorithm(self, car):
        return 0, 1, 1
