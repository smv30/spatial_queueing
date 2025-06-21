import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from arrivals import Trip
from sim_metadata import TripState, CarState, SimMetaData, ChargingAlgoParams, MatchingAlgo, ChargerState, Dataset, \
    AdaptivePowerOfDParams, ChargingAlgo, AvailableCarsForMatching, PickupThresholdType, PickupThresholdMatchingParams
from utils import calc_dist_between_two_points
from data_logging import DataLogging
from datetime import timedelta
from mpl_toolkits.basemap import Basemap


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
                 start_datetime,
                 dist_func,
                 d,
                 plot_dir):
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
        self.original_charging_algo = charging_algo
        self.dist_correction_factor = dist_correction_factor
        self.df_list_charger = pd.DataFrame([
            self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
        ])
        self.d = d
        self.dist_func = dist_func
        self.pickup_threshold_type = pickup_threshold_type
        self.start_datetime = start_datetime
        self.plot_dir = plot_dir

    def match_trips(self):
        time_to_go_for_data_logging = 0
        time_to_go_for_plotting = 0
        # add a counter to record the current number (index) of the trip
        counter = 0
        n_served_trips_before_updating_d = 0
        n_cars_idle_full_soc_average = 0
        charge_threshold = 0.95
        is_it_night = False
        if self.original_charging_algo in [
            ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value,
            ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT_WITH_RELOCATION.value,
            ChargingAlgo.CAN_WITH_IC_AT_NIGHT.value
            ]:
            # Precompute the workload in the night
            first_trip_pickup_datetime = self.trip_data["pickup_datetime"].min()
            last_trip_pickup_datetime = self.trip_data["pickup_datetime"].max()
            total_sim_time_datetime = last_trip_pickup_datetime - first_trip_pickup_datetime
            total_days_float = total_sim_time_datetime.total_seconds() / 24 / 60 / 60
            total_days = int(total_days_float) + (total_days_float > 0)  # Round up
            list_total_kwh_spent_night = []
            for day in range(total_days):
                delta_start_hours = max((day - 1) * 24 + ChargingAlgoParams.start_of_the_night, 0)
                delta_end_hours = day * 24 + ChargingAlgoParams.end_of_the_night
                datetime_night_start = self.start_datetime + timedelta(hours=delta_start_hours)
                datetime_night_end = self.start_datetime + timedelta(hours=delta_end_hours)
                df_night_trips = self.trip_data[
                    (self.trip_data["pickup_datetime"] >= datetime_night_start) &
                    (self.trip_data["pickup_datetime"] <= datetime_night_end)
                    ]
                total_kwh_spent = sum(df_night_trips["trip_distance"]) * SimMetaData.consumption_kwhpmi
                list_total_kwh_spent_night.append(total_kwh_spent)
        while True:  # everything inside runs every arrival
            df_car_tracker = pd.DataFrame([self.car_tracker[car].to_dict() for car in range(self.n_cars)])
            self.df_list_charger = pd.DataFrame([
                self.list_chargers[charger].to_dict() for charger in range(len(self.list_chargers))
            ])
            self.df_list_charger = self.df_list_charger.set_index("idx")
            if counter == 0:
                inter_arrival_time_min = 0
            elif counter == len(self.trip_data):
                break
            else:
                previous_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter - 1]
                current_arrival_datetime = self.trip_data["pickup_datetime"].iloc[counter]
                inter_arrival_time_datetime = current_arrival_datetime - previous_arrival_datetime
                inter_arrival_time_min = inter_arrival_time_datetime.total_seconds() / 60.0
            list_idle_cars = df_car_tracker[
                df_car_tracker["state"] == CarState.IDLE.value].sort_values(by="soc", ascending=True)["id"]
            available_charger_mask = (self.df_list_charger["n_available_posts"]
                                      - ChargingAlgoParams.n_cars_driving_to_charger_discounter
                                      * self.df_list_charger["n_cars_driving_to_charger"] > 0
                                      )
            list_available_chargers = self.df_list_charger[available_charger_mask]
            if self.original_charging_algo == ChargingAlgo.CAN_WITH_IC_AT_NIGHT.value: # Switches between two charging algorithms depending on day and night
                curr_time_in_the_day_min = self.env.now % (24 * 60)
                # Check if it is currently day
                # Provide a buffer of 5 mins for night activities to be finished
                if (
                    (curr_time_in_the_day_min >= ChargingAlgoParams.end_of_the_night * 60 
                     or curr_time_in_the_day_min <= ChargingAlgoParams.start_of_the_night * 60 + 5 - 24 * 60
                         )
                    and (curr_time_in_the_day_min <= ChargingAlgoParams.start_of_the_night * 60 + 5
                         )
                    ): 
                    self.charging_algo = ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT_WITH_RELOCATION.value
                    self.available_cars_for_matching = AvailableCarsForMatching.IDLE_AND_RELOCATING.value
                    self.d = SimMetaData.d_during_the_day # Set to be equal to 1 by default
                else:
                    self.charging_algo = ChargingAlgo.CHARGE_ALL_IDLE_CARS.value
                    self.available_cars_for_matching = AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value
                    self.d = SimMetaData.d_during_the_night # Set to be the input d to the simulation
            if self.charging_algo in [ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value, ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT_WITH_RELOCATION.value]:
                curr_time_in_the_day_min = self.env.now % (24 * 60)
                was_it_night = is_it_night
                if (ChargingAlgoParams.end_of_the_night * 60
                        <= curr_time_in_the_day_min
                        <= ChargingAlgoParams.start_of_the_night * 60
                ):  # Check if it is currently day
                    if SimMetaData.pack_size_kwh <= 40: # Make charge threshold higher if pack size is low
                        charge_threshold = 0.2  # Send to charge only if SoC is very low
                    else:
                        charge_threshold = 0.1
                    is_it_night = False
                else:
                    charge_threshold = 0.95  # Send all EVs to charge (at night)
                    is_it_night = True
                if is_it_night is True and was_it_night is False:  # start of the night
                    is_it_start_of_night = True
                else:
                    is_it_start_of_night = False
                if is_it_start_of_night is True:
                    # Interrupt all fake charging sessions at the start of the night
                    charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
                    fake_charging_mask = (df_car_tracker["fake_charging_bool"] == True)
                    list_fake_charging_evs = df_car_tracker[charging_mask & fake_charging_mask].sort_values(by="soc", ascending=True)["id"]
                    for car_id in list_fake_charging_evs:
                        car = self.car_tracker[car_id]
                        car.interrupt_charging(
                                charger_idx=car.charging_at_idx,
                                end_soc=car.end_soc_post_charging,
                                dist_correction_factor=self.dist_correction_factor,
                                dist_func=self.dist_func
                            )
                    start_of_night_soc = np.mean(df_car_tracker["soc"])
                    curr_day = int(self.env.now / (24 * 60))
                    kwh_to_gain = ((1 - start_of_night_soc) * SimMetaData.pack_size_kwh * self.n_cars
                                   + list_total_kwh_spent_night[curr_day]
                                   )
                    hours_in_night = ChargingAlgoParams.end_of_the_night - ChargingAlgoParams.start_of_the_night + 24
                    kw_at_night = kwh_to_gain / hours_in_night
                    target_n_cars_charge_night = kw_at_night / SimMetaData.charge_rate_kw
                    curr_n_cars_charging_at_night = np.zeros(hours_in_night * 60)
                    curr_n_cars_driving_to_charger = len(df_car_tracker[df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value])
                    required_charge_time_min = int((1 - charge_threshold) * SimMetaData.pack_size_kwh / SimMetaData.charge_rate_kw * 60)
                    curr_n_cars_charging_at_night[0:required_charge_time_min] += curr_n_cars_driving_to_charger
                for car_id in list_idle_cars:
                    car = self.car_tracker[car_id]
                    if car.soc <= charge_threshold:
                        closest_available_charger_idx, dist_to_charger_mi = self.closest_available_charger(car,
                                                                                                           list_available_chargers)
                        if closest_available_charger_idx is not None:
                            if car.soc <= dist_to_charger_mi * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh:
                                continue # If the car cannot reach the charger with the soc it has, then do not send it to charge - try next time!
                            # set end_soc = just enough to drive the rest of the day without charging in charge at night algorithm
                            if is_it_night is False:  # Check if it is currently day
                                time_to_go_for_night_min = ChargingAlgoParams.start_of_the_night * 60 - curr_time_in_the_day_min
                                delta_soc_to_drive_all_day = (
                                        time_to_go_for_night_min / 60 *
                                        SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi
                                        / SimMetaData.pack_size_kwh
                                )
                                end_soc = min(car.soc + delta_soc_to_drive_all_day, 1)
                            else:
                                drive_to_charger_min = int(dist_to_charger_mi / SimMetaData.avg_vel_mph * 60)
                                end_soc = 1
                                charging_time_min = int(
                                    (1 - car.soc) * SimMetaData.pack_size_kwh / SimMetaData.charge_rate_kw * 60)
                                start_of_the_night_min = ChargingAlgoParams.start_of_the_night * 60 % (24 * 60)
                                mins_elapsed_at_night = int(curr_time_in_the_day_min - start_of_the_night_min)
                                start_of_charge_min = mins_elapsed_at_night + drive_to_charger_min
                                end_of_charge_min = start_of_charge_min + charging_time_min
                                curr_n_cars_charging_at_night[mins_elapsed_at_night] = len(df_car_tracker[df_car_tracker["state"] == CarState.CHARGING.value])
                                if start_of_charge_min >= len(curr_n_cars_charging_at_night): # Checking if the charging starts only after the night is over
                                    continue # In that case, do not send to charge
                                else:
                                    if max(curr_n_cars_charging_at_night[start_of_charge_min:end_of_charge_min]) >= 1.1 * target_n_cars_charge_night: 
                                        break # If there are too many cars charging already, then do not send this car to charge
                                        # Should be ideally continue but break runs much faster and performance is not degraded with break as we are sending cars to charge every once in a while
                                    else:
                                        curr_n_cars_charging_at_night[start_of_charge_min:end_of_charge_min] += 1
                            car.prev_charging_process = self.env.process(
                                car.drive_to_charger(
                                    end_soc=end_soc,
                                    charger_idx=closest_available_charger_idx,
                                    dist_correction_factor=self.dist_correction_factor,
                                    dist_func=self.dist_func,
                                    list_available_chargers=list_available_chargers,
                                    fake_charging_bool=False
                                ))
                            list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] -= 1
                            if list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] <= 0:
                                list_available_chargers = list_available_chargers.drop(
                                    closest_available_charger_idx)
                    elif car.soc <= 0.95 and is_it_night is False and self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT_WITH_RELOCATION.value:
                        closest_available_charger_idx, dist_to_charger_mi = self.closest_available_charger(car,
                                                                                                           list_available_chargers)
                        if closest_available_charger_idx is not None:
                            if car.soc <= dist_to_charger_mi * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh:
                                continue # If the car cannot reach the charger with the soc it has, then do not send it to charge - try next time!
                            car.prev_charging_process = self.env.process(
                                    car.drive_to_charger(
                                        end_soc=1,
                                        charger_idx=closest_available_charger_idx,
                                        dist_correction_factor=self.dist_correction_factor,
                                        dist_func=self.dist_func,
                                        list_available_chargers=list_available_chargers,
                                        fake_charging_bool=True
                                    ))
                            list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] -= 1
                            if list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] <= 0:
                                list_available_chargers = list_available_chargers.drop(closest_available_charger_idx)
            elif self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS.value:
                for car_id in list_idle_cars:
                    car = self.car_tracker[car_id]
                    if car.soc <= charge_threshold:
                        closest_available_charger_idx, dist_to_charger_mi = self.closest_available_charger(car,
                                                                                                           list_available_chargers)
                        if closest_available_charger_idx is not None:
                            if car.soc <= dist_to_charger_mi * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh:
                                continue # If the car cannot reach the charger with the soc it has, then do not send it to charge - try next time!
                            car.prev_charging_process = self.env.process(
                                    car.drive_to_charger(
                                        end_soc=1,
                                        charger_idx=closest_available_charger_idx,
                                        dist_correction_factor=self.dist_correction_factor,
                                        dist_func=self.dist_func,
                                        list_available_chargers=list_available_chargers
                                    ))
                            list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] -= 1
                            if list_available_chargers.at[closest_available_charger_idx, "n_available_posts"] <= 0:
                                list_available_chargers = list_available_chargers.drop(closest_available_charger_idx)
            elif self.charging_algo == ChargingAlgo.CHARGE_ALL_IDLE_CARS_RADIUS_DISPATCH.value:
                list_available_chargers["dist_to_car"] = 100
                for car_id in list_idle_cars:
                    car = self.car_tracker[car_id]
                    if car.soc <= charge_threshold:
                        list_available_chargers = list_available_chargers[list_available_chargers["n_available_posts"] > 0]
                        list_available_chargers.loc["dist_to_car"] = calc_dist_between_two_points(
                            start_lat=car.lat,
                            start_lon=car.lon,
                            end_lat=list_available_chargers["lat"],
                            end_lon=list_available_chargers["lon"],
                            dist_correction_factor=self.dist_correction_factor,
                            dist_func=self.dist_func
                        )
                        radius_mask = (list_available_chargers["dist_to_car"] <= ChargingAlgoParams.radius_dispatch_min / 60 * SimMetaData.avg_vel_mph)
                        charger_to_send = list_available_chargers[radius_mask].sort_values(by=["n_available_posts", "dist_to_car"], ascending=[False, True])
                        if len(charger_to_send) > 0:
                            charger_idx_to_send = charger_to_send.index[0]
                        else:
                            charger_idx_to_send = None
                        if charger_idx_to_send is not None:
                            car.prev_charging_process = self.env.process(
                                car.drive_to_charger(
                                    end_soc=1,
                                    charger_idx=charger_idx_to_send,
                                    dist_correction_factor=self.dist_correction_factor,
                                    dist_func=self.dist_func,
                                    list_available_chargers=list_available_chargers
                                ))
                            list_available_chargers.at[charger_idx_to_send, "n_available_posts"] -= 1
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

            car_id, pickup_time_min, n_available_cars_to_match = self.matching_algorithms(trip=trip,
                                                                                          df_car_tracker=df_car_tracker)
            trip.n_available_cars_to_match = n_available_cars_to_match

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
                    n_cars_relocating = sum(
                        df_car_tracker["state"] == CarState.RELOCATING.value
                    )
                    n_cars_fake_driving_to_charger = sum(
                        (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value) & df_car_tracker["fake_charging_bool"]
                    )
                    n_cars_fake_charging = sum(
                        (df_car_tracker["state"] == CarState.CHARGING.value) & df_car_tracker["fake_charging_bool"]
                    )
                    n_cars_fake_waiting_for_charger = sum(
                        (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value) & df_car_tracker["fake_charging_bool"]
                    )
                    avg_soc = np.mean(df_car_tracker["soc"])
                    idle = df_car_tracker[df_car_tracker["state"] == CarState.IDLE.value]
                    if not idle.empty:
                        avg_soc_of_idle_evs = idle["soc"].mean()
                    else:
                        avg_soc_of_idle_evs = None  # or 0.0, or whatever makes sense
                    stdev_soc = np.std(df_car_tracker["soc"])
                    d = self.d
                    self.data_logging.update_data(curr_list_soc=list_soc,
                                                  n_cars_idle=n_cars_idle,
                                                  n_cars_charging=n_cars_charging,
                                                  n_cars_driving_to_charger=n_cars_driving_to_charger,
                                                  n_cars_driving_without_passenger=n_cars_driving_without_passenger,
                                                  n_cars_driving_with_passenger=n_cars_driving_with_passenger,
                                                  n_cars_waiting_for_charger=n_cars_waiting_for_charger,
                                                  n_cars_relocating=n_cars_relocating,
                                                  time_of_logging=self.env.now,
                                                  avg_soc=avg_soc,
                                                  stdev_soc=stdev_soc,
                                                  d=d,
                                                  charge_threshold=charge_threshold,
                                                  n_cars_fake_driving_to_charger=n_cars_fake_driving_to_charger,
                                                  n_cars_fake_charging=n_cars_fake_charging,
                                                  n_cars_fake_waiting_for_charger=n_cars_fake_waiting_for_charger,
                                                  avg_soc_of_idle_evs=avg_soc_of_idle_evs
                                                  )
                    time_to_go_for_data_logging = SimMetaData.freq_of_data_logging_min - inter_arrival_time_min
                else:
                    time_to_go_for_data_logging = time_to_go_for_data_logging - inter_arrival_time_min
            if time_to_go_for_plotting <= 0:
                 # Spatial Plots
                # Get latitude and longitude data
                pickup_lat = []
                pickup_lon = []
                dropoff_lat = []
                dropoff_lon = []
                for trip in self.list_trips:
                    pickup_lat.append(trip.start_lat)
                    pickup_lon.append(trip.start_lon)
                    dropoff_lat.append(trip.end_lat)
                    dropoff_lon.append(trip.end_lon)
                self.spatial_plot(
                    lat=pickup_lat,
                    lon=pickup_lon,
                    file_name=f"{int(self.env.now)}_pickup_loc.png"
                )
                self.spatial_plot(
                    lat=dropoff_lat,
                    lon=dropoff_lon,
                    file_name=f"{int(self.env.now)}_dropoff_loc.png"
                )
                mask = (df_car_tracker["state"] == CarState.IDLE.value)
                self.spatial_plot(
                    lat=df_car_tracker[mask]["lat"],
                    lon=df_car_tracker[mask]["lon"],
                    file_name=f"{int(self.env.now)}_idle_car_loc.png"
                )
                time_to_go_for_plotting = 500 - inter_arrival_time_min
            else:
                time_to_go_for_plotting = time_to_go_for_plotting - inter_arrival_time_min

    def matching_algorithms(self, trip, df_car_tracker, drive_to_charger_update=True, relocating_update=True):
        idle_cars_mask = (df_car_tracker["state"] == CarState.IDLE.value)
        charging_mask = (df_car_tracker["state"] == CarState.CHARGING.value)
        waiting_for_charger_mask = (df_car_tracker["state"] == CarState.WAITING_FOR_CHARGER.value)
        driving_to_charger_mask = (df_car_tracker["state"] == CarState.DRIVING_TO_CHARGER.value)
        relocating_mask = (df_car_tracker["state"] == CarState.RELOCATING.value)
        fake_charging_mask = (df_car_tracker["fake_charging_bool"] == True)
        df_car_tracker["curr_soc"] = np.copy(df_car_tracker["soc"])
        df_car_tracker.loc[(df_car_tracker["state"] == CarState.CHARGING.value) & (df_car_tracker["fake_charging_bool"] == False), "curr_soc"] = (
                (self.env.now - df_car_tracker.loc[
                    (df_car_tracker["state"] == CarState.CHARGING.value) & (df_car_tracker["fake_charging_bool"] == False), "state_start_time"])
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
        df_car_tracker.loc[df_car_tracker["state"] == CarState.RELOCATING.value, "curr_soc"] = \
            (
                    df_car_tracker.loc[df_car_tracker["state"] == CarState.RELOCATING.value, "soc"]
                    - (self.env.now
                       - df_car_tracker.loc[
                           df_car_tracker["state"] == CarState.RELOCATING.value, "state_start_time"]
                       )
                    / 60 * SimMetaData.avg_vel_mph * SimMetaData.consumption_kwhpmi / SimMetaData.pack_size_kwh
            )
        df_car_tracker["curr_lat"] = np.copy(df_car_tracker["lat"])
        df_car_tracker["curr_lon"] = np.copy(df_car_tracker["lon"])
        if (drive_to_charger_update is True
                and self.available_cars_for_matching in [AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value, AvailableCarsForMatching.IDLE_AND_RELOCATING.value]
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
            dist_to_charger_mask = dist_to_charger > 0
            filtered_car_tracker = filtered_car_tracker.loc[dist_to_charger_mask]
            dist_to_charger = dist_to_charger[dist_to_charger_mask]
            dist_traveled = (self.env.now - filtered_car_tracker["state_start_time"]) / 60 * SimMetaData.avg_vel_mph

            filtered_car_tracker["curr_lat"] = (filtered_car_tracker["lat"] +
                                                (filtered_car_tracker["charging_at_lat"] - filtered_car_tracker["lat"])
                                                * dist_traveled / dist_to_charger
                                                )
            filtered_car_tracker["curr_lon"] = (filtered_car_tracker["lon"] +
                                                (filtered_car_tracker["charging_at_lon"] - filtered_car_tracker["lon"])
                                                * dist_traveled / dist_to_charger
                                                )
            df_car_tracker.loc[filtered_car_tracker.index, "curr_lat"] = filtered_car_tracker["curr_lat"]
            df_car_tracker.loc[filtered_car_tracker.index, "curr_lon"] = filtered_car_tracker["curr_lon"]
        if (relocating_update is True
                and self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_RELOCATING.value
                and sum(df_car_tracker["state"] == CarState.RELOCATING.value) > 1
        ):
            filtered_car_tracker = df_car_tracker.loc[df_car_tracker["state"] == CarState.RELOCATING.value]
            dist_to_relocate = calc_dist_between_two_points(
                start_lat=filtered_car_tracker["lat"],
                start_lon=filtered_car_tracker["lon"],
                end_lat=filtered_car_tracker["relocating_to_lat"],
                end_lon=filtered_car_tracker["relocating_to_lon"],
                dist_correction_factor=self.dist_correction_factor,
                dist_func=self.dist_func
            )
            dist_to_relocate_mask = dist_to_relocate > 0
            filtered_car_tracker = filtered_car_tracker.loc[dist_to_relocate_mask]
            dist_to_relocate = dist_to_relocate[dist_to_relocate_mask]
            dist_traveled = (self.env.now - filtered_car_tracker["state_start_time"]) / 60 * SimMetaData.avg_vel_mph

            filtered_car_tracker["curr_lat"] = (filtered_car_tracker["lat"] +
                                                (filtered_car_tracker["relocating_to_lat"] - filtered_car_tracker["lat"])
                                                * dist_traveled / dist_to_relocate
                                                )
            filtered_car_tracker["curr_lon"] = (filtered_car_tracker["lon"] +
                                                (filtered_car_tracker["relocating_to_lon"] - filtered_car_tracker["lon"])
                                                * dist_traveled / dist_to_relocate
                                                )
            df_car_tracker.loc[filtered_car_tracker.index, "curr_lat"] = filtered_car_tracker["curr_lat"]
            df_car_tracker.loc[filtered_car_tracker.index, "curr_lon"] = filtered_car_tracker["curr_lon"]
        df_car_tracker["pickup_time_min"] = (
                calc_dist_between_two_points(start_lat=df_car_tracker["curr_lat"],
                                             start_lon=df_car_tracker["curr_lon"],
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
        if self.available_cars_for_matching == AvailableCarsForMatching.ONLY_IDLE.value:
            n_available_cars_to_match = len(df_car_tracker[idle_cars_mask])
        elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_CHARGING.value:
            n_available_cars_to_match = len(df_car_tracker[(idle_cars_mask | charging_mask)])
        elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value:
            n_available_cars_to_match = len(df_car_tracker[(idle_cars_mask | charging_mask | waiting_for_charger_mask)])
        elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_RELOCATING.value:
            n_available_cars_to_match = len(df_car_tracker[(idle_cars_mask | relocating_mask | (waiting_for_charger_mask | charging_mask | driving_to_charger_mask) & fake_charging_mask)])
        else:
            raise ValueError("Such an input for the available cars for matching is invalid")
        if self.pickup_threshold_type == PickupThresholdType.MIN_AVAILABLE_CARS_PERCENT.value:
            if n_available_cars_to_match <= PickupThresholdMatchingParams.min_available_cars_percent * self.n_cars:
                return None, None, n_available_cars_to_match
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
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_RELOCATING.value:
                cars_of_interest = df_car_tracker[
                    (idle_cars_mask | relocating_mask | (waiting_for_charger_mask | charging_mask | driving_to_charger_mask) & fake_charging_mask)
                ].sort_values(by=["pickup_time_min", "curr_soc"], ascending=[True, False])
            else:
                raise ValueError("Such an input for the available cars for matching is invalid")
            if len(cars_of_interest) == 0:
                return None, None, n_available_cars_to_match
            if isinstance(self.d, int) is False:
                frac_part = self.d % 1
                random_d = int(self.d + SimMetaData.random_seed_gen.binomial(n=1, p=frac_part))
            else:
                random_d = self.d
            d_closest_cars = cars_of_interest.iloc[0:random_d]
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
                                n_available_cars_to_match
                                )
                elif self.pickup_threshold_type == PickupThresholdType.CONSTANT_THRESHOLD.value:
                    if possible_car_to_dispatch["pickup_time_min"] <= PickupThresholdMatchingParams.threshold_min:
                        return (
                            int(possible_car_to_dispatch["id"]),
                            possible_car_to_dispatch["pickup_time_min"],
                            n_available_cars_to_match
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
                            n_available_cars_to_match
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
                            n_available_cars_to_match
                        )
                elif self.pickup_threshold_type in [
                    PickupThresholdType.NO_THRESHOLD.value,
                    PickupThresholdType.MIN_AVAILABLE_CARS_PERCENT.value
                ]:
                    return (
                        int(possible_car_to_dispatch["id"]),
                        possible_car_to_dispatch["pickup_time_min"],
                        n_available_cars_to_match
                    )
                else:
                    raise ValueError("No such thresholding scheme exists")
            return None, None, n_available_cars_to_match
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
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_RELOCATING.value:
                available_cars = df_car_tracker[
                    (idle_cars_mask | relocating_mask | (waiting_for_charger_mask | charging_mask | driving_to_charger_mask) & fake_charging_mask) & enough_soc_mask
                ].sort_values(by=["pickup_time_min", "soc"], ascending=[True, False])
            else:
                raise ValueError("Such an input for the available cars for matching is invalid")
            if len(available_cars) > 0:
                car_to_dispatch = available_cars.iloc[0]
                if self.pickup_threshold_type == PickupThresholdType.PERCENT_THRESHOLD.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min
                    ):
                        return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"], n_available_cars_to_match
                elif self.pickup_threshold_type == PickupThresholdType.CONSTANT_THRESHOLD.value:
                    if car_to_dispatch["pickup_time_min"] <= PickupThresholdMatchingParams.threshold_min:
                        return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"], n_available_cars_to_match
                elif self.pickup_threshold_type == PickupThresholdType.BOTH_PERCENT_AND_CONSTANT.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= min(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"], n_available_cars_to_match
                elif self.pickup_threshold_type == PickupThresholdType.EITHER_PERCENT_OR_CONSTANT.value:
                    if (
                            car_to_dispatch["pickup_time_min"]
                            <= max(PickupThresholdMatchingParams.threshold_percent * trip.trip_time_min,
                                   PickupThresholdMatchingParams.threshold_min
                                   )
                    ):
                        return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"], n_available_cars_to_match
                elif self.pickup_threshold_type in [
                    PickupThresholdType.NO_THRESHOLD.value,
                    PickupThresholdType.MIN_AVAILABLE_CARS_PERCENT.value
                ]:
                    return int(car_to_dispatch["id"]), car_to_dispatch["pickup_time_min"], n_available_cars_to_match
                else:
                    raise ValueError("No such thresholding scheme exists")
            return None, None, n_available_cars_to_match
        elif self.matching_algo == MatchingAlgo.POWER_OF_RADIUS.value:
            if self.available_cars_for_matching == AvailableCarsForMatching.ONLY_IDLE.value:
                cars_of_interest = df_car_tracker[idle_cars_mask]
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_CHARGING.value:
                cars_of_interest = df_car_tracker[(idle_cars_mask | charging_mask | waiting_for_charger_mask)]
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value:
                cars_of_interest = df_car_tracker[
                    (idle_cars_mask | charging_mask | waiting_for_charger_mask | driving_to_charger_mask)
                ]
            elif self.available_cars_for_matching == AvailableCarsForMatching.IDLE_AND_RELOCATING.value:
                cars_of_interest = df_car_tracker[
                    (idle_cars_mask | relocating_mask | (waiting_for_charger_mask | charging_mask | driving_to_charger_mask) & fake_charging_mask)
                ]
            else:
                raise ValueError("Such an input for the available cars for matching is invalid")
            pickup_threshold_mask = (cars_of_interest["pickup_time_min"] <= PickupThresholdMatchingParams.threshold_min)
            cars_in_a_radius = cars_of_interest[pickup_threshold_mask]
            if len(cars_in_a_radius) == 0:
                return None, None, n_available_cars_to_match
            possible_car_to_dispatch = cars_in_a_radius.sort_values("curr_soc", ascending=False).iloc[0]
            if (possible_car_to_dispatch["curr_soc"]
                    - possible_car_to_dispatch["delta_soc"]
                    - soc_to_reach_closest_supercharger > SimMetaData.min_allowed_soc
            ):
                return (
                    int(possible_car_to_dispatch["id"]),
                     possible_car_to_dispatch["pickup_time_min"],
                      n_available_cars_to_match
                )
            else:
                return None, None, n_available_cars_to_match
        else:
            raise ValueError(f"Matching algorithm {self.matching_algo} does not exist")

    def closest_available_charger(self, car, list_available_chargers):
        n_avail = len(list_available_chargers["lat"])
        if n_avail <= 1:
            return None, None
        dist_to_supercharger = calc_dist_between_two_points(
            start_lat=car.lat,
            start_lon=car.lon,
            end_lat=list_available_chargers["lat"],
            end_lon=list_available_chargers["lon"],
            dist_correction_factor=self.dist_correction_factor,
            dist_func=self.dist_func,
        )
        argmin_idx = np.argmin(dist_to_supercharger)
        min_dist_to_charger = min(dist_to_supercharger)
        closest_charger_idx = list_available_chargers.index[argmin_idx]
        return closest_charger_idx, min_dist_to_charger
    
    def spatial_plot(self, lat, lon, file_name):
        # Create a map of Chicago
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='merc', llcrnrlat=41.4, urcrnrlat=42.2, llcrnrlon=-88.1, urcrnrlon=-87.4, resolution='h')
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='lightgray',lake_color='aqua')
        x, y = m(lon, lat)
        m.scatter(x, y, marker='o', color='g', alpha=0.7)

        # Add title and show plot
        plot_file = os.path.join(self.plot_dir, file_name)
        plt.savefig(plot_file)
        plt.clf()
        plt.close()