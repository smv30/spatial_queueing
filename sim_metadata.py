import numpy as np
from enum import Enum, auto


class SimMetaData(object):
    avg_vel_mph = 30
    consumption_kwhpmi = 0.25
    pack_size_kwh = 50
    charge_rate_kw = 20
    min_allowed_soc = 0.05
    quiet_sim = True  # if False, it will print everything (make sure every value in main is small)
    results_folder = "simulation_results"
    random_seed = 2024
    random_seed_gen = np.random.default_rng(2021)
    save_results = True  # able to plot
    freq_of_data_logging_min = 0.01
    demand_curve_res_min = 1
    test = True


class MatchingAlgo(Enum):
    POWER_OF_D = auto()
    CLOSEST_AVAILABLE_DISPATCH = auto()
    POWER_OF_RADIUS = auto()


class AvailableCarsForMatching(Enum):
    ONLY_IDLE = auto()
    IDLE_AND_CHARGING = auto()
    IDLE_CHARGING_DRIVING_TO_CHARGER = auto()


class PickupThresholdType(Enum):
    PERCENT_THRESHOLD = auto()
    CONSTANT_THRESHOLD = auto()
    NO_THRESHOLD = auto()
    BOTH_PERCENT_AND_CONSTANT = auto()
    EITHER_PERCENT_OR_CONSTANT = auto()
    MIN_AVAILABLE_CARS_PERCENT = auto()


class PickupThresholdMatchingParams(object):
    threshold_percent = 0.8
    threshold_min = 45
    min_available_cars_percent = 0.1


class AdaptivePowerOfDParams(object):
    threshold_percent_of_cars_idling = 0.05
    n_trips_before_updating_d = 1000
    adaptive_d = False


class ChargingAlgoParams(object):
    safety_factor_to_reach_closest_charger = 1.5
    infinite_chargers = False
    start_of_the_night = 23  # Should be (0, 24] (use 24 if you want to use 0)
    end_of_the_night = 6
    n_cars_driving_to_charger_discounter = 0.5


class ChargingAlgo(Enum):
    CHARGE_ALL_IDLE_CARS = auto()
    CHARGE_ALL_IDLE_CARS_AT_NIGHT = auto()


class CarState(Enum):
    DRIVING_WITH_PASSENGER = auto()
    DRIVING_WITHOUT_PASSENGER = auto()
    CHARGING = auto()
    WAITING_FOR_CHARGER = auto()
    DRIVING_TO_CHARGER = auto()
    IDLE = auto()


class TripState(Enum):
    UNAVAILABLE = auto()
    WAITING = auto()
    MATCHED = auto()
    RENEGED = auto()


class ChargerState(Enum):
    AVAILABLE = auto()
    BUSY = auto()


class Dataset(Enum):
    NYTAXI = auto()
    OLD_NYTAXI = auto()
    RANDOMLYGENERATED = auto()
    CHICAGO = auto()


class DatasetParams(object):
    percent_of_trips_filtered = 0.6
    percentile_lat_lon = 99.9
    longitude_range_min = -87.7
    latitude_range_min = 41.85
    longitude_range_max = -87.6
    latitude_range_max = 41.95
    delta_latitude = 1
    delta_longitude = 1
    uniform_locations = False


class Initialize(Enum):
    RANDOM_UNIFORM = auto()
    RANDOM_PICKUP = auto()
    RANDOM_DESTINATION = auto()
    EQUAL_TO_INPUT = auto()


class DistFunc(Enum):
    HAVERSINE = auto()
    MANHATTAN = auto()
