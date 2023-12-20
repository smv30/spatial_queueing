import numpy as np
from enum import Enum, auto


class SimMetaData(object):
    avg_vel_mph = 40
    consumption_kwhpmi = 0.25
    pack_size_kwh = 50
    charge_rate_kw = 20
    min_allowed_soc = 0.05
    quiet_sim = True  # if False, it will print everything (make sure every value in main is small)
    results_folder = "simulation_results"
    home_dir = '/Users/chenzhang/Desktop/Georgia Tech/Research/spatial_queueing/spatial_queueing/'
    random_seed_gen = np.random.default_rng(2021)
    save_results = True  # able to plot
    freq_of_data_logging_min = 0.1
    demand_curve_res_min = 1
    percent_of_trips = 0.5
    test = False
    max_lat = 10
    max_lon = 10
    quiet_sim = True
    results_folder = "simulation_results"
    random_seed_gen = np.random.default_rng(2022)
    save_results = True
    freq_of_data_logging_min = 5


class ChargingAlgoParams(object):
    lower_soc_threshold = 0.95
    higher_soc_threshold = 1
    safety_factor_to_reach_closest_charger = 1.5
    send_all_idle_cars_to_charge = True
    infinite_chargers = True


class MatchingAlgo(Enum):
    POWER_OF_D_IDLE = auto()
    POWER_OF_D_IDLE_OR_CHARGING = auto()
    CLOSEST_AVAILABLE_DISPATCH = auto()


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
    RANDOMLYGENERATED = auto()


class DatasetParams(object):
    percentile_lat_lon = 99.9
    longitude_range_min = -74
    latitude_range_min = 40.7
    longitude_range_max = -73.9
    latitude_range_max = 40.9
    delta_latitude = 1
    delta_longitude = 1
