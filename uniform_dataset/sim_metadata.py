import numpy as np
from enum import Enum, auto


class SimMetaData(object):
    avg_vel_mph = 20
    consumption_kwhpmi = 0.25
    pack_size_kwh = 40
    charge_rate_kw = 20
    min_allowed_soc = 0.05
    max_lat = 10
    max_lon = 10
    quiet_sim = True    # if False, it will print everything (make sure every value in main is small)
    results_folder = "simulation_results"
    random_seed_gen = np.random.default_rng(2023)
    save_results = True    # able to plot
    freq_of_data_logging_min = 5
    random_data = False


class ChargingAlgo(Enum):
    CHARGE_AFTER_TRIP_END = auto()


class ChargingAlgoParams(object):
    charging_soc_threshold = 0.9
    higher_soc_threshold = 1
    send_all_idle_cars_to_charge = True
    infinite_chargers = False


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

