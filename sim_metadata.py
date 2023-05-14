import numpy as np
from enum import Enum, auto


class SimMetaData(object):
    avg_vel_mph = 15
    consumption_kwhpmi = 0.25
    pack_size_kwh = 30
    charge_rate_kw = 10
    min_allowed_soc = 0.05
    max_lat = 15
    max_lon = 15
    quiet_sim = True
    results_folder = "simulation_results"
    random_seed_gen = np.random.default_rng(2022)
    save_results = True
    freq_of_data_logging_min = 5


class ChargingAlgoParams(object):
    lower_soc_threshold = 0.95
    higher_soc_threshold = 1
    send_all_idle_cars_to_charge = True
    infinite_chargers = False


class MatchingAlgo(Enum):
    POWER_OF_D_IDLE = auto()
    POWER_OF_D_IDLE_OR_CHARGING = auto()


class CarState(Enum):
    DRIVING_WITH_PASSENGER = auto()
    DRIVING_WITHOUT_PASSENGER = auto()
    CHARGING = auto()
    DRIVING_TO_CHARGER = auto()
    IDLE = auto()


class ChargerState(Enum):
    AVAILABLE = auto()
    BUSY = auto()


class TripState(Enum):
    UNAVAILABLE = auto()
    WAITING = auto()
    MATCHED = auto()
    RENEGED = auto()

