import numpy as np
from enum import Enum, auto


class SimMetaData(object):
    avg_vel_mph = 15
    consumption_kwhpmi = 0.25
    pack_size_kwh = 50
    charge_rate_kw = 50
    min_allowed_soc = 0.05
    max_lat = 10
    max_lon = 10
    n_charger_loc = 5
    n_posts = 8
    quiet_sim = False
    results_folder = "simulation_results"
    random_seed_gen = np.random.default_rng(2022)
    save_results = True


class ChargingAlgoParams(object):
    lower_soc_threshold = 0.2
    higher_soc_threshold = 1


class MatchingAlgoParams(object):
    d = 3


class CarState(Enum):
    DRIVING_WITH_PASSENGER = auto()
    DRIVING_WITHOUT_PASSENGER = auto()
    CHARGING = auto()
    DRIVING_TO_CHARGER = auto()
    IDLE = auto()


class TripState(Enum):
    UNAVAILABLE = auto()
    WAITING = auto()
    MATCHED = auto()
    RENEGED = auto()

