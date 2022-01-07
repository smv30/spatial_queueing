from enum import Enum, auto


class SimMetaData(object):
    avg_vel = 1
    consumption_kwhpmi = 4
    pack_size_kwh = 50
    charge_rate_kw = 50
    min_allowed_soc = 0.05
    max_lat = 1
    max_lon = 1
    n_charger_loc = 5
    n_posts = 8


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

