import pandas as pd


class DataLogging(object):
    def __init__(self):
        self.list_soc = []
        self.n_cars_idle = []
        self.n_cars_driving_with_passenger = []
        self.n_cars_driving_without_passenger = []
        self.n_cars_driving_to_charger = []
        self.n_cars_charging = []
        self.n_cars_waiting_for_charger = []
        self.time_of_logging = []
        self.avg_soc = []
        self.stdev_soc = []

    def update_data(self, curr_list_soc, n_cars_idle, n_cars_driving_with_passenger, n_cars_driving_without_passenger,
                    n_cars_driving_to_charger, n_cars_charging, n_cars_waiting_for_charger, avg_soc, stdev_soc, time_of_logging):
        self.list_soc.append(curr_list_soc)
        self.n_cars_idle.append(n_cars_idle)
        self.n_cars_charging.append(n_cars_charging)
        self.n_cars_driving_to_charger.append(n_cars_driving_to_charger)
        self.n_cars_driving_without_passenger.append(n_cars_driving_without_passenger)
        self.n_cars_driving_with_passenger.append(n_cars_driving_with_passenger)
        self.n_cars_waiting_for_charger.append(n_cars_waiting_for_charger)
        self.time_of_logging.append(time_of_logging)
        self.avg_soc.append(avg_soc)
        self.stdev_soc.append(stdev_soc)

    def demand_curve_to_dict(self):
        return pd.DataFrame({
            "time": self.time_of_logging,
            "idle": self.n_cars_idle,
            "driving_to_charger": self.n_cars_driving_to_charger,
            "charging": self.n_cars_charging,
            "driving_with_passenger": self.n_cars_driving_with_passenger,
            "driving_without_passenger": self.n_cars_driving_without_passenger,
            "waiting_for_charger": self.n_cars_waiting_for_charger,
            "avg_soc": self.avg_soc,
            "stdev_soc": self.stdev_soc
        })
