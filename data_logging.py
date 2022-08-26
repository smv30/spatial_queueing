import pandas as pd


class DataLogging(object):
    def __init__(self):
        self.list_soc = []
        self.n_cars_idle = []
        self.n_cars_driving_with_passenger = []
        self.n_cars_driving_without_passenger = []
        self.n_cars_driving_to_charger = []
        self.n_cars_charging = []
        self.time_of_logging = []
        self.avg_soc = []
        self.stdev_soc = []
        self.norm_soc = []
        self.norm_perp_soc = []
        self.n_cars_with_100_percent_charge_driving = []
        self.n_cars_with_100_percent_charge_idle = []
        self.n_cars_with_99_percent_charge = []
        self.n_cars_with_98_percent_charge = []
        self.n_cars_with_97_percent_charge = []
        self.n_cars_with_96_percent_charge = []
        self.n_cars_with_95_percent_charge = []
        self.n_cars_rest_of_them = []

    def update_data(self, curr_list_soc, n_cars_idle, n_cars_driving_with_passenger, n_cars_driving_without_passenger,
                    n_cars_driving_to_charger, n_cars_charging, avg_soc, stdev_soc, time_of_logging,
                    norm_soc, norm_perp_soc):
        self.list_soc.append(curr_list_soc)
        self.n_cars_idle.append(n_cars_idle)
        self.n_cars_charging.append(n_cars_charging)
        self.n_cars_driving_to_charger.append(n_cars_driving_to_charger)
        self.n_cars_driving_without_passenger.append(n_cars_driving_without_passenger)
        self.n_cars_driving_with_passenger.append(n_cars_driving_with_passenger)
        self.time_of_logging.append(time_of_logging)
        self.avg_soc.append(avg_soc)
        self.stdev_soc.append(stdev_soc)
        self.norm_soc.append(norm_soc)
        self.norm_perp_soc.append(norm_perp_soc)

    def update_dist_array_q(self, n_cars_with_100_percent_charge_driving,
                            n_cars_with_100_percent_charge_idle,
                            n_cars_with_99_percent_charge, n_cars_with_98_percent_charge,
                            n_cars_with_97_percent_charge, n_cars_with_96_percent_charge,
                            n_cars_with_95_percent_charge, n_cars_rest_of_them):
        self.n_cars_with_100_percent_charge_driving.append(n_cars_with_100_percent_charge_driving)
        self.n_cars_with_100_percent_charge_idle.append(n_cars_with_100_percent_charge_idle)
        self.n_cars_with_99_percent_charge.append(n_cars_with_99_percent_charge)
        self.n_cars_with_98_percent_charge.append(n_cars_with_98_percent_charge)
        self.n_cars_with_97_percent_charge.append(n_cars_with_97_percent_charge)
        self.n_cars_with_96_percent_charge.append(n_cars_with_96_percent_charge)
        self.n_cars_with_95_percent_charge.append(n_cars_with_95_percent_charge)
        self.n_cars_rest_of_them.append(n_cars_rest_of_them)

    def dist_array_q_to_dict(self):
        return pd.DataFrame({
            "n_cars_with_100_percent_charge_driving": self.n_cars_with_100_percent_charge_driving,
            "n_cars_with_100_percent_charge_idle": self.n_cars_with_100_percent_charge_idle,
            "n_cars_with_99_percent_charge": self.n_cars_with_99_percent_charge,
            "n_cars_with_98_percent_charge": self.n_cars_with_98_percent_charge,
            "n_cars_with_97_percent_charge": self.n_cars_with_97_percent_charge,
            "n_cars_with_96_percent_charge": self.n_cars_with_96_percent_charge,
            "n_cars_with_95_percent_charge": self.n_cars_with_95_percent_charge,
            "n_cars_rest_of_them": self.n_cars_rest_of_them,
        })

    def demand_curve_to_dict(self):
        return pd.DataFrame({
            "time": self.time_of_logging,
            "idle": self.n_cars_idle,
            "driving_to_charger": self.n_cars_driving_to_charger,
            "charging": self.n_cars_charging,
            "driving_with_passenger": self.n_cars_driving_with_passenger,
            "driving_without_passenger": self.n_cars_driving_without_passenger,
            "avg_soc": self.avg_soc,
            "stdev_soc": self.stdev_soc
        })
