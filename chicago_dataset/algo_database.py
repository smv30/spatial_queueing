from sim_metadata import MatchingAlgo, AvailableCarsForMatching, ChargingAlgo


class AlgorithmConfig:
    def __init__(self, name, matching_algo, available_cars_for_matching, charging_algo):
        self.name = name
        self.matching_algo = matching_algo
        self.available_cars_for_matching = available_cars_for_matching
        self.charging_algo = charging_algo

class AlgorithmDatabase:
    def __init__(self):
        self.database = {
            "POD": AlgorithmConfig(
                "Power of d with interrupt charging (POD)",
                MatchingAlgo.POWER_OF_D.value,
                AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS.value
            ),
            "CAD": AlgorithmConfig(
                "Closest available dispatch with interrupt charging (CAD)",
                MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value,
                AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS.value
            ),
            "POTP": AlgorithmConfig(
                "Power of radius with interrupt charging (POTP)",
                MatchingAlgo.POWER_OF_RADIUS.value,
                AvailableCarsForMatching.IDLE_CHARGING_DRIVING_TO_CHARGER.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS.value
            ),
            "CAN_POD": AlgorithmConfig(
                "Charge at Night with Power of d (CAN_POD)",
                MatchingAlgo.POWER_OF_D.value,
                AvailableCarsForMatching.ONLY_IDLE.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value
            ),
            "CAN_CAD": AlgorithmConfig(
                "Charge at Night with Closest Available Dispatch (CAN_CAD)",
                MatchingAlgo.CLOSEST_AVAILABLE_DISPATCH.value,
                AvailableCarsForMatching.ONLY_IDLE.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT.value
            ),
            "CAN_R_POD": AlgorithmConfig(
                "Charge at Night with Relocations and Power of d (CAN_R_POD)",
                MatchingAlgo.POWER_OF_D.value,
                AvailableCarsForMatching.IDLE_AND_RELOCATING.value,
                ChargingAlgo.CHARGE_ALL_IDLE_CARS_AT_NIGHT_WITH_RELOCATION.value
            ),
            "CAN_R_POD_N": AlgorithmConfig(
                "Charge at Night with Relocations and Power of d at Night (CAN_R_POD_N)",
                MatchingAlgo.POWER_OF_D.value,
                None,
                ChargingAlgo.CAN_WITH_IC_AT_NIGHT.value
            ),
        }

    def add_algorithm(self, key, name, matching_algo, available_cars_for_matching, charging_algo):
        self.database[key] = AlgorithmConfig(
            name, matching_algo, available_cars_for_matching, charging_algo
        )

    def get_algorithm(self, key):
        return self.database.get(key)

# --- usage example ---
if __name__ == "__main__":
    alg_db = AlgorithmDatabase()
    pod = alg_db.get_algorithm("POD")
    print(pod.name, pod.matching_algo, pod.charging_algo)
