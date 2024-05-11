class ElectricVehicle:
    def __init__(self, name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc):
        self.name = name
        self.pack_size_kwh = pack_size_kwh
        self.consumption_kwhpmi = consumption_kwhpmi
        self.battery_degradation_perc = battery_degradation_perc


class ElectricVehicleDatabase:
    def __init__(self):
        self.database = {
            "Tesla Model 3": ElectricVehicle("Tesla Model 3", 57.5, 0.23, 0.1),
            "Nissan Leaf": ElectricVehicle("Nissan Leaf", 39, 0.27, 0.1),
            "Mustang Mach-E ER AWD": ElectricVehicle("Mustang Mach-E ER AWD", 72, 0.3, 0.1),
            "Tesla Model S Dual Motor": ElectricVehicle("Tesla Model S Dual Motor", 95, 0.27, 0.1)
        }  # Dictionary to store EV information

    def add_vehicle(self, name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc):
        self.database[name] = ElectricVehicle(name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc)

    def get_vehicle_info(self, name):
        return self.database.get(name, None)


