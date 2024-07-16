class ElectricVehicle:
    def __init__(self, name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc):
        self.name = name
        self.pack_size_kwh = pack_size_kwh
        self.consumption_kwhpmi = consumption_kwhpmi
        self.battery_degradation_perc = battery_degradation_perc


class ElectricVehicleDatabase:
    def __init__(self):
        self.database = {
            "Tesla_Model_3": ElectricVehicle("Tesla_Model_3", 57.5, 0.23, 0.1), # 200 miles
            "Nissan_Leaf": ElectricVehicle("Nissan_Leaf", 39, 0.27, 0.1), # 130 miles
            "Hybrid": ElectricVehicle("Hybrid", 50, 0.25, 0.1), # 180 miles
            "Mustang_Mach_E_ER_AWD": ElectricVehicle("Mustang_Mach_E_ER_AWD", 72, 0.25, 0.1), # 260 miles
            "Tesla_Model_S_Dual_Motor": ElectricVehicle("Tesla_Model_S_Dual_Motor", 95, 0.27, 0.1)
        }  # Dictionary to store EV information

    def add_vehicle(self, name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc):
        self.database[name] = ElectricVehicle(name, pack_size_kwh, consumption_kwhpmi, battery_degradation_perc)

    def get_vehicle_info(self, name):
        return self.database.get(name, None)


