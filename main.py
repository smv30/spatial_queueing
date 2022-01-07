import simpy
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger


def run_simulation(
        n_cars,
        arrival_rate,
        n_chargers,
        n_posts,
        renege_time=None,
):
    env = simpy.Environment()

    # Initializing all the cars
    car_tracker = []
    for car_id in range(n_cars):
        car = Car(car_id=car_id,
                  env=env)
        car_tracker.append(car)

    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts)
        list_chargers.append(charger)

    fleet_manager = FleetManager(arrival_rate=arrival_rate,
                                 env=env,
                                 car_tracker=car_tracker,
                                 n_cars=n_cars,
                                 renege_time=renege_time,
                                 list_chargers=list_chargers)
    env.process(fleet_manager.match_trips())
    env.run(until=20)


if __name__ == "__main__":
    run_simulation(10, 2, 10, 1, 1)

