import simpy
from car import Car
from fleet_manager import FleetManager
from chargers import SuperCharger


def run_simulation(
        sim_duration,
        n_cars,
        arrival_rate_pmin,
        n_chargers,
        n_posts,
        renege_time_min=None
):
    env = simpy.Environment()

    # Initialize all the supercharging stations
    list_chargers = []
    for charger_idx in range(n_chargers):
        charger = SuperCharger(idx=charger_idx,
                               n_posts=n_posts)
        list_chargers.append(charger)

    # Initializing all the cars
    car_tracker = []
    for car_id in range(n_cars):
        car = Car(car_id=car_id,
                  env=env,
                  list_chargers=list_chargers)
        car_tracker.append(car)

    fleet_manager = FleetManager(arrival_rate_pmin=arrival_rate_pmin,
                                 env=env,
                                 car_tracker=car_tracker,
                                 n_cars=n_cars,
                                 renege_time_min=renege_time_min,
                                 list_chargers=list_chargers)
    env.process(fleet_manager.match_trips())
    env.run(until=sim_duration)


if __name__ == "__main__":
    run_simulation(sim_duration=500,
                   n_cars=10,
                   arrival_rate_pmin=1 / 10,
                   n_chargers=10,
                   n_posts=1,
                   renege_time_min=1
                   )

