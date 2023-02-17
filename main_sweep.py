from main import run_simulation
from sim_metadata import MatchingAlgo
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import os

sim_duration = [200]
n_cars = [10, 20, 30, 40]
arrival_rate_pmin = [1 / 2]
n_chargers = [5, 10, 20, 25]
n_posts = [1]
renege_time_min = [1]
infinite_chargers = [False]

Z = np.array(list(itertools.product(sim_duration, n_cars)))
print(Z)

dt = [('sim_duration', '<i8'), ('n_cars', '<i8'), ('arrival_rate_pmin', '<f8'), ('n_chargers', '<i8'),
      ('n_posts', '<i8'), ('renege_time_min', '<i8'), ('infinite_chargers', '<?')]
# change infinite_charger datatype to boolean
Z = np.array(list(itertools.product(sim_duration, n_cars, arrival_rate_pmin, n_chargers, n_posts, renege_time_min,
                                    infinite_chargers)), dtype=dt)
print(Z["sim_duration"])

today = datetime.now()
curr_date_and_time = today.strftime("%b_%d_%Y_%H_%M_%S")
top_level_dir = os.path.join("simulation_results/sweep_folder", curr_date_and_time)

consolidated_kpi = run_simulation(sim_duration=Z["sim_duration"][0],
                                  n_cars=Z["n_cars"][0],
                                  arrival_rate_pmin=Z["arrival_rate_pmin"][0],
                                  n_chargers=Z["n_chargers"][0],
                                  n_posts=Z["n_posts"][0],
                                  renege_time_min=Z["renege_time_min"][0],
                                  matching_algo=MatchingAlgo.POWER_OF_D_IDLE.value,
                                  d=2,
                                  infinite_chargers=Z["infinite_chargers"][0],
                                  results_folder=top_level_dir)

for j in range(1, len(Z["sim_duration"])):
    kpi = run_simulation(sim_duration=Z["sim_duration"][j],
                         n_cars=Z["n_cars"][j],
                         arrival_rate_pmin=Z["arrival_rate_pmin"][j],
                         n_chargers=Z["n_chargers"][j],
                         n_posts=Z["n_posts"][j],
                         renege_time_min=Z["renege_time_min"][j],
                         matching_algo=MatchingAlgo.POWER_OF_D_IDLE.value,
                         d=2,
                         infinite_chargers=Z["infinite_chargers"][j],
                         results_folder=top_level_dir)
    # Append kpi to consolidate_kpi dataframe
    consolidated_kpi = pd.concat([consolidated_kpi, kpi], axis=0)

kpi_data_file = os.path.join(top_level_dir, "consolidated_kpi.csv")
consolidated_kpi.to_csv(kpi_data_file)
