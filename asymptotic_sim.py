import cProfile
from sim_metadata import SimMetaData, MatchingAlgo
from main import run_simulation
from markovian_model import markovian_sim


def asymptotic_sim(list_n=[10], alpha=0.5, c1=4, beta=1, c2=2, d=2, results_folder="simulation_results/test",
                   bool_markovian_sim=False):
    pd_kpi = None
    for n in list_n:
        consumption_by_charge_rate = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
        # Defining the fleet size
        fleet_size = int((consumption_by_charge_rate + 1) * n + c1 * n ** alpha)
        if d == "sqrtn":
            actual_d = int(max((fleet_size - n) / 2, 2))
        else:
            actual_d = d

        # Defining the number of charging stations
        n_chargers = int(consumption_by_charge_rate * n + c2 * n ** beta)
        n_chargers = 1

        # Defining the arrival rate
        avg_trip_dist = 0.5214 * SimMetaData.max_lon
        avg_trip_time_min = avg_trip_dist / SimMetaData.avg_vel_mph * 60
        arrival_rate_pmin = n / avg_trip_time_min
        if not markovian_sim:
            kpi = run_simulation(sim_duration=3000,
                                 n_cars=fleet_size,
                                 arrival_rate_pmin=arrival_rate_pmin,
                                 n_chargers=n_chargers,
                                 n_posts=10,
                                 matching_algo=MatchingAlgo.POWER_OF_D_IDLE_OR_CHARGING,
                                 renege_time_min=1,
                                 results_folder=f"{results_folder}/scenario_{n}",
                                 d=actual_d
                                 )
        else:
            kpi = markovian_sim(n_cars=fleet_size,
                                sim_duration=3000,
                                arrival_rate=arrival_rate_pmin,
                                n_chargers=n_chargers,
                                n_posts=10,
                                results_folder=results_folder
                                )
        kpi["n"] = n
        kpi["alpha"] = alpha
        kpi["c1"] = c1
        kpi["beta"] = beta
        kpi["c2"] = c2
        if pd_kpi is None:
            pd_kpi = kpi
        else:
            pd_kpi = pd_kpi.append(kpi, ignore_index=True)

        pd_kpi.to_csv(f"{results_folder}/consolidated_kpi.csv")


if __name__ == "__main__":
    profile = False
    if profile:
        cProfile.run('asymptotic_sim()')
    else:
        list_alpha = [0.5, 0.4, 0.6]
        for alpha in list_alpha:
            asymptotic_sim(list_n=[10, 30, 50, 100, 200, 300, 500, 1000],
                           alpha=alpha,
                           c1=4,
                           beta=1,
                           c2=2,
                           d=2,
                           results_folder=f"simulation_results/Jan_19/markovian_model_alpha_{alpha}",
                           bool_markovian_sim=True
                           )
