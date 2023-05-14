from sim_metadata import SimMetaData, MarkovianModelParams


def single_two_sided_queue_sim(sim_duration, arrival_rate_pmin, n):
    curr_time = 0
    q = 0
    p = 0
    total_met_demand = 0
    total_demand = 0

    s = (
            100 * SimMetaData.charge_rate_kw / SimMetaData.pack_size_kwh / 60
            / MarkovianModelParams.charge_in_one_transition
    )

    r = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
    adj_arrival_rate_pmin = arrival_rate_pmin * (r + 1) / r / n

    trip_time_min = SimMetaData.AVG_TRIP_DIST_PER_MILE_SQ * SimMetaData.max_lat / SimMetaData.avg_vel_mph * 60
    pickup_time_min = MarkovianModelParams.pickup_time_const / n ** 0.5
    mu = 1 / (trip_time_min + pickup_time_min)

    charge_used_in_one_service = (
            100 * SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / mu / 60
            / SimMetaData.pack_size_kwh
    )

    while curr_time <= sim_duration:
        if q == 0:
            time = SimMetaData.random_seed_gen.exponential(s + adj_arrival_rate_pmin)
            uniform_random_number = SimMetaData.random_seed_gen.uniform(0, s + adj_arrival_rate_pmin)
            if uniform_random_number <= s:
                p = min(p + MarkovianModelParams.charge_in_one_transition, 100)
            else:
                total_demand += 1
                if p >= charge_used_in_one_service:
                    q += 1
                    total_met_demand += 1
        else:
            time = SimMetaData.random_seed_gen.exponential(mu)
            p = p - charge_used_in_one_service
            q = 0

        curr_time = curr_time + time
    print(100 - total_met_demand / total_demand * 100)


if __name__ == "__main__":
    list_n = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10, 10 ** 11,
              10 ** 12, 10 ** 13, 10 ** 14, 10 ** 15, 10 ** 16]
    gamma = 0.3
    r = SimMetaData.consumption_kwhpmi * SimMetaData.avg_vel_mph / SimMetaData.charge_rate_kw
    trip_time_min = SimMetaData.AVG_TRIP_DIST_PER_MILE_SQ * SimMetaData.max_lat / SimMetaData.avg_vel_mph * 60
    for n in list_n:
        arrival_rate_pmin = n / (r + 1) / trip_time_min - n ** (1 - gamma) / 4
        single_two_sided_queue_sim(
            sim_duration=100000,
            arrival_rate_pmin=arrival_rate_pmin,
            n=n
        )

