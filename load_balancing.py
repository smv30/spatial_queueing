import numpy as np
from sim_metadata import SimMetaData
import math


def power_of_d(array_q, d):
    array_prob = np.ones(len(array_q)) / len(array_q)
    array_queues_idx = SimMetaData.random_seed_gen.choice(a=len(array_q),
                                                          size=d,
                                                          replace=False,
                                                          p=array_prob
                                                          )
    min_queue_len_idx_tmp = array_q[array_queues_idx].argmin()
    min_queue_len_idx = array_queues_idx[min_queue_len_idx_tmp]
    return min_queue_len_idx


def main(sim_duration, n_queues, arrival_rate, service_rate, buffer, d, quiet_sim=False):
    curr_time = 0
    n_arrivals = 0
    avg_queue_levels = np.zeros(buffer + 1)
    array_q = np.ones(n_queues) * 3
    while curr_time <= sim_duration:
        non_zero_queues = sum(array_q > 0)
        total_rate = non_zero_queues * service_rate + arrival_rate
        array_service_rate = np.ones(n_queues) * service_rate * (array_q > 0).astype(int)
        array_prob_tmp = np.concatenate((array_service_rate, [arrival_rate]))
        array_prob = array_prob_tmp / total_rate
        inter_arrival_time = SimMetaData.random_seed_gen.exponential(total_rate)
        curr_time = curr_time + inter_arrival_time
        next_event = SimMetaData.random_seed_gen.choice(n_queues + 1, p=array_prob)
        if next_event != n_queues:
            array_q[next_event] = max(array_q[next_event] - 1, 0)
            if not quiet_sim:
                print(f"Queue {next_event} is served, updated queue length = {array_q[next_event]}")
        else:
            queue_idx = power_of_d(array_q, d)
            if not quiet_sim:
                print(f"Arrival is routed to queue {queue_idx} with queue length = {array_q[queue_idx]}")
            array_q[queue_idx] = min(array_q[queue_idx] + 1, buffer)
            if curr_time >= sim_duration * 0.3:
                n_arrivals = n_arrivals + 1
                queue_levels = [sum(array_q == k) for k in range(buffer + 1)]
                avg_queue_levels = (avg_queue_levels * n_arrivals + queue_levels) / (n_arrivals + 1)
    print(avg_queue_levels)


if __name__ == "__main__":
    n = 500
    alpha = 0.4
    main(sim_duration=5 * 10 ** 5,
         n_queues=n,
         arrival_rate=n - n ** (1 - alpha),
         service_rate=1,
         buffer=10,
         d=int(n ** (alpha / 4) * math.log(n))
         )
