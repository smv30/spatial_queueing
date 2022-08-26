import numpy as np
import matplotlib.pyplot as plt


def calc_derivatives(s_c, s_b, r, arrival_rate, d, n):
    d_s_0_c_dt = -s_c[0] / r + s_b[0]
    d_s_1_c_dt = -s_c[1] / r + s_c[0] / r + s_b[1] - arrival_rate * (
        (s_c[1] / (n - s_b[1])) ** d - (s_c[0] / (n - s_b[1])) ** d
    )
    d_s_2_c_dt = s_b[1] - arrival_rate * (1 - (s_c[0] / (n - s_b[1])) ** d)

    d_s_1_b_dt = -s_b[0] + arrival_rate * ((s_c[1] / (n - s_b[1])) ** d - (s_c[0] / (n - s_b[1])) ** d)
    d_s_2_b_dt = -s_b[1] + arrival_rate * (1 - (s_c[0] / (n - s_b[1])) ** d)

    d_s_c_dt = np.array([d_s_0_c_dt, d_s_1_c_dt, d_s_2_c_dt])
    d_s_b_dt = np.array([d_s_1_b_dt, d_s_2_b_dt])

    return d_s_c_dt / 10, d_s_b_dt / 10


def fluid_model(n, s_c_init, s_b_init, sim_duration, r, arrival_rate, d, gamma):
    # Calculate derivatives
    s_c = s_c_init
    s_b = s_b_init
    list_s_c_0 = []
    list_s_c_1 = []
    list_s_c_2 = []
    list_s_b_1 = []
    list_s_b_2 = []
    for k in range(sim_duration):
        d_s_c_dt, d_s_b_dt = calc_derivatives(s_c=s_c,
                                              s_b=s_b,
                                              r=r,
                                              arrival_rate=arrival_rate,
                                              d=d,
                                              n=n)
        s_c = np.minimum(np.maximum(s_c + d_s_c_dt, 0.1), n)
        s_b = np.minimum(np.maximum(s_b + d_s_b_dt, 0.1), n)
        list_s_c_0.append(s_c[0])
        list_s_c_1.append(s_c[1])
        list_s_c_2.append(s_c[2])
        list_s_b_1.append(s_b[0])
        list_s_b_2.append(s_b[1])
    print(d_s_c_dt)
    print(d_s_b_dt)
    plotting(list_s_c_0, list_s_c_1, list_s_c_2, list_s_b_1, list_s_b_2, n, r, gamma)
    return list_s_c_0, list_s_c_1, list_s_c_2, list_s_b_1, list_s_b_2


def plotting(list_s_c_0, list_s_c_1, list_s_c_2, list_s_b_1, list_s_b_2, n, r, gamma):
    plt.plot(list_s_b_2, list_s_b_1)
    plt.xlabel("s_b_2")
    plt.ylabel("s_b_1")
    plt.title(f"n={n}, r={r}, gamma={gamma}, \n d=n^gamma, lambda=n/(1+r) - n^(1-gamma)")
    plt.show()
    plt.clf()

    plt.plot(list_s_b_2, label="s_b_2")
    plt.plot(list_s_c_1, label="s_c_1")
    plt.plot(list_s_b_1, label="s_b_1")
    plt.plot(list_s_c_0, label="s_c_0")
    plt.xlabel("time")
    plt.ylabel("s")
    plt.legend()
    plt.title(f"n={n}, r={r}, gamma={gamma}, \n d=n^gamma, lambda=n/(1+r) - n^(1-gamma)")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    input_gamma = 0.4
    input_n = 1000
    input_r = 1
    input_arrival_rate = input_n / (1 + input_r) - input_n ** (1 - input_gamma)
    input_d = input_n ** (input_gamma / 2)
    input_s_c = np.array([0, input_n, input_n/2])
    input_s_b = np.array([0, 0])
    out_list_s_c_0, out_list_s_c_1, out_list_s_c_2, out_list_s_b_1, out_list_s_b_2 = (
        fluid_model(n=input_n,
                    s_c_init=input_s_c,
                    s_b_init=input_s_b,
                    sim_duration=200,
                    r=input_r,
                    arrival_rate=input_arrival_rate,
                    d=input_d,
                    gamma=input_gamma)
    )

