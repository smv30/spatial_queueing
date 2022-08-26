import numpy as np
import matplotlib.pyplot as plt


def get_derivatives(s, b, arrival_rate, d, n):
    delta_s = np.zeros(b+2)
    s = np.concatenate((s, [0]))
    for i in range(1, b+1):
        delta_s[i] = arrival_rate * ((s[i-1] / n) ** d - (s[i] / n) ** d) - s[i] + s[i + 1]
    return delta_s / n


def twod_data(b, arrival_rate, d, n, n_points):
    res = n / n_points
    list_s1 = np.arange(0, n, res)
    list_s2 = np.arange(0, n, res)

    val = np.zeros((n_points, n_points, 2))

    for i in range(n_points):
        for j in range(n_points):
            if list_s1[i] >= list_s2[j]:
                s = np.concatenate(([n, list_s1[i], list_s2[j]], np.zeros(b-2)))
                val[i, j, 0] = get_derivatives(s, b, arrival_rate, d, n)[1]
                val[i, j, 1] = get_derivatives(s, b, arrival_rate, d, n)[2]
            else:
                val[i, j, 0] = 0
                val[i, j, 1] = 0
    return val


def twod_plot(val, n, n_points):
    res = n / n_points
    list_s1 = np.arange(0, n, res) / n
    list_s2 = np.arange(0, n, res) / n
    x, y = np.meshgrid(list_s2, list_s1)
    plt.quiver(y, x, val[:, :, 0], val[:, :, 1])
    plt.xlabel("s1/n")
    plt.ylabel("s2/n")
    plt.show()


def main(b, arrival_rate, d, n, n_points):
    val = twod_data(b, arrival_rate, d, n, n_points)
    twod_plot(val, n, n_points)


if __name__ == "__main__":
    input_n = 400
    alpha = 0.4
    input_arrival_rate = input_n - input_n ** (1 - alpha)
    input_d = input_n ** alpha
    main(b=2,
         arrival_rate=input_arrival_rate,
         d=input_d,
         n=input_n,
         n_points=50)


