import matplotlib.pyplot as plt
import numpy as np
import os


def fleet_can_pod():
    cust_per_min = [21.2, 31.8, 42.3, 52.9, 63.5]
    tesla_pod = [900, 1366, 1840, 2295, 2727]
    tesla_can = [1075, 1600, 2138, 2647, 3140]
    plt.plot(cust_per_min, tesla_pod, label="Po2", linestyle="solid")
    plt.plot(cust_per_min, tesla_pod, label="Po2", linestyle="solid")


def load_balancing(ev_type):
    plot_dir = ("/Users/sushilvarma/Library/Mobile Documents/com~apple~CloudDocs/" +
                "Academics/Research/EV/MS_Second_Round_Plots_Final/final_plots"
                )
    if ev_type == "Nissan Leaf":
        pickup = [2.67, 3.18, 3.53, 3.82, 4.13, 4.41, 4.70, 4.95, 5.11, 5.36]
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        workload = np.array([0.815, 0.868, 0.885, 0.901, 0.905, 0.905, 0.903, 0.900, 0.899, 0.891]) * 100
    elif ev_type == "Tesla Model 3":
        pickup = [3.27, 3.53, 3.96, 4.51, 5.19, 5.91, 6.56, 6.91, 7.19, 7.36]
        d = [1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6]
        workload = np.array([0.907, 0.918, 0.926, 0.916, 0.903, 0.890, 0.885, 0.882, 0.879, 0.877]) * 100
    else:
        raise ValueError("No such EV exists")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(d, workload, linewidth=3, linestyle="solid", label="Workload %", marker="o", markersize=10)
    ax2.plot(d, pickup, linewidth=3, linestyle="dashed", label="Pickup (min)", marker="v", markersize=10)
    ax1.set_xlabel("d in Power-of-d")
    ax1.set_ylabel("Workload Percentage")
    ax2.set_ylabel("Pickup Time (min)")
    plt.title(f"{ev_type}")
    plot_file = os.path.join(plot_dir, f"{ev_type}_load_balancing.png")
    plt.savefig(plot_file)
    plt.clf()


if __name__ == "__main__":
    load_balancing(ev_type="Tesla Model 3")



