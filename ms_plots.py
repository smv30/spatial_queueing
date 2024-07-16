import matplotlib.pyplot as plt
import numpy as np
import os


def fleet_can_pod(data_type):
    plot_dir = ("/Users/sushilvarma/Library/Mobile Documents/com~apple~CloudDocs/" +
                "Academics/Research/EV/MS_Second_Round_Plots_Final/final_plots"
                )
    if data_type == "Chicago Dataset":
        cust_per_min = [21.2, 31.8, 42.3, 52.9, 63.5]
        tesla_po2_45_min = [865, 1256, 1738, 2166, 2586]
        nissan_po2_45_min = [841, 1232, 1768, 2312, 2818]
        mustang_po2_45_min = [857, 1314, 1741, 2168, 2569]
        tesla_can_po2_no_th = [1075, 1600, 2138, 2647, 3140]
        tesla_can_po2_45_min = [1140, 1704, 2237, 2818, 3374]
        tesla_can_cad_45_min = [1177, 1722, 2266, 2817, 3390]
        mustang_can_po2_45_min = [1088, 1632, 2100, 2691, 3218]
        mustang_can_po2_no_th = [1047, 1565, 2028, 2563, 3084]
        mustang_can_cad_45_min = [1087, 1644, 2116, 2722, 3277]
    elif data_type == "Uniform Origin and Destinations":
        cust_per_min = [21.2, 31.8, 42.3, 52.9, 63.5]
        tesla_pod = [550, 826, 1046, 1300, 1538]
        nissan_pod = [541, 818, 1034, 1273, 1493]
        tesla_can = [706, 877, 1108, 1394, 1672]
        nissan_can = [796, 996, 1245, 1561, 1928]
    else:
        raise ValueError("No such datatype exists")
    plt.plot(cust_per_min, tesla_po2_45_min, "#377eb8",  linewidth=3,
             linestyle="solid", label="Tesla, Po2+45min", marker="o", markersize=8)
    plt.plot(cust_per_min, nissan_po2_45_min, "#ff7f00", linewidth=3,
             linestyle="dashed", label="Nissan Po2+45min", marker="s", markersize=8)
    plt.plot(cust_per_min, mustang_po2_45_min, "#9467BD",  linewidth=3,
             linestyle="dashdot", label="Mustang Po2+45min", marker="*", markersize=12)
    plt.plot(cust_per_min, tesla_can_po2_no_th, "#E6AB02", linewidth=3,
                 linestyle="dotted", label="Tesla CaN+Po2", marker="D", markersize=8)
    plt.plot(cust_per_min, mustang_can_po2_no_th, "#2CA02C", linewidth=3,
                 linestyle=(0, (3, 1, 1, 1)), label="Mustang CaN+Po2", marker="p", markersize=8)
    plt.xlabel("Customers per minute", fontsize=15)
    plt.ylabel("90% Fleet Size", fontsize=15)
    # plt.title(f"{data_type}", fontsize=18)
    plt.legend(fontsize=12)
    plot_file_png = os.path.join(plot_dir, f"2_fleet_{data_type}.png")
    plot_file_eps = os.path.join(plot_dir, f"2_fleet_{data_type}.eps")
    plt.savefig(plot_file_png, bbox_inches='tight')
    plt.savefig(plot_file_eps, bbox_inches='tight')
    plt.clf()


def load_balancing(ev_type, plot_type):
    plot_dir = ("/Users/sushilvarma/Library/Mobile Documents/com~apple~CloudDocs/" +
                "Academics/Research/EV/MS_Second_Round_Plots_Final/final_plots"
                )
    if ev_type == "nissan" and plot_type == "workload":
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        t_30_min = np.array([0.786, 0.854, 0.872, 0.886, 0.895, 0.899, 0.900, 0.900, 0.901, 0.901]) * 100
        t_45_min = np.array([0.815, 0.868, 0.885, 0.901, 0.905, 0.905, 0.903, 0.900, 0.899, 0.891]) * 100
        t_60_min = np.array([0.813, 0.864, 0.879, 0.888, 0.889, 0.891, 0.891, 0.886, 0.873, 0.865]) * 100
        t_infty = np.array([0.813, 0.853, 0.861, 0.865, 0.869, 0.864, 0.863, 0.859, 0.857, 0.853]) * 100
    elif ev_type == "nissan" and plot_type == "pickup_time":
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        t_30_min = [2.47, 2.81, 3.10, 3.33, 3.59, 3.88, 4.08, 4.29, 4.50, 4.68]
        t_45_min = [2.67, 3.18, 3.53, 3.82, 4.13, 4.41, 4.70, 4.95, 5.11, 5.36]
        t_60_min = [2.96, 3.45, 3.89, 4.26, 4.64, 4.88, 5.01, 5.28, 5.61, 5.87]
        t_infty = [2.88, 3.97, 4.54, 4.96, 5.22, 5.51, 5.73, 5.89, 6.06, 6.25]
    elif ev_type == "tesla" and plot_type == "workload":
        d = [1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 6]
        t_30_min = np.array([0.888, 0.899, 0.918, 0.921, 0.917, 0.911, 0.908, 0.907, 0.906]) * 100
        t_45_min = np.array([0.907, 0.918, 0.926, 0.916, 0.903, 0.890, 0.885, 0.882, 0.877]) * 100
        t_60_min = np.array([0.908, 0.916, 0.927, 0.905, 0.892, 0.871, 0.866, 0.862, 0.857]) * 100
        t_infty = np.array([0.903, 0.915, 0.914, 0.902, 0.882, 0.865, 0.861, 0.827, 0.810]) * 100
    elif ev_type == "tesla" and plot_type == "pickup_time":
        d = [1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 6]
        t_30_min = [3.17, 3.27, 3.55, 3.81, 4.37, 4.82, 5.28, 5.44, 5.68]
        t_45_min = [3.27, 3.53, 3.96, 4.51, 5.19, 5.91, 6.56, 6.91, 7.36]
        t_60_min = [3.52, 3.72, 3.94, 4.99, 5.63, 6.75, 7.51, 7.98, 8.66]
        t_infty = [3.75, 3.99, 4.57, 5.26, 6.33, 7.13, 8.11, 9.36, 10.03]
    else:
        raise ValueError("No such EV exists")

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    line1 = ax1.plot(d, t_30_min, "#377eb8",  linewidth=3, linestyle="solid", label="30 min", marker="o", markersize=8)
    line2 = ax1.plot(d, t_45_min, "#ff7f00", linewidth=3, linestyle="dashed", label="45 min", marker="s", markersize=8)
    line3 = ax1.plot(d, t_60_min, "#9467BD",  linewidth=3,
                       linestyle="dashdot", label="60 min", marker="*", markersize=12)
    line4 = ax1.plot(d, t_infty, "#E6AB02",  linewidth=3,
                       linestyle="dashdot", label=r"No Threshold", marker="D", markersize=8)
    ax1.set_xlabel("d in Power-of-d", fontsize=20)
    if plot_type == "pickup_time":
        ax1.set_ylabel(f"Pickup Time (min)", fontsize=20)
        ax1.set_ylim([2, 10.5])
    elif plot_type == "workload":
        ax1.set_ylabel(f"Workload %", fontsize=20)
        ax1.set_ylim([77, 94])
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    # ax2.set_ylabel("Pickup Time (min)", fontsize=15)
    if ev_type == "nissan":
        plt.title("Nissan Leaf", fontsize=20)
    elif ev_type == "tesla":
        plt.title("Tesla Model 3", fontsize=20)
    # added these three lines
    # lines = line1 + line2 + line3 + line4
    # legends = [l.get_label() for l in lines]
    # ax1.legend(lines, legends, fontsize=10)
    plot_file_png = os.path.join(plot_dir, f"3_load_balancing_{plot_type}_{ev_type}.png")
    plot_file_eps = os.path.join(plot_dir, f"3_load_balancing_{plot_type}_{ev_type}.eps")
    plt.savefig(plot_file_png, bbox_inches='tight')
    plt.savefig(plot_file_eps, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    load_balancing(ev_type="tesla", plot_type="pickup_time")
    load_balancing(ev_type="nissan", plot_type="pickup_time")
    load_balancing(ev_type="tesla", plot_type="workload")
    load_balancing(ev_type="nissan", plot_type="workload")
    # fleet_can_pod(data_type="Chicago Dataset")
    # fleet_can_pod(data_type="Uniform Origin and Destinations")
