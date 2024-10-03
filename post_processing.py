import os
import pandas as pd

dir_name = "5_ode_comparison_45_mins_attempt_2"
kpi_consolidated = None
for fname in os.listdir(dir_name):

    # build the path to the folder
    folder_path = os.path.join(dir_name, fname)

    if os.path.isdir(folder_path):
        # we are sure this is a folder; now lets iterate it
        filepath = os.path.join(folder_path, "kpi.csv")

        kpi = pd.read_csv(filepath)
        kpi["folder_name"] = fname
        if kpi_consolidated is None:
            kpi_consolidated = kpi
        else:
            kpi_consolidated = pd.concat([kpi_consolidated, kpi], axis=0)

results_file = os.path.join(dir_name, "kpi_consolidated.csv")
kpi_consolidated.to_csv(results_file)