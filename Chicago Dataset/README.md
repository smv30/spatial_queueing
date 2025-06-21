# Chicago Dataset
We first need to download the Chicago Trip Dataset. While this dataset was downloaded from [2], we provide a pre-filtered dataset ready to use for our purposes on [this dropbox link](https://www.dropbox.com/scl/fo/137ug19aq72zxsqr3zh54/ABKw2E8-hdDSjoqj7iU0cSM?rlkey=lxbtdw5a0uko44zd92m04y31z&st=jjbzieyc&dl=0). Download this file and place it in the root folder: spatial_queueing. Now we are ready to run the simulations.

## Running your first simulation
Run the detailed simulation using the following command:
```
python main.py -ev <ev> -ckw <ckw> -algo <algo> -d <d> -t <t> -nev <nev> -nc <nc> -pt <pt> -rf <rf>
```
Below we provide a brief description of the inputs:
- ev (str): sets a type of an EV which determines the battery pack size (kWh), consumption (kWh/mi), and battery degredation (%). Possible values: {"Nissan_Leaf", "Tesla_Model_3", "Mustang_Mach_E_ER_AWD", "Waymo"}.
- ckw (int): charge rate in kW.
- algo (str): sets the matching and charging policy. Possible values: {"POD",  "CAD", "POTP", "CAN_POD",  "CAN_CAD", "CAN_R_POD", "CAN_R_POD_N"}.
- d (float): value of d in Power-of-d. If d=1, then, we obtain the closest dispatch (CD) policy.
- t (int): maximum pickup threshold in mins. All trips with pickup time greater than t mins are dropped. 
- nev (int): number of EVs.
- nc (int): number of chargers.
- pt (float): uniformly samples pt fraction of total trips to modulate the average arrival rate.
- rf (str): the results folder where all the data will be saved.

The following sample simulation takes about 30 mins to run:
```
python main.py -ev "Tesla_Model_3" -ckw 20 -algo "POD" -d 2 -t 45 -nev 800 -nc 150 -pt 0.2 -rf "simulation_results"
```
Note that the run time grows as -pt increases, reaching 3.5 hours for pt=0.6. Each simulation creates a parent folder and the stackplot for the sim is saved in parent folder -> plots -> demand_curve_stackplot.png.

## Fleet size for 90% workload
In this study, we compare the fleet size corresponding to 90% workload served under various policies. As described above, run main.py for 593 different combinations of parameters as documented in inputs_fleet_plot.csv. To postprocess the resultant data to generate Figure 5 and 6, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "fleet_plot"
```

Note that running 593 simulations in series is intractable as each sim takes about 1.5 hours on average to finish. We suggest to parallelize these instances on a server.


## Policy Comparison
In this study, we compare performance of various matching policies. As described in "Running your first simulation", run main.py for 24 different parameters as documented in inputs_policy_comparison.csv. To postprocess the resultant data to generate the data for Table 4 and 7, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "policy_comparison_table"
```

## The effect of d and $T_{P, \max}$
In this study, we compare performance of Power-of-d and Power-of-$T_{P, \max}$ as a function of $d$ and $T_{P, \max}$. As described in "Running your first simulation", run main.py with ckw=20, nev=2400, nc=225, pt=0.6
and ev = {Nissan_Leaf, Tesla_Model_3, Mustang_Mach_E_ER_AWD}.
- For algo=POTP, set d=0 and t = {5, 6, ... , 15, 20}
- For algo=POD set t = {30, 45, 60, 0}
    - if ev=Nissan_Leaf, set d = {1, ..., 10}
    - else, set d = {1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6}
To postprocess the resultant data to generate Figure 7, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "pod_as_a_function_of_d_plot"
```

## Fleet size + Number of Charger Contour
In this study, we evaluate the percentage workload served under Power-of-2 as the fleet size and number of chargers vary. As described in "Running your first simulation", run main.py with ckw=20, algo="POD", d=2, t=45, pt=0.6, and ev = {Nissan_Leaf, Tesla_Model_3}. In addition, set nev={1500, 1800, 2100, 2400, 2700} and nc={25, 75, 125, 175, 225}. To postprocess the resultant data to generate Figure 4, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "contour_plot"
```

## Infrastructure Planning via POD-ODE
In this study, we compare the infrastructure planning prescription of the detailed simulations versus the POD-ODE (Eq 8 in [1]). 

We first run the detailed simulations. As described in "Running your first simulation", run main.py with ev="Tesla_Model_3", ckw=20, algo="POD", t=45, pt=0.6, nev = {2100, 2200, 2300, 2400, 2500}. In addition, we set (nc, d) = {(125, 2.6), (175, 1.6), (225, 1.4)}. To postprocess the resultant data to generate "Sim" part of Table 3, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "pod_ode_detailed_sims"
```

We now run the POD-ODE using the following command:
```
python pod_ode.py -root_sim <directory of detailed sims> -od <where POD-ODE results are saved>
```
This simulation will first fit the pickup time as a function of number of available EVs and the drive to the charger time as a function of number of idle chargers using the power law and it will print the results. Then, using these fitted curves, it will run the POD-ODE for several values of fleet size, number of chargers, and error in estimation parameters and the results are saved in kpi_consolidated.csv. Then, it will post process these results to print the fleet size corresponding to 90% workload served. The results are also saved in 90_percent_fleet_size.csv. The whole process takes about one hour.

[1]: Sushil Mahavir Varma, Francisco Castro, and Siva Theja Maguluri. "Electric vehicle fleet and charging infrastructure planning." arXiv preprint arXiv:2306.10178 (2023).