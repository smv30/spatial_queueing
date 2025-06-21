# Uniform Dataset
While this dataset was generated uniformly at random, we provide a pre-generated dataset that we used for our simulations on [this dropbox link](https://www.dropbox.com/scl/fo/137ug19aq72zxsqr3zh54/ABKw2E8-hdDSjoqj7iU0cSM?rlkey=lxbtdw5a0uko44zd92m04y31z&st=jjbzieyc&dl=0). Download this file and place it in the folder: "spatial_queueing/Uniform Data/data", where "spatial_queueing" is our root folder. Now we are ready to run the simulations.

## Running your first simulation
Run the detailed simulation using the following command:
```
python main.py -nev <nev> -nc <nc> -lambda <lambda> -r <r> -ps <ps> -np <np> -rf <rf>
```
Below we provide a brief description of the inputs:
- nev (int): number of EVs.
- nc (int): number of chargers.
- lambda (int): arrival rate of customers per minute.
- r (int): Takes values in {1, 2, 3, 4, 5} as each sim is repeated 5 times.
- ps (int): Battery pack size in kWh.
- np (int): Number of posts per charger.
- rf (str): results folder

The following sample simulation takes about 20 mins to run:
```
python main.py -nev 1000 -nc 500 -lambda 80 -r 1 -ps 40 -np 1 -rf "simulation_results"
```
Note that the run time grows as -lambda increases, reaching 3 hours for lambda=320. Each simulation creates a parent folder and the stackplot for the sim is saved in parent folder -> plots -> demand_curve_stackplot.png.

## Asymptotic Sim
In this study, we verify the infrastructure planning prescription of Theorem 1 and 2 in [1] by deducing several fleet size and number of charger combinations resulting in 90% service level for arrival rates {5, 10, 20, ..., 320}. To reproduce Figure 11, run main.py for 140 different combinations of parameters as documented in inputs_asymptotic_sim.csv. To postprocess the resultant data to generate Figure 5 and 6, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "asymptotic_sim"
```
The above command will generate kpi_consolidated.csv file. First, verify that the the "service_level_percentage_second_half" is in the range [89, 91] to ensure that the input fleet size and number of charger combinations indeed results in 90% service level (approximately). In addition, the above command will also generate all plots in Figure 11.

Note that running 140 simulations in series is intractable. If you would like to run them in series, we recommend ommitting the sims corresponding to arrival rates 160 and 320 to ensure reasonable run times.

## Fleet size + Number of Charger / Packsize Contour
In this study, we evaluate the service level percentage under Power-of-2 as the fleet size and number of chargers / battery pack size vary. 

First we generate the fleet size versus number of charger contour. As described in "Running your first simulation", run main.py with lambda=80, ps=40, np=1, r = {1, 2, 3, 4, 5}, nev = {1200, ..., 1900}, and nc = {200, ..., 1100}. To postprocess the resultant data to generate Figure 2 (left), simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "charger_contour"
```

Similarly, to generate the fleet size versus battery pack size contour, run main.py with lambda=80, nc=1500, np=1, r = {1, 2, 3, 4, 5}, ps = {5, 10, ..., 30}, nev = {1100, ..., 1700}. To postprocess the resultant data to generate Figure 2 (right), simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "packsize_contour"
```

## Policy Comparison
In this study, we compare performance of various matching policies. As described in "Running your first simulation", run main.py for 210 different parameters as documented in inputs_policy_comparison_uniforms.csv. To postprocess the resultant data to generate Figure 13, simply run the following command:
```
python post_processing.py -root <results_folder (rf)> -output "policy_comparison_table"
```



[1]: Sushil Mahavir Varma, Francisco Castro, and Siva Theja Maguluri. "Electric vehicle fleet and charging infrastructure planning." arXiv preprint arXiv:2306.10178 (2023).
