# EV-based Transportation System

In this document, we outline how to use this simulation framework to reproduce the results reported in [1].

## Setup

Start by cloning the repository and switching to the branch "management-science-version".
```
git clone https://github.com/smv30/spatial_queueing.git
cd spatial_queueing
git checkout management-science-version
```
Next, we create a virtual conda environment and install all the necessary packages.
```
conda create -n spatial_queueing -y python=3.9
conda activate spatial_queueing
pip install -r requirements.txt
```
If you would like to run the simulations for the Chicago Dataset, run the command
```
cd chicago_dataset/
```
and follow the instructions in the README.md file found inside the "[chicago_dataset](./chicago_dataset/)" folder. Similarly, if you would like to run the simulations for the Uniform Dataset, run the command
```
cd uniform_dataset/
```
and follow the instructions in the README.md file found inside the "[uniform_dataset](./uniform_dataset/)" folder.
