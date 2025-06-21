# EV-based Transportation System

In this document, we outline how to use this simulation framework to reproduce the results reported in [1].

## Setup

Start by cloning the repository and switching to the branch "management-science-version".
```
git clone https://github.com/smv30/spatial_queueing.git
cd spatial_queueing
git checkout feature/management_science_version
```
Next, we create a virtual conda enivornment and install all the necessary packages.
```
conda create -n spatial_queueing python=3.9
conda activate spatial_queueing
pip install -r requirements.txt
```