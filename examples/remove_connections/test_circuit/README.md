# How to Run

This tutorial demonstrates how to remove connections from an existing test circuit from BlueCelluLab. 

## Prerequistes

- A virtual environment with connectome manipulator and its dependencies installed
- BBP Supercomputer Access 


## 1. Make sure you have a virtual environment with connectome manipulator and its dependencies installed

Current state of the tool has a specific requirements like python=3.10.8 and its only tested in BBP supercomputer. So it is suggested to use a virtual environment using python3.10.8 and then install the manipulator.


## 2. Run run_rewiring_parallel.sh with necessary fields

Example:

```
sbatch run_rewiring_parallel.sh  manip_config.json /path/to/output 100

```