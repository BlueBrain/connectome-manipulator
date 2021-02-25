# Connectome Manipulator

## Project: ReSyCo (Rewiring of Synaptic Connectivity)

Applies manipulations to the connectome of a SONATA circuit, and runs a structural and topological comparison of the raw and manipulated connectomes


## Overview

* __/bin__  
  Shell scripts to launch connectome manipulations independently as SLURM jobs
* __/notebooks__  
  Contains the main scripts (Jupyter notebooks) for running connectome manipulations and visualizations, together with all produced results figures
* __/pipeline__  
  Processing pipeline code, containing the specific implementation of all manipulations and visualizations
* __/working_dir__  
  Working directory to store all pre-computed results for visualizations (Note: the actual manipulations are stored directly at the circuit location)
