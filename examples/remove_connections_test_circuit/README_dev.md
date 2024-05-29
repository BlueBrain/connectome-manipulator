# How to Run

This tutorial demonstrates how to remove connections from an existing circuit.

## Prerequistes

- A virtual environment with connectome manipulator and its dependencies installed
- BBP Supercomputer Access 


## 1. Make sure you have a virtual environment with connectome manipulator and its dependencies installed

Current state of the tool has a specific requirements like python=3.10.8 and its only tested in BBP supercomputer. So i suggest creating python3.10.8 venv and then install the manipulator.


## 2. Run run_rewiring_parallel.sh with necessary fields

Example:

```
sbatch run_rewiring_parallel.sh  manip_config.json /path/to/output 100

```

## Developer Info

In the configuration file , anything that works with Bluepysnap querying should work under sel_source or sel_dest

### Allowed Use Cases Example

```
"sel_src": "Mosaic_A",
        "sel_dest": {
            "node_set":"Mosaic_A",
            "mtype": [
              "L4_PC", 
              "L4_MC"
            ]
          }
```

### Not Allowed

1. Deleting the entire connectome
2. Giving node population name directly to sel_src value

### To debug on salloc , after allocation run

```
. /etc/profile.d/modules.sh
unset MODULEPATH
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/config/modules.sh
module purge
module load archive/2023-07 python-dev parquet-converters/0.8.0 py-mpi4py
source /gpfs/bbp.cscs.ch/project/proj112/home/kurban/christoph_paper/github/venv_3_10_8/bin/activate
```

and then call manipulator directly like this


```
connectome-manipulator -v manipulate-connectome manip_config.json --output-dir=/gpfs/bbp.cscs.ch/project/proj112/home/kurban/christoph_paper/github/forked/connectome-manipulator_forked/examples/remove_connections_test_circuit/output/example_circuit_Mosaic_A --convert-to-sonata --splits=1
```