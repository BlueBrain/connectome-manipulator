# Connectome Manipulator

~~~
ℹ️ Project: ReSyCo - Rewiring of Synaptic Connectivity
~~~

An important way to study cortical function _in-silico_ lies in manipulations that are impossible to perform _in-vivo_ and _in-vitro_. One type of manipulations includes manipulations of the (micro-)structure of synaptic connections. The aim of this project is to provide a general framework to apply and study structural manipulations of a circuit connectome, referred to as rewiring.

With the tools implemented in this repository, it is possible to apply manipulations to the connectome of a [SONATA](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md) circuit, and to run a structural and topological comparison of the raw and manipulated connectomes. Importantly, the resulting (manipulated) SONATA connectomes are intended to be used in neural network simulations, which opens new possibilities for all kinds of simulation experiments to find causal relationships between structure and function.


## Folder overview

* __[/bin](bin)__\
  Shell scripts to launch connectome manipulations independently as SLURM jobs
* __[/images](images)__\
  Image files used in documentation pages
* __[/notebooks](notebooks)__\
  Main scripts (Jupyter notebooks) for running connectome manipulations, model building, and visualizations
* __[/pipeline](pipeline)__\
  Processing pipeline code, containing the specific implementation of all manipulations, model building, and visualizations
* __/working_dir__\
  Working directory (created at runtime) to store all pre-computed results for analyses and visualizations
  
  > __Note:__ The actual connectome manipulations are stored directly at the circuit location!


## Processing pipeline

The connectome manipulation pipeline is illustrated in Figure 1 and consists of the following modules. The Python library [Blue Brain SNAP](https://github.com/BlueBrain/snap) is used to access SONATA circuits.

* __Connectome manipulator__\
  Depending on the config, applies one or a sequence of manipulations to a given SONATA connectome, and writes the manipulated connectome to a new SONATA file. All manipulations are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/pipeline/connectome_manipulation/README.md](pipeline/connectome_manipulation/)

* __Model building__\
  Depending on the config, builds a model from a given connectome and writes the model to a file to be loaded and used by some manipulations requiring a model (e.g., model-based rewiring based on given connection probability model). All models are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/pipeline/model_building/README.md](pipeline/model_building/)
  
  > __Note 1:__ Some models may not even require a connectome as input.
  
  > __Note 2:__ Some models may depend on other models as input for model building.

* __Structural comparator__\
  Performs a structural comparison of the original and manipulated connectomes. Different structural parameters to compare (connection probability, synapses per connection, ...) are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/pipeline/connectome_comparison/README.md](pipeline/connectome_comparison/)

* __Topological comparator__\
  Performs a topological comparison of the original and manipulated connectomes based on advanced topological metrics.\
  External GitHub project: [MWolfR / topological_comparator](https://github.com/MWolfR/topological_comparator)

| ![Schematic overview](images/schematic_overview.png "Schematic overview of the connectome manipulation pipeline, consisting of the 'Connectome manipulator', 'Model building', 'Structural comparator', and 'Topological comparator' modules.") |
| :-: |
| __Figure 1:__ Schematic overview of the connectome manipulation pipeline, consisting of the _Connectome manipulator_, _Model building_, _Structural comparator_, and _Topological comparator_ modules. |


## Operation principle of the _Connectome manipulator_

As illustrated in Figure 2, the synapses of the connectome (SONATA edges) are divided into k blocks targeting disjoint sets of N post-synaptic neurons (SONATA nodes), which reduces the memory consumption and facilitates parallelization on multiple computation nodes. Each block is an edge table loaded as Pandas dataframe and comprising a list of synapses together with all synapse properties, an example is shown in Figure 3. The manipulations are then applied separately to each edge table in sequence (or alternatively, in parallel), resulting in manipulated edge tables which are then written to separate .parquet files. In the end, all .parquet files are merged into one manipulated SONATA connectome file using [Parquet Converters](https://bbpgitlab.epfl.ch/hpc/circuit-building/parquet-converters).

> __Note 1:__ A SONATA edges file is expected to contain a single edge population (to be manipulated) connecting two arbitrary source/target node populations.

> __Note 2:__ All synapses belonging to a certain pre-post connection are always within the same edge table.

> __Note 3:__ The synapses in each loaded edge table are assumed to be sorted by post-synaptic neuron ID. Likewise, the manipulated edge tables are to be returned with synapses sorted by post-synaptic neuron ID.

> __Note 4:__ Synapse indices do not need to be unique across all manipulated edge tables, as synapse indices are not stored in the resulting SONATA connectome.

| ![Operation principle](images/operation_principle.png "Operation principle of the 'Connectome manipulator', illustrating the block-based processing architecture.") |
| :-: |
| __Figure 2:__ Operation principle of the _Connectome manipulator_, illustrating the block-based processing architecture. |

| ![Edge table](images/edge_table.png "Example of an edge table (Pandas dataframe) comprising all synapse properties.") |
| :-: |
| __Figure 3:__ Example of an edge table (Pandas dataframe), comprising a list of synapses together with all synapse properties. |
