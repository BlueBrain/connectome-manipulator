![Connectome-Manipulator](BPP-Connectome-Manipulator-Banner.jpg)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI:10.1101/2024.05.24.593860](http://img.shields.io/badge/DOI-10.1101/2024.05.24.593860-B31B1B.svg)](https://doi.org/10.1101/2024.05.24.593860)
[![Run status](https://github.com/BlueBrain/connectome-manipulator/actions/workflows/run-tox.yml/badge.svg)](https://github.com/BlueBrain/connectome-manipulator/actions/workflows/run-tox.yml)
[![Coverage](https://codecov.io/github/BlueBrain/connectome-manipulator/coverage.svg?branch=main)](https://codecov.io/github/BlueBrain/connectome-manipulator)
[![Latest release](https://img.shields.io/pypi/v/connectome-manipulator.svg)](https://pypi.org/project/connectome-manipulator/)


# Connectome-Manipulator

A connectome manipulation framework for SONATA circuits


## Table of contents

1. [Introduction](#introduction)
2. [Install](#install)
3. [Framework overview](#framework-overview)
    - [Main components](#main-components)
    - [Operation principle of the "connectome manipulator"](#operation-principle-of-the-connectome-manipulator)
4. [How to run](#how-to-run)
    - [Connectome manipulation or building](#connectome-manipulation-or-building)
    - [Model building](#model-building)
    - [Structural comparison](#structural-comparison)
5. [Examples](#examples)
6. [How to contribute](#how-to-contribute)
7. [Citation](#citation)
8. [Publications that use or mention Connectome-Manipulator](#publications-that-use-or-mention-connectome-manipulator)
    - [Scientific papers that use Connectome-Manipulator](#scientific-papers-that-use-connectome-manipulator)
    - [Posters that use Connectome-Manipulator](#posters-that-use-connectome-manipulator)
9. [Funding & Acknowledgment](#funding--acknowledgment)


## Introduction

An important way to study cortical function _in silico_ lies in manipulations that are impossible to perform _in vivo_ and _in vitro_. The purpose of the _Connectome-Manipulator_ is to provide a general framework to apply and study various types of structural manipulations of a circuit connectome. The framework allows for rapid connectome manipulations of biophysically detailed network models in [SONATA](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md) format, the standard for large-scale network models. Manipulations can be targeted to entire models, specific sub-networks, or even single neurons, ranging from insertion or removal of specific motifs to complete rewiring based on stochastic connectivity models at various levels of complexity.
Important use cases include wiring a connectome from scratch in a biologically realistic way based on given features of connectivity, rewiring an existing connectome while preserving certain aspects of connectivity, and transplanting specific connectivity characteristics from one connectome to another. The resulting connectomes can be readily simulated using any simulator supporting SONATA, allowing systematic and reproducible characterization of causal effects of manipulations on network activity.


## Install

From [PyPI](https://pypi.org/project/connectome-manipulator) (recommended):
~~~
pip install connectome-manipulator
~~~

Alternatively, from [GitHub](https://github.com/BlueBrain/connectome-manipulator):
~~~
git clone https://github.com/BlueBrain/connectome-manipulator.git
cd connectome-manipulator
pip install .
~~~

All dependencies declared in `setup.py` and are available from [PyPI](https://pypi.org/), including one optional dependency, `mpi4py` (v3.1.4), which is required for parallel processing, i.e., to run `parallel-manipulator`. Another optional dependency, `parquet-converters` (v0.8.0 or higher), required for converting .parquet output files to SONATA must be installed separately, see instructions under https://github.com/BlueBrain/parquet-converters.

Recommended Python version: v3.10.8

## Framework overview

### Main components

The _Connectome-Manipulator_ framework is illustrated in Figure 1 and consists of the following main components: 

* __Connectome manipulator__\
  As specified in the config, applies one or a sequence of manipulations to a given SONATA connectome, and writes the manipulated connectome to a new SONATA edges file. All manipulations are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/connectome_manipulation/README.md](connectome_manipulator/connectome_manipulation/)
* __Model building__\
  As specified in the config, builds a model from a given connectome and writes the model to a file to be loaded and used by specific manipulations requiring a model (e.g., model-based rewiring based on connection probability model). All models are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/model_building/README.md](connectome_manipulator/model_building/)
  
  > <ins>Notes</ins>:
  > - Some models may not even require a connectome as input.
  > - Some models may depend on other models as input for model building.

* __Structural comparator__\
  As specified in the config, performs a structural comparison of the original and manipulated connectomes. Different structural parameters to compare (connection probability, synapses per connection, ...) are separately implemented in sub-modules and can be easily extended.\
  Details can be found in the corresponding README file: [/connectome_comparison/README.md](connectome_manipulator/connectome_comparison/)

ℹ️ More details can be also found in the accompanying publication (esp. _Supplementary tables_), see [Citation](#citation).

| ![Schematic overview](doc/source/images/schematic_overview.png "Schematic overview of the connectome manipulation pipeline, consisting of the 'Connectome manipulator', 'Model building', 'Structural comparator', and 'Topological comparator' modules.") |
| :-: |
| __Figure 1:__ Schematic overview of the connectome manipulation framework, consisting of the "connectome manipulator", "model building", and "structural comparator" components. |


### Operation principle of the "connectome manipulator"

As illustrated in Figure 2, the synapses of the connectome (SONATA edges) are divided into k splits targeting disjoint sets of N post-synaptic neurons (SONATA nodes), which reduces the memory consumption and facilitates parallelization on multiple computation nodes. Each split is an edge table loaded as Pandas dataframe and comprising a list of synapses together with all synapse properties, an example is shown in Figure 3. The manipulations are then applied separately to each edge table in sequence (or alternatively, in parallel), resulting in manipulated edge tables which are then written to separate .parquet files. In the end, all .parquet files are merged into one manipulated SONATA connectome file using `parquet-converters`.

| ![Operation principle](doc/source/images/operation_principle.png "Operation principle of the 'connectome manipulator', illustrating its split-based processing architecture.") |
| :-: |
| __Figure 2:__ Operation principle of the "connectome manipulator", illustrating its split-based processing architecture. |

| ![Edge table](doc/source/images/edge_table.png "Example of an edge table (Pandas dataframe) comprising all synapse properties.") |
| :-: |
| __Figure 3:__ Example of an edge table (Pandas dataframe), comprising a list of synapses together with all synapse properties. |

> <ins>Notes</ins>:
> - Manipulations can only be applied to a single SONATA edges population at a time.
> - The synapses in each loaded edge table are assumed to be sorted by post-synaptic neuron ID. Likewise, the manipulated edges tables are to be returned with synapses sorted by post-synaptic neuron ID.
> - Optionally, processing can be resumed from an earlier (incomplete) run, by re-using all .parquet files that already exist instead of re-computing them.
> - By default, all .parquet files will be deleted after successfull completion, i.e., after the manipulated SONATA connectome file has been generated. Optionally, these temporary .parquet files can be kept as well.


## How to run

~~~
Usage: connectome-manipulator [OPTIONS] COMMAND [ARGS]...

  Connectome manipulation tools.

Options:
  --version      Show the version and exit.
  -v, --verbose  -v for INFO, -vv for DEBUG  [default: 0]
  --help         Show this message and exit.

Commands:
  build-model            Extract and build models from existing connectomes.
  compare-connectomes    Compare connectome structure of two circuits.
  manipulate-connectome  Manipulate or build a circuit's connectome.
~~~

### Connectome manipulation or building

~~~
Usage: connectome-manipulator manipulate-connectome [OPTIONS] CONFIG

  Manipulate or build a circuit's connectome.

Options:
  --output-dir PATH         Output directory.  [required]
  --profile                 Enable profiling.
  --resume                  Resume from exisiting .parquet files.
  --keep-parquet            Keep temporary parquet files.
  --convert-to-sonata       Convert parquet to sonata and generate circuit
                            config
  --overwrite-edges         Overwrite existing edges file
  --splits INTEGER          Number of blocks, overwrites value in config file
                            [default: 0]
  --target-payload INTEGER  Number of gid-gid pairs to consider for one block.
                            Supersedes splits when a parquet based
                            configuration is used  [default: 20000000000]
  --parallel                Run using a parallel DASK job scheduler
  -a, --parallel-arg TEXT   Overwrite the arguments for the Dask Client with
                            key=value
  --help                    Show this message and exit.

~~~

Just running serially you can do something like this:

```
connectome-manipulator -v manipulate-connectome wiring_config.json \
    --output-dir PATH_TO_OUTPUT --profile --convert-to-sonata --splits 1
```

Running splits in parallel (with Dask) you can use the `parallel-manipulator` executable that will set up Dask automatically (and switch the `--parallel` flag by default to `True`, too):

```
srun --nodes 10 --tasks-per-node=2 --cpus-per-task=20 --constraint=clx --mem=0 \
    parallel-manipulator -v manipulate-connectome wiring_config.json \
    --output-dir PATH_TO_OUTPUT --profile --convert-to-sonata --splits=100
```

Please note that this feature will require at least 4 MPI ranks. Dask will use 2 ranks to manage the distributed cluster. We recommend to use a high number for `--cpus-per-task` to create Dask workers that will be able to process a lot of data in parallel.

When processing with `parallel-manipulator`, one may pass the flag `--target-payload` to determine how big the individual workload for each process should be. The default value of 20e9 was determined empirically to run on the whole mouse brain with 75 million neurons. We recommend to use this value as a starting point and scale it up or down to achieve the desired runtime characteristics.


### Model building

~~~
Usage: connectome-manipulator build-model [OPTIONS] CONFIG

  Extract and build models from existing connectomes.

Options:
  --force-reextract  Force re-extraction of data, in case already existing.
  --force-rebuild    Force model re-building, in case already existing.
  --help             Show this message and exit.
~~~


### Structural comparison

~~~
Usage: connectome-manipulator compare-connectomes [OPTIONS] CONFIG

  Compare connectome structure of two circuits.

Options:
  --force-recomp-circ1  Force re-computation of 1st circuit's comparison data,
                        in case already existing.
  --force-recomp-circ2  Force re-computation of 2nd circuit's comparison data,
                        in case already existing.
  --help                Show this message and exit.
~~~


## Examples

Examples can be found in the Jupyter notebooks listed [here](examples/README.md).


## How to contribute

Contribution guidelines can be found [here](CONTRIBUTING.md).


## Citation

If you use this software, we kindly ask you to cite the following publication:

C. Pokorny, O. Awile, J.B. Isbister, K. Kurban, M. Wolf, and M.W. Reimann (2024). __A connectome manipulation framework for the systematic and reproducible study of structure-function relationships through simulations.__ bioRxiv 2024.05.24.593860. DOI: [10.1101/2024.05.24.593860](https://doi.org/10.1101/2024.05.24.593860)

```
@article{pokorny2024connectome,
  author = {Pokorny, Christoph and Awile, Omar and Isbister, James B and Kurban, Kerem and Wolf, Matthias and Reimann, Michael W},
  title = {A connectome manipulation framework for the systematic and reproducible study of structure--function relationships through simulations},
  journal = {bioRxiv},
  year = {2024},
  publisher={Cold Spring Harbor Laboratory},
  doi = {10.1101/2024.05.24.593860}
}
```

## Publications that use or mention Connectome-Manipulator

### Scientific papers that use Connectome-Manipulator

- Michael W. Reimann, Sirio Bolaños-Puchet, Jean-Denis Courcol, Daniela Egas Santander, et al. __Modeling and Simulation of Neocortical Micro- and Mesocircuitry. Part I: Anatomy.__ bioRxiv 2022.08.11.503144. DOI: [10.1101/2022.08.11.503144](https://doi.org/10.1101/2022.08.11.503144)

- James B. Isbister, András Ecker, Christoph Pokorny, Sirio Bolaños-Puchet, Daniela Egas Santander, et al. __Modeling and Simulation of Neocortical Micro- and Mesocircuitry.__ Part II: Physiology and Experimentation. bioRxiv 2023.05.17.541168. DOI: [10.1101/2023.05.17.541168](https://doi.org/10.1101/2023.05.17.541168)

- Daniela Egas Santander, Christoph Pokorny, András Ecker, Jānis Lazovskis, Matteo Santoro, Jason P. Smith, Kathryn Hess, Ran Levi, and Michael W. Reimann. __Efficiency and reliability in biological neural network architectures.__ bioRxiv 2024.03.15.585196. DOI: [10.1101/2024.03.15.585196](https://doi.org/10.1101/2024.03.15.585196)

### Posters that use Connectome-Manipulator

- Christoph Pokorny, Sirio Bolaños-Puchet, András Ecker, James B. Isbister, Michael Gevaert, Joni Herttuainen, Henry Markram, and Michael W. Reimann. __Impact of simplified network structure on cortical activity.__ Bernstein Conference, 2022, Berlin.

- Kerem Kurban, Christoph Pokorny, Julian Budd, Alberto Antonietti, Armando Romani, and Henry Markram. __Topological properties of a full-scale model of rat hippocampus CA1 and their functional implications.__ Annual meeting of the Society for Neuroscience, 2022, San Diego.

- Christoph Pokorny, Omar Awile, Sirio Bolaños-Puchet, András Ecker, Daniela Egas Santander, James B. Isbister, Matthias Wolf, Henry Markram, and Michael W. Reimann. __A connectome manipulation framework for the systematic and reproducible study of the structure-function relationship through simulations.__ Bernstein Conference, 2023, Berlin.

- Christoph Pokorny, Omar Awile, James B. Isbister, Kerem Kurban, Matthias Wolf, and Michael W. Reimann. __A connectome manipulation framework for the systematic and reproducible study of structure-function relationships through simulations.__ FENS Forum, 2024, Vienna.


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2024 Blue Brain Project/EPFL
