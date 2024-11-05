# Contribution Guide

We would love for you to contribute to the connectome-manipulator project and help make it better than it is today.
As a contributor, here are the guidelines we would like you to follow:

 - [Question or Problem?](#got-a-question)
 - [Issues and Bugs](#found-a-bug)
 - [Feature Requests](#missing-a-feature)
 - [Submissions](#submission-guidelines)
 - [How to extend](#how-to-extend)

# Got a Question?

Please do not hesitate to raise an issue on [github project page][github].

# Found a Bug?

If you find a bug in the source code, you can help us by [submitting an issue](#issues)
to our [GitHub Repository][github]. Even better, you can [submit a Pull Request](#pull-requests) with a fix.

# Missing a Feature?

You can *request* a new feature by [submitting an issue](#issues) to our GitHub Repository.
If you would like to *implement* a new feature, please submit an issue with a proposal for your 
work first, to be sure that we can use it.

Please consider what kind of change it is:

* For a **Major Feature**, first open an issue and outline your proposal so that it can be
discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
and help you to craft the change so that it is successfully accepted into the project.
* **Small Features** can be crafted and directly [submitted as a Pull Request](#pull-requests).

# Submission Guidelines

## Issues

Before you submit an issue, please search the issue tracker, maybe an issue for your problem
already exists and the discussion might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we need to reproduce
and confirm it. In order to reproduce bugs we will need as much information as possible, and
preferably with an example.

## Pull Requests

When you wish to contribute to the code base, please consider the following guidelines:

* Make a [fork](https://guides.github.com/activities/forking/) of this repository.
* Make your changes in your fork, in a new git branch:

     ```shell
     git checkout -b my-fix-branch master
     ```
* Create your patch, **including appropriate Python test cases**.
  Please check the coding [conventions](#coding-conventions) for more information.
* Run the full test suite, and ensure that all tests pass.
* Commit your changes using a descriptive commit message.

     ```shell
     git commit -a
     ```
* Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```
* In GitHub, send a Pull Request to the `master` branch of the upstream repository of the relevant component.
* If we suggest changes then:
  * Make the required updates.
  * Re-run the test suites to ensure tests are still passing.
  * Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

       ```shell
        git rebase master -i
        git push -f
       ```

That’s it! Thank you for your contribution!

### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes from
the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```
* Check out the master branch:

    ```shell
    git checkout master -f
    ```
* Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```
* Update your master with the latest upstream version:

    ```shell
    git pull --ff upstream master
    ```

[github]: https://github.com/BlueBrain/connectome-manipulator

# How to extend

The connectome manipulation framework has been developed using reusable primitives, such as Python classes and specific file structures for individual code modules that allow easy extension of the framework in order to add new functionality. Specifically, new types of (stochastic) models, tools for fitting them, new manipulation operations, and additional structural validation methods can be added to the code repository as outlined below:

## Models

All models are implemented under [`/model_building/model_types.py`](connectome_manipulator/model_building/model_types.py) and are derived from an abstract base class `AbstractModel` which provides general functionality for loading/saving models and evaluating them, i.e., returning the model output given its input. Specific functionality must be implemented in a respective derived class which must define the model parameter names (`param_names`; i.e., variables storing the internal representation of the model), default parameter values (`param_defaults`), names of data frames (`data_names`; for large data elements, if any, that would be stored as associated HDF5 file), and input names (`input_names`; i.e., input variables the model output depends on). Moreover, the derived class must provide implementations of `get_model_output()` for returning the model output given its input variables, and `__str__()` for returning a string representation describing the model. When initializing a concrete model instance, values for all specified model parameters and data frames must be provided. Values for model parameters can be omitted in case default parameter values have been defined instead.

Another useful (abstract) base class `PathwayModel` exists which can be used in the same way as outlined above, but which already includes pathway-specific model parameterization. Specifically, it allows to store different parameter values dependent on pairs of pre-synaptic (`src_type`) and post-synaptic (`tgt_type`) m-types, together with default values in case no pathway is specified.

## Model fitting functions

All model fitting functions are implemented as separate code modules (.py files) under [`/model_building`](connectome_manipulator/model_building) and must always contain the following functions for implementing the three steps of model building:

  - `extract()` for extracting relevant data (e.g., connection probabilities at binned distances) from a given connectome which will be stored automatically in a .pickle file by the framework
  - `build()` for fitting model parameters against the data extracted during the previous step and initializing a model instance which will then be stored automatically as a .json file, optionally together with an associated HDF5 file
  - `plot()` for generating visualizations of the extracted data versus the model output, and storing them in the output folder

Importantly, arbitrary parameters (optionally, including default values) can be added as keyword arguments to any of the three functions, values of which must be provided through a configuration file (see *Configuration file structure* in the [Documentation](https://connectome-manipulator.readthedocs.io/en/netneuro-24-0092-rev1/config_file_structure.html)) when launching model building.

## Manipulations

All manipulations are derived from an abstract base class `Manipulation` which is implemented in [`/connectome_manipulation/manipulation/base.py`](connectome_manipulator/connectome_manipulation/manipulation/base.py). The base class provides access to the neurons of a network model (through `self.nodes`) as well as to the input (i.e., before a manipulation) and output (i.e., after a manipulation) synapse tables (through `self.writer`). An alternative (abstract) base class, `MorphologyCachingManipulation`, exists which additionally provides efficient access to morphologies (through `self._get_tgt_morphs`) including a caching mechanism, i.e., without reloading them from the file system in case of repeated invocations.

A concrete manipulation must be implemented in a derived classes and stored in a separate code module (.py file) under [`/connectome_manipulation/manipulation`](connectome_manipulator/connectome_manipulation/manipulation). It must contain an implementation for the `apply()` method which must return a new synapse table (through `self.writer`) as a result of the manipulation. Importantly, arbitrary parameters (optionally, including default values) can be added as keyword arguments to the `apply()` method, values of which must be provided through a configuration file (see *Configuration file structure* in the [Documentation](https://connectome-manipulator.readthedocs.io/en/netneuro-24-0092-rev1/config_file_structure.html)) when launching a manipulation.

## Structural comparison functions

All structural comparison functions are implemented as separate code modules (.py files) under [`/connectome_comparison`](connectome_manipulator/connectome_comparison) and must always contain functions for implementing the two following steps:

  - `compute()` for computing specific structural features from a given connectome (e.g., connection probability by layer), which will be evaluated for both connectomes to compare and results of which will be automatically stored as .pickle files by the framework
  - `plot()` for plotting a graphical representation of individual feature instances (e.g., 2D matrix plot of connection probabilities by layer) or the difference between two such instances, which will be automatically stored in a compound output figure when comparing two connectomes

Importantly, arbitrary parameters (optionally, including default values) can be added as keyword arguments to the two functions, values of which must be provided through a configuration file (see *Configuration file structure* in the [Documentation](https://connectome-manipulator.readthedocs.io/en/netneuro-24-0092-rev1/config_file_structure.html)) when launching a structural comparison.
