Configuration file structure
============================

1. `Model fitting config`_
2. `Connectome manipulation config`_
3. `Structural comparison config`_

Model fitting config
--------------------

.. code-block:: JSON
    :linenos:

    {
      "working_dir": "/path/to/working/directory",
      "out_dir": "/path/to/output/directory",
      "seed": 1234,
      "circuit_config": "/path/to/circuit_config.json",
      "model": {
        "name": "User-defined-model-name",
        "fct": {
          "source": "Name-of-code-module",
          "kwargs": {
            "key1": "value1",
            "key2": "value2",
            "...": "..."
          }
        }
      }
    }

====================  ====================================================================================================
Key                   Description
====================  ====================================================================================================
"working_dir"         Working directory to store data and model
"out_dir"             Output directory to store output figures (can be the same as ``working_dir``)
"seed"                Random seed, e.g., used when randomly subsampling data
"circuit_config"      Path to SONATA circuit config file (.json)
"model"               Model-specific settings
↳"name"               User-defined name of the model; will be used in filenames
↳"fct"                Specification of model fitting function
↳↳"source"            Name of source code module for model fitting, e.g., "conn_prob" or "delay"
↳↳"kwargs"            Keyword arguments provided as key-value pairs with model-specific settings that are passed to the
                      code module's extract(), build(), and plot() methods
====================  ====================================================================================================

**Working example:**

.. code-block:: JSON
    :linenos:

    {
      "working_dir": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release",
      "out_dir": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release",
      "seed": 7531,
      "circuit_config": "/gpfs/bbp.cscs.ch/project/proj83/jira-tickets/NSETM-1948-extract-hex-O1/data/O1_data/circuit_config.json",
      "model": {
        "name": "DistDepDelay-SSCxO1-Hex0EE",
        "fct": {
          "source": "delay",
          "kwargs": {
            "sel_src": {
              "node_set": "hex0",
              "synapse_class": "EXC"
            },
            "sel_dest": {
              "node_set": "hex0",
              "synapse_class": "EXC"
            },
            "sample_size": null,
            "bin_size_um": 50,
            "max_range_um": null
          }
        }
      }
    }

Connectome manipulation config
------------------------------

.. code-block:: JSON
    :linenos:

    {
      "circuit_config": "circuit_config.json OR /path/to/circuit_config.json",
      "circuit_path": "/path/to/circuit",
      "edges_popul_name": "Name-of-edges-population",
      "src_node_popul_name": "Name-of-source-nodes-population",
      "tgt_node_popul_name": "Name-of-target-nodes-population",
      "seed": 1234,
      "N_split_nodes": 100,
      "manip": {
        "name": "User-defined-manipulation-name",
        "syn_props_init": {
          "property1": "data-type1",
          "property2": "data-type2",
          "...": "..."
        },
        "fcts": [
          {
            "source": "Name-of-code-module",
            "key1": "value1",
            "key2": "value2",
            "...": "...",
            "pos_map_file": "/path/to/position/mapping/file",
            "model_config": {
              "Model-spec-key1": {
                "file": "path/to/model/file"
              },
              "Model-spec-key2": {
                "file": "path/to/model/file"
              }
            }
          }
        ]
      }
    }

=====================  ====================================================================================================
Key                    Description
=====================  ====================================================================================================
"circuit_config"       Circuit config filename (requires "circuit_path") OR full path to SONATA circuit config file (.json)
"circuit_path"         Optional path to SONATA circuit; required if "circuit_config" only contains a filename
"edges_popul_name"     Optional name of SONATA edges population
"src_node_popul_name"  Optional name of SONATA source nodes population
"tgt_node_popul_name"  Optional name of SONATA target nodes population
"seed"                 Random seed for stochastic manipulation
"N_split_nodes"        Optional number of data splits; will be overwritten by command line argument "--splits=N"
"manip"                Manipulation-specific settings
↳"name"                User-defined name of the manipulation; will be used in filenames
↳"syn_props_init"      Optional key-value pairs of property names and data types for initializing an enpty connectome
↳"fcts"                List for specifying a single or sequence of manipulation functions
↳↳"source"             Name of manipulation source code module, e.g., "conn_rewiring" or "syn_removal"
↳↳"key1", "key2", ...  Optional key-value pairs with manipulation-specific settings that are passed to the apply() method
                       of the code module's manipulation class
↳↳"pos_map_file"       Optional path to position mapping file
↳↳"model_config"       Optinal key-value pairs containing model specifications that are passed to the apply() method of the
                       code module's manipulation class; set to ``"model_config": {}`` if no models required
=====================  ====================================================================================================

**Working example:**

.. code-block:: JSON
    :linenos:

    {
      "circuit_config": "/gpfs/bbp.cscs.ch/project/proj83/jira-tickets/NSETM-1948-extract-hex-O1/data/O1_data/circuit_config.json",
      "seed": 3210,
      "manip": {
        "name": "ConnRewireOrder1Hex0EE",
        "fcts": [
          {
            "source": "conn_rewiring",
            "sel_src": {
              "node_set": "hex0",
              "synapse_class": "EXC"
            },
            "sel_dest": {
              "node_set": "hex0",
              "synapse_class": "EXC"
            },
            "syn_class": "EXC",
            "keep_indegree": false,
            "reuse_conns": false,
            "gen_method": "duplicate_randomize",
            "amount_pct": 100,
            "estimation_run": false,
            "opt_nconn": true,
            "p_scale": 1.0,
            "pos_map_file": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release/model/FlatPosMapping-SSCxO1.json",
            "model_config": {
              "prob_model_spec": {
                "file": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release/model/ConnProb1stOrder-SSCxO1-Hex0EE.json"
              },
              "delay_model_spec": {
                "file": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release/model/DistDepDelay-SSCxO1-Hex0EE.json"
              },
              "props_model_spec": {
                "file": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/model_building_v2/SSCx-HexO1-Release/model/ConnPropsPerPathway-SSCxO1-Hex0EE.json"
              }
            }
          }
        ]
      }
    }

Structural comparison config
----------------------------

.. code-block:: JSON
    :linenos:

    {
      "working_dir": "/path/to/working/directory",
      "out_dir": "/path/to/output/directory",
      "circuits": {
        "0": {
          "circuit_config": "/path/to/first/circuit_config.json",
          "circuit_name": "Name-of-first-circuit"
        },
        "1": {
          "circuit_config": "/path/to/second/circuit_config.json",
          "circuit_name": "Name-of-second-circuit"
        }
      },
      "plot_types": [
        {
          "name": "User-defined-plot-name",
          "fct": {
            "source": "Name-of-code-module",
            "kwargs": {
              "key1": "value1",
              "key2": "value2",
              "...": "..."
            }
          },
          "res_sel": [
            "results-selection1",
            "results-selection2",
            "..."
          ],
          "range_prctile": 100,
          "fig_size": [
            11,
            3
          ],
          "fig_file": {
            "format": "png",
            "dpi": 600
          }
        },
        {
          "name": "Another-user-defined-plot-name",
          "fct": {
            "source": "...",
            "kwargs": {
              "...": "..."
            }
          },
          "res_sel": [
            "..."
          ],
          "range_prctile": 100,
          "fig_size": [
            11,
            3
          ],
          "fig_file": {
            "format": "png",
            "dpi": 600
          }
        }
      ]
    }

====================  ====================================================================================================
Key                   Description
====================  ====================================================================================================
"working_dir"         Working directory to store extracted data
"out_dir"             Output directory to store output figures (can be the same as ``working_dir``)
"circuits"            Selection of two SONATA circuits to compare
↳"0" & "1"            Specification of first and second circuit
↳↳"circuit_config"    Path to SONATA circuit config file (.json)
↳↳"circuit_name"      User-defined name; will be used in figures and filenames
"plot_types"          List of plots to generate
↳"name"               User-defined name of the plot; will be used in filenames
↳"fct"                Specification of structural comparison function
↳↳"source"            Name of source code module for structural comparison, e.g., "connectivity" or "properties"
↳↳"kwargs"            Keyword arguments provided as key-value pairs with comparison-specific settings that are passed to
                      the code module's compute() and plot() methods
↳"res_sel"            Selection of results for plotting; can be a list of keys corresponding to data items as returned by
                      compute()
↳"range_prctile"      Optional range percentile used for plotting the selected results
↳"fig_size"           Optional two-element list with width and height (in inch) of generated results figure(s)
↳"fig_file"           Optional settings for generated results figure file(s)
↳↳"format"            Output file format of generated figure(s), e.g., "png"
↳↳"dpi"               Resolution of the generated output figure(s) in dots-per-inch
====================  ====================================================================================================

**Working example:**

.. code-block:: JSON
    :linenos:

    {
      "working_dir": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/structural_comparator_v2/SSCx-HexO1-Release",
      "out_dir": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/structural_comparator_v2/SSCx-HexO1-Release",
      "circuits": {
        "0": {
          "circuit_config": "/gpfs/bbp.cscs.ch/project/proj83/jira-tickets/NSETM-1948-extract-hex-O1/data/O1_data/circuit_config.json",
          "circuit_name": "Orig"
        },
        "1": {
          "circuit_config": "/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SimplifiedConnectomeModels/circuits_v2/SSCx-HexO1-Release__ConnRewireOrder1Hex0EE/circuit_config.json",
          "circuit_name": "Order-1"
        }
      },
      "plot_types": [
        {
          "name": "ConnPerLayer_Hex0EE",
          "fct": {
            "source": "connectivity",
            "kwargs": {
              "group_by": "layer",
              "skip_empty_groups": false,
              "sel_src": {
                "node_set": "hex0",
                "synapse_class": "EXC"
              },
              "sel_dest": {
                "node_set": "hex0",
                "synapse_class": "EXC"
              }
            }
          },
          "res_sel": [
            "nsyn_conn",
            "conn_prob"
          ],
          "range_prctile": 100,
          "fig_size": [
            11,
            3
          ],
          "fig_file": {
            "format": "png",
            "dpi": 600
          }
        },
        {
          "name": "PropsPerMtype_Hex0EE",
          "fct": {
            "source": "properties",
            "kwargs": {
              "group_by": "mtype",
              "skip_empty_groups": true,
              "sel_src": {
                "node_set": "hex0",
                "synapse_class": "EXC"
              },
              "sel_dest": {
                "node_set": "hex0",
                "synapse_class": "EXC"
              },
              "fct": "np.mean"
            }
          },
          "res_sel": [
            "conductance",
            "decay_time",
            "delay",
            "depression_time",
            "facilitation_time",
            "n_rrp_vesicles",
            "syn_type_id",
            "u_syn"
          ],
          "range_prctile": 100,
          "fig_size": [
            11,
            3
          ],
          "fig_file": {
            "format": "png",
            "dpi": 600
          }
        },
        {
          "name": "Adjacency_Hex0",
          "fct": {
            "source": "adjacency",
            "kwargs": {
              "sel_src": {
                "node_set": "hex0"
              },
              "sel_dest": {
                "node_set": "hex0"
              }
            }
          },
          "res_sel": [
            "adj",
            "adj_cnt"
          ],
          "range_prctile": 95,
          "fig_size": [
            11,
            3
          ],
          "fig_file": {
            "format": "png",
            "dpi": 600
          }
        }
      ]
    }
