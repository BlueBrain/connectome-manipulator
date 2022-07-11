"""
Main module for model building:
  - main(...): Main function for running model building, given a model building config dict
    * Loads a SONATA connectome
    * Extracts connectivity specific data
    * Fits a model to data
    * Stores the data and model to disk
    * Visualizes and compares data and model
  - create_model_config_per_pathway(...): Pathway-specific model building wrapper function
"""

import importlib
import json
import os.path
import pandas as pd
import pickle
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from bluepysnap.circuit import Circuit

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_node_ids, get_edges_population


# NOT USED ANY MORE
# def get_model(model, model_inputs, model_params):
#     """Returns model function from string representation [so any model function can be saved to file]."""
#     input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
#     input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
#     model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters

#     inner_model_str = f'lambda {input_param_str}: {model}'
#     full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables

#     model_fct = eval(full_model_str) # Build function

#     # print(f'INFO: Model function: {inner_model_str}')

#     return model_fct


def create_model_config_per_pathway(model_config, grouped_by, src_sel_key='sel_src', dest_sel_key='sel_dest', group_fct=None, group_type=None):
    """Create model config dict for pathways between all pairs of groups (e.g. layer, mtype, ...)."""
    # Check model config
    assert 'model' in model_config.keys(), 'ERROR: "model" key missing in model_config!'
    assert 'working_dir' in model_config.keys(), 'ERROR: "working_dir" key missing in model_config!'
    assert 'out_dir' in model_config.keys(), 'ERROR: "out_dir" key missing in model_config!'
    assert 'name' in model_config['model'].keys(), 'ERROR: "name" key missing in model_config["model"]!'
    assert 'fct' in model_config['model'].keys(), 'ERROR: "fct" key missing in model_config["model"]!'
    assert 'kwargs' in model_config['model']['fct'].keys(), 'ERROR: "kwargs" key missing in model_config["model"]["fct"]!'

    if not isinstance(grouped_by, list):
        grouped_by = [grouped_by]
    grouped_by_name = '-'.join(grouped_by)

    if group_type is None or (isinstance(group_type, str) and group_type.upper() == 'BOTH'):
        src_grouping = True
        tgt_grouping = True
        print('INFO: Using PRE/POST grouping')
    elif isinstance(group_type, str) and group_type.upper() == 'PRE':
        src_grouping = True
        tgt_grouping = False
        grouped_by_name = f'PRE-{grouped_by_name}'
        print('INFO: Using PRE grouping only')
    elif isinstance(group_type, str) and group_type.upper() == 'POST':
        src_grouping = False
        tgt_grouping = True
        grouped_by_name = f'POST-{grouped_by_name}'
        print('INFO: Using POST grouping only')
    else:
        assert False, f'ERROR: Grouping type "{group_type}" unknown. Must be PRE, POST, or BOTH (=default).!'

    if group_fct is not None:
        if not isinstance(group_fct, list):
            group_fct = [group_fct]
        assert len(group_fct) == len(grouped_by), f'ERROR: Group functions to be provided for {len(grouped_by)} groups!'

    # Load circuit
    circuit_config = model_config['circuit_config']
    circuit = Circuit(circuit_config)
    print(f'INFO: Circuit loaded: {circuit_config}')

    # Select edge population [assuming exactly one edge population in given edges file]
    edges = get_edges_population(circuit)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    # Find pathways between pairs of groups (within current selection)
    if src_grouping:
        sel_src = model_config['model']['fct']['kwargs'].get(src_sel_key)
        if sel_src is not None:
            assert isinstance(sel_src, dict), 'ERROR: Source node selection must be a dict or empty!' # Otherwise, it cannot be merged with pathway selection

        assert np.all([g in src_nodes.property_names for g in grouped_by]), f'ERROR: "{grouped_by_name}" property not found in source nodes!'
        node_ids_src = get_node_ids(src_nodes, sel_src)
        src_types_nodes = src_nodes.get(node_ids_src, properties=grouped_by)
        src_types = sorted(src_types_nodes.groupby(grouped_by).indices.keys())
        if len(grouped_by) == 1:
            src_types = [[s] for s in src_types]

        if group_fct is not None:
            src_types_nodes_base = pd.concat([src_types_nodes[grouped_by[i]].apply(group_fct[i]) for i in range(len(group_fct))], axis=1)
            src_types_base = sorted(src_types_nodes_base.groupby(grouped_by).indices.keys())
            if len(grouped_by) == 1:
                src_types_base = [[s] for s in src_types_base]
            src_types = [[list(np.unique(src_types_nodes[grouped_by[i]][src_types_nodes_base[grouped_by[i]] == s[i]])) for i in range(len(group_fct))] for s in src_types_base]
        else:
            src_types_base = src_types
    else:
        src_types = src_types_base = [None]

    if tgt_grouping:
        sel_dest = model_config['model']['fct']['kwargs'].get(dest_sel_key)
        if sel_dest is not None:
            assert isinstance(sel_dest, dict), 'ERROR: Target node selection must be a dict or empty!' # Otherwise, it cannot be merged with pathway selection
        
        assert np.all([g in tgt_nodes.property_names for g in grouped_by]), f'ERROR: "{grouped_by_name}" property not found in target nodes!'
        node_ids_dest = get_node_ids(tgt_nodes, sel_dest)
        tgt_types_nodes = tgt_nodes.get(node_ids_dest, properties=grouped_by)
        tgt_types = sorted(tgt_types_nodes.groupby(grouped_by).indices.keys())
        if len(grouped_by) == 1:
            tgt_types = [[t] for t in tgt_types]

        if group_fct is not None:
            tgt_types_nodes_base = pd.concat([tgt_types_nodes[grouped_by[i]].apply(group_fct[i]) for i in range(len(group_fct))], axis=1)
            tgt_types_base = sorted(tgt_types_nodes_base.groupby(grouped_by).indices.keys())
            if len(grouped_by) == 1:
                tgt_types_base = [[t] for t in tgt_types_base]
            tgt_types = [[list(np.unique(tgt_types_nodes[grouped_by[i]][tgt_types_nodes_base[grouped_by[i]] == t[i]])) for i in range(len(group_fct))] for t in tgt_types_base]
        else:
            tgt_types_base = tgt_types
    else:
        tgt_types = tgt_types_base = [None]

    # Create list of model configs per pathway
    model_build_name = model_config['model']['name']
    model_config_pathways = []
    for s, sname in zip(src_types, src_types_base):
        for t, tname in zip(tgt_types, tgt_types_base):
            m_dict = deepcopy(model_config)
            if src_grouping:
                if sel_src is None:
                    m_dict['model']['fct']['kwargs'].update({src_sel_key: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in zip(grouped_by, s)}}) # tolist ... to get rid of hidden numpy data types (e.g., numpy.int64)
                else:
                    m_dict['model']['fct']['kwargs'][src_sel_key].update({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in zip(grouped_by, s)}) # tolist ... to get rid of hidden numpy data types (e.g., numpy.int64)
                pre_str = ["-".join([str(sn) for sn in sname]).replace(":", "_")]
            else:
                pre_str = []
            if tgt_grouping:
                if sel_dest is None:
                    m_dict['model']['fct']['kwargs'].update({dest_sel_key: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in zip(grouped_by, t)}}) # tolist ... to get rid of hidden numpy data types (e.g., numpy.int64)
                else:
                    m_dict['model']['fct']['kwargs'][dest_sel_key].update({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in zip(grouped_by, t)}) # tolist ... to get rid of hidden numpy data types (e.g., numpy.int64)
                post_str = ["-".join([str(tn) for tn in tname]).replace(":", "_")]
            else:
                post_str = []
            m_dict['model']['name'] += f'__{grouped_by_name}__{"-".join(pre_str + post_str)}'
            m_dict['working_dir'] = os.path.join(m_dict['working_dir'], model_build_name + f'__{grouped_by_name}_pathways')
            m_dict['out_dir'] = os.path.join(m_dict['out_dir'], model_build_name + f'__{grouped_by_name}_pathways')
            model_config_pathways.append(m_dict)

    print(f'INFO: Created model configurations for {len(model_config_pathways)} pathways between {len(src_types)}x{len(tgt_types)} {grouped_by_name}{"e" if grouped_by_name[-1] == "s" else ""}s')

    return model_config_pathways


def main(model_config_input, show_fig=False, force_recomp=False):  # pragma: no cover
    """Main entry point for connectome model building."""
    # Check model building config(s)
    if not isinstance(model_config_input, list):
        assert isinstance(model_config_input, dict), 'ERROR: model_config_input must be of type list or dict!'
        model_config_input = [model_config_input]

    if len(model_config_input) > 1:
        print(f'INFO: Building {len(model_config_input)} models: {model_config_input[0]["model"]["name"]}..{model_config_input[-1]["model"]["name"]}')

    # Run model building
    for midx, model_config in enumerate(model_config_input):
        if len(model_config_input) > 1:
            print(f'\n>>> BUILDING MODEL {midx + 1}/{len(model_config_input)}: {model_config["model"]["name"]} <<<')

        np.random.seed(model_config.get('seed', 123456))

        if np.isscalar(force_recomp):
            force_reextract = force_recomp
            force_rebuild = force_recomp
        else:
            assert len(force_recomp) == 2, 'ERROR: Two "force_recomp" entries expected!'
            force_reextract = force_recomp[0]
            force_rebuild = force_recomp[1]

        # Prepare saving
        model_build_name = model_config['model']['name']
        # out_dir = os.path.join(model_config['out_dir'], model_build_name) # Where to put output/figures
        out_dir = os.path.join(model_config['out_dir'], 'output', model_build_name) # Where to put output/figures
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data_dir = os.path.join(model_config['working_dir'], 'data') # Where to load/put data
        if not os.path.exists(os.path.split(data_dir)[0]):
            os.makedirs(os.path.split(data_dir)[0])

        model_dir = os.path.join(model_config['working_dir'], 'model') # Where to put model
        if not os.path.exists(os.path.split(model_dir)[0]):
            os.makedirs(os.path.split(model_dir)[0])

        # Initialize logger
        log_file = log.logging_init(out_dir, name='model_building')

        # Prepare computation module
        comp_source = model_config['model']['fct']['source']
        comp_kwargs = model_config['model']['fct']['kwargs']

        comp_module = importlib.import_module('connectome_manipulator.model_building.' + comp_source)
        log.log_assert(hasattr(comp_module, 'extract'), f'ERROR: Model building module "{comp_source}" requires extract() function!')
        log.log_assert(hasattr(comp_module, 'build'), f'ERROR: Model building module "{comp_source}" requires build() function!')
        log.log_assert(hasattr(comp_module, 'plot'), f'ERROR: Model building module "{comp_source}" requires plot() function!')

        # Load circuit
        circuit_config = model_config['circuit_config']
        circuit = Circuit(circuit_config)
        log.info(f'Circuit loaded: {circuit_config}')

        # Extract data (or load from file)
        data_file = os.path.join(data_dir, model_build_name + '.pickle')
        if os.path.exists(data_file) and not force_reextract:
            # Load from file
            log.info(f'Loading data from {data_file}')
            with open(data_file, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            # Compute & save to file
            t_start = time.time()
            data_dict = comp_module.extract(circuit, **comp_kwargs)
            log.info(f'<TIME ELAPSED (data extraction): {time.time() - t_start:.1f}s>')
            log.info(f'Writing data to {data_file}')
            if not os.path.exists(os.path.split(data_file)[0]):
                os.makedirs(os.path.split(data_file)[0])
            with open(data_file, 'wb') as f:
                pickle.dump(data_dict, f)

        # Build model (or load from file)
        model_file = os.path.join(model_dir, model_build_name + '.json')
        if os.path.exists(model_file) and not force_rebuild:
            # Load from file
            log.info(f'Loading model from {model_file}')
            model = model_types.AbstractModel.model_from_file(model_file)
        else:
            # Compute & save to file
            t_start = time.time()
            model = comp_module.build(**data_dict, **comp_kwargs)
            log.info(f'<TIME ELAPSED (model building): {time.time() - t_start:.1f}s>')
            log.info(f'Writing model to {model_file}')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_model(model_dir, model_build_name)

            # Save model config dict for reproducibility
            with open(os.path.join(out_dir, 'model_config.json'), 'w') as f:
                json.dump(model_config, f, indent=2)

        # Visualize data vs. model
        comp_module.plot(**data_dict, **comp_kwargs, model=model, out_dir=out_dir)

        if show_fig:
            plt.show()
        else:
            plt.close("all")


def main_wrapper():
    """Main function wrapper when called from command line.
       Command line arguments:
         model_config: Config file (.json), specifying the model building settings
         force_reextract (optional): Disable (0; default) or enable (1) forced re-extraction of data, in case already existing
         force_rebuild (optional): Disable (0) or enable (1) forced model re-building, in case model already exists (default: force_reextract)
    """

    # Parse inputs
    args = sys.argv[1:]
    if len(args) < 1:
        print(f'Usage: {__file__} <model_config.json> [force_reextract] [force_rebuild]')
        sys.exit(2)

    # Load config dict
    with open(args[0], 'r') as f:
        config_dict = json.load(f)

    # Read force_reextract flag
    if len(args) > 1:
        force_reextract = bool(int(args[1]))
    else:
        force_reextract = False

    # Read force_rebuild flag
    if len(args) > 2:
        force_rebuild = bool(int(args[2]))
    else:
        force_rebuild = force_reextract

    # Call main function
    main(config_dict, show_fig=False, force_recomp=[force_reextract, force_rebuild])


if __name__ == "__main__":
    main_wrapper()
