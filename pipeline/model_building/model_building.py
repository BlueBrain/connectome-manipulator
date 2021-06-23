# Connectome model building
#   Main module for
#   - loading a SONATA connectome
#   - extracting connectivity specific data
#   - building a data-based model
#   - storing the data and model to disk (to be used by the manipulation pipeline)
#   - visualizing and comparing data and model

import numpy as np
from bluepysnap.circuit import Circuit
import os.path
import pickle
import sys
import time
import importlib
import matplotlib.pyplot as plt
from copy import deepcopy

""" Returns model function from string representation [so any model function can be saved to file] """
def get_model(model, model_inputs, model_params):
    
    input_str = ','.join(model_inputs + ['model_params=model_params']) # String representation of input variables
    input_param_str = ','.join(model_inputs + list(model_params.keys())) # String representation of input variables and model parameters
    model_param_str = ','.join(model_inputs + ['**model_params']) # String representation propagating model parameters
    
    inner_model_str = f'lambda {input_param_str}: {model}'
    full_model_str = f'lambda {input_str}: ({inner_model_str})({model_param_str})' # Use nested lambdas to bind local variables
    
    model_fct = eval(full_model_str) # Build function
    
    # print(f'INFO: Model function: {inner_model_str}')
    
    return model_fct


""" Create model config dict for all pairs of pathways (m-types) """
def create_model_config_per_pathway(model_config, src_sel_key='sel_src', dest_sel_key='sel_dest'):
    
    # Check model config
    assert 'model' in model_config.keys(), 'ERROR: "model" key missing in model_config!'
    assert 'working_dir' in model_config.keys(), 'ERROR: "working_dir" key missing in model_config!'
    assert 'out_dir' in model_config.keys(), 'ERROR: "out_dir" key missing in model_config!'
    assert 'name' in model_config['model'].keys(), 'ERROR: "name" key missing in model_config["model"]!'
    assert 'fct' in model_config['model'].keys(), 'ERROR: "fct" key missing in model_config["model"]!'
    assert 'kwargs' in model_config['model']['fct'].keys(), 'ERROR: "kwargs" key missing in model_config["model"]["fct"]!'
    
    # Load circuit
    circuit_config = model_config['circuit_config']
    circuit = Circuit(circuit_config)
    print(f'INFO: Circuit loaded: {circuit_config}')
    
    # Select edge population [assuming exactly one edge population in given edges file]
    assert len(circuit.edges.population_names) == 1, 'ERROR: Only a single edge population per file supported for modelling!'
    edges = circuit.edges[circuit.edges.population_names[0]]
    
    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
        
    # Find pathways between pairs of m-types (within current selection)
    sel_src = model_config['model']['fct']['kwargs'].get(src_sel_key)
    sel_dest = model_config['model']['fct']['kwargs'].get(dest_sel_key)
    
    src_mtypes = list(np.unique(src_nodes.get(sel_src, properties='mtype')))
    tgt_mtypes = list(np.unique(tgt_nodes.get(sel_dest, properties='mtype')))
    
    # Create list of model configs per pathway
    model_build_name = model_config['model']['name']
    model_config_pathways = []
    for s in src_mtypes:
        for t in tgt_mtypes:
            m_dict = deepcopy(model_config)
            m_dict['model']['fct']['kwargs'].update({src_sel_key: {'mtype': s}, dest_sel_key: {'mtype': t}})
            m_dict['model']['name'] += f'__{s.replace(":", "_")}-{t.replace(":", "_")}'
            m_dict['working_dir'] = os.path.join(m_dict['working_dir'], model_build_name + '__Per_Pathway')
            m_dict['out_dir'] = os.path.join(m_dict['out_dir'], model_build_name + '__Per_Pathway')
            model_config_pathways.append(m_dict)
    
    print(f'INFO: Created model configurations for {len(model_config_pathways)} pathways between {len(src_mtypes)}x{len(tgt_mtypes)} m-types')
    
    return model_config_pathways


""" Main entry point for connectome model building """
def main(model_config_input, show_fig=False, force_recomp=False):
    
    if not isinstance(model_config_input, list):
        assert isinstance(model_config_input, dict), 'ERROR: model_config_input must be of type list or dict!'
        model_config_input = [model_config_input]
    
    if len(model_config_input) > 1:
        print(f'INFO: Building {len(model_config_input)} models: {model_config_input[0]["model"]["name"]}..{model_config_input[-1]["model"]["name"]}')
    
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
        
        # Load circuit
        circuit_config = model_config['circuit_config']
        circuit = Circuit(circuit_config)
        print(f'INFO: Circuit loaded: {circuit_config}')
        
        # Prepare saving
        model_build_name = model_config['model']['name']
        out_dir = os.path.join(model_config['out_dir'], model_build_name) # Where to put figures
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        data_dir = os.path.join(model_config['working_dir'], 'data') # Where to load/put data
        if not os.path.exists(os.path.split(data_dir)[0]):
            os.makedirs(os.path.split(data_dir)[0])
        
        model_dir = os.path.join(model_config['working_dir'], 'model') # Where to put model
        if not os.path.exists(os.path.split(model_dir)[0]):
            os.makedirs(os.path.split(model_dir)[0])
        
        # Prepare computation module
        import_root = os.path.split(__file__)[0]
        sys.path.insert(0, import_root)
        
        comp_source = model_config['model']['fct']['source']
        comp_kwargs = model_config['model']['fct']['kwargs']
        
        comp_module = importlib.import_module(comp_source)
        
        # Extract data (or load from file)
        data_file = os.path.join(data_dir, model_build_name + '.pickle')
        if os.path.exists(data_file) and not force_reextract:
            # Load from file
            print(f'INFO: Loading data from {data_file}')
            with open(data_file, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            # Compute & save to file
            t_start = time.time()
            data_dict = comp_module.extract(circuit, **comp_kwargs)
            print(f'<TIME ELAPSED (data extraction): {time.time() - t_start:.1f}s>')
            print(f'INFO: Writing data to {data_file}')
            if not os.path.exists(os.path.split(data_file)[0]):
                os.makedirs(os.path.split(data_file)[0])
            with open(data_file, 'wb') as f:
                pickle.dump(data_dict, f)
        
        # Build model (or load from file)
        model_file = os.path.join(model_dir, model_build_name + '.pickle')
        if os.path.exists(model_file) and not force_rebuild:
            # Load from file
            print(f'INFO: Loading model from {model_file}')
            with open(model_file, 'rb') as f:
                model_dict = pickle.load(f)
        else:
            # Compute & save to file
            t_start = time.time()
            model_dict = comp_module.build(**data_dict, **comp_kwargs)
            print(f'<TIME ELAPSED (model building): {time.time() - t_start:.1f}s>')
            print(f'INFO: Writing model to {model_file}')
            if not os.path.exists(os.path.split(model_file)[0]):
                os.makedirs(os.path.split(model_file)[0])
            with open(model_file, 'wb') as f:
                pickle.dump(model_dict, f)
        
        # Visualize data vs. model
        comp_module.plot(**data_dict, **model_dict, **comp_kwargs, out_dir=out_dir)
        
        if show_fig:
            plt.show()
        else:
            plt.close("all")
