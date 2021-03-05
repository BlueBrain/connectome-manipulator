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
import importlib
import matplotlib.pyplot as plt

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


""" Main entry point for connectome model building """
def main(model_config, show_fig=False, force_recomp=False):
    
    np.random.seed(model_config.get('seed', 123456))
    
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
    if os.path.exists(data_file) and not force_recomp:
        # Load from file
        print(f'INFO: Loading data from {data_file}')
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        # Compute & save to file
        data_dict = comp_module.extract(circuit, **comp_kwargs)
        print(f'INFO: Writing data to {data_file}')
        if not os.path.exists(os.path.split(data_file)[0]):
            os.makedirs(os.path.split(data_file)[0])
        with open(data_file, 'wb') as f:
            pickle.dump(data_dict, f)
    
    # Build model (or load from file)
    model_file = os.path.join(model_dir, model_build_name + '.pickle')
    if os.path.exists(model_file) and not force_recomp:
        # Load from file
        print(f'INFO: Loading model from {model_file}')
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
    else:
        # Compute & save to file
        model_dict = comp_module.build(**data_dict, **comp_kwargs)
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
