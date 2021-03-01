# Connectome model building
#   Main module for
#   - loading a SONATA connectome
#   - extracting connectivity specific data
#   - building a data-based model
#   - storing the data and model to disk (to be used by the manipulation pipeline)
#   - visualizing and comparing data and model

from bluepysnap.circuit import Circuit
import os.path
import matplotlib.pyplot as plt

""" Main entry point for connectome model building """
def main(model_config, show_fig=False, force_recomp=False):
    
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

    comp_source = model_config['fct']['source']
    comp_kwargs = model_config['fct']['kwargs']

    comp_module = importlib.import_module(comp_source)
    
    # Extract data
    data_dict = comp_module.extract(circuit, data_dir, model_build_name, force_recomp, **comp_kwargs)
    
    # Build model
    model_dict = comp_module.build(data_dict, model_dir, model_build_name, force_recomp, **comp_kwargs)
    
    # Visualize data vs. model
    vis_dict = comp_module.plot(data_dict, model_dict, out_dir, **comp_kwargs)
    
    if show_fig:
        plt.show()
    else:
        plt.close("all")
