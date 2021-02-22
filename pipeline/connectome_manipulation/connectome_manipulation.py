# Connectome manipulation:
#   Main module for
#   - loading a SONATA connectome using SNAP
#   - applying manipulations to the connectome
#   - writing back the manipulated connectome and a new circuit config

from bluepysnap.circuit import Circuit
from bluepysnap.sonata_constants import Node
from bluepysnap.sonata_constants import Edge
import sys
import os
import subprocess
import importlib
import numpy as np
import resource
import time
import json
import logging
from datetime import datetime

""" Load SONATA circuit using SNAP """
def load_circuit(circuit_config, N_split=1):
    
    # Load circuit
    logging.info(f'Loading circuit from {circuit_config} (N_split={N_split})')
        
    c = Circuit(circuit_config)
    nodes = c.nodes['All']
    edges = c.edges['default']
    
    nodes_file = c.config['networks']['nodes'][0]['nodes_file']
    edges_file = c.config['networks']['edges'][0]['edges_file']
    
    node_ids = nodes.ids()
    node_ids_split = np.split(node_ids, np.cumsum([np.ceil(len(node_ids) / N_split).astype(int)] * (N_split - 1)))
    
    return nodes, nodes_file, node_ids_split, edges, edges_file 


""" Apply manipulation to connectome (edges_table) as specified in the manip_config """
def apply_manipulation(edges_table, nodes, manip_config, aux_dict):

    import_root = os.path.split(__file__)[0]
    sys.path.insert(0, import_root)
    
    logging.info(f'APPLYING MANIPULATION "{manip_config["manip"]["name"]}"')
    for m_step in range(len(manip_config['manip']['fcts'])):
        manip_source = manip_config['manip']['fcts'][m_step]['source']
        manip_kwargs = manip_config['manip']['fcts'][m_step]['kwargs']
        logging.info(f'>>Step {m_step + 1} of {len(manip_config["manip"]["fcts"])}: source={manip_source}, kwargs={manip_kwargs}')
        
        manip_module = importlib.import_module(manip_source)
        edges_table = manip_module.apply(edges_table, nodes, aux_dict, **manip_kwargs)
    
    return edges_table


""" Write edge properties table to parquet file """
def edges_to_parquet(edges_table, output_file):
    
    edges_table = edges_table.rename(columns={'@target_node': 'connected_neurons_post', '@source_node': 'connected_neurons_pre'}) # Convert column names
    edges_table['synapse_type_id'] = 0 # Add type ID, required for SONATA
    edges_table.to_parquet(output_file, index=False)


""" Convert parquet file(s) to SONATA format (using parquet-converters tool; recomputes indices!!) """
def parquet_to_sonata(input_file_list, output_file, nodes_file):
    
    logging.info(f'Converting {len(input_file_list)} .parquet file(s) to SONATA')
    input_files = ' '.join(input_file_list)
    
    proc = subprocess.Popen(f'module load unstable parquet-converters;\
                              parquet2hdf5 --format SONATA --from {nodes_file} All --to {nodes_file} All -o {output_file} {input_files}',
                              shell=True, stdout=subprocess.PIPE)
    logging.info(proc.communicate()[0].decode())


""" Create new text file from template with replacements """
def create_new_file_from_template(new_file, template_file, replacements_dict):
    
    logging.info(f'Creating file {new_file}')
    with open(template_file, 'r') as file:
        content = file.read()
    
    for (src, dest) in replacements_dict.items():
        content = content.replace(src, dest)
    
    with open(new_file, 'w') as file:
        file.write(content)


def resource_profiling(enabled=False, description=''):
    
    if not enabled:
        return

    mem_curr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if not hasattr(resource_profiling, 'mem_before'):
        mem_diff = None
    else:
        mem_diff = mem_curr - resource_profiling.mem_before
    resource_profiling.mem_before = mem_curr

    if not hasattr(resource_profiling, 't_init'):
        resource_profiling.t_init = time.time()
        resource_profiling.t_start = resource_profiling.t_init
        t_tot = None
        t_dur = None
    else:
        t_end = time.time()
        t_tot = t_end - resource_profiling.t_init
        t_dur = t_end - resource_profiling.t_start
        resource_profiling.t_start = t_end

    if len(description) > 0:
        description = ' [' + description + ']'
    
    field_width = 36 + max(len(description) - 14, 0)
    
    log_msg = '\n'
    log_msg = log_msg + '*' * field_width + '\n'
    log_msg = log_msg + '* ' + 'RESOURCE PROFILING{}'.format(description).ljust(field_width - 4) + ' *' + '\n'
    log_msg = log_msg + '*' * field_width + '\n'

    log_msg = log_msg + '* ' + 'Max. memory usage (GB):' + '{:.3f}'.format(mem_curr).rjust(field_width - 27) + ' *' + '\n'
    
    if not mem_diff is None:
        log_msg = log_msg + '* ' + 'Max. memory diff. (GB):' + '{:.3f}'.format(mem_diff).rjust(field_width - 27) + ' *' + '\n'

    if not t_tot is None and not t_dur is None:
        log_msg = log_msg + '*' * field_width + '\n'
        
        if t_tot > 3600:
            t_tot = t_tot / 3600
            t_tot_unit = 'h'
        else:
            t_tot_unit = 's'
        log_msg = log_msg + '* ' + f'Total time ({t_tot_unit}):        ' + '{:.3f}'.format(t_tot).rjust(field_width - 27) + ' *' + '\n'
        
        if t_dur > 3600:
            t_dur = t_dur / 3600
            t_dur_unit = 'h'
        else:
            t_dur_unit = 's'
        log_msg = log_msg + '* ' + f'Elapsed time ({t_dur_unit}):      ' + '{:.3f}'.format(t_dur).rjust(field_width - 27) + ' *' + '\n'
    
    log_msg = log_msg + '*' * field_width + '\n'
    
    logging.profiling(log_msg)


""" Initialize logger (with custom log level for profiling and assert with logging) """
def logging_init(circuit_path):
    
    # Configure logging
    log_file = os.path.join(circuit_path, 'logs', f'{__name__.split(".")[-1]}.{datetime.today().strftime("%Y%m%dT%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(module)s] %(levelname)s: %(message)s'))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    
    # Add custom log level for profiling
    profiling_log_level_name = 'PROFILING'
    setattr(logging, profiling_log_level_name.upper(), logging.INFO + 5)
    setattr(logging, profiling_log_level_name.lower(), lambda msg, *args, **kwargs: logging.log(logging.PROFILING, msg, *args, **kwargs))
    logging.addLevelName(logging.PROFILING, profiling_log_level_name)
    
    # Add assert with logging
    def log_assert(cond, msg):
        if not cond:
            logging.error(msg)
        assert cond, msg
    setattr(logging, 'log_assert', log_assert)


""" Main entry point for circuit manipulations [OPTIMIZATION FOR HUGE CONNECTOMES: SPLIT INTO N PARTS (OPTIONAL)] """
def main(manip_config, do_profiling=False):

    # Initialize logger
    logging_init(manip_config['circuit_path'])
    
    # Initialize profiler
    resource_profiling(do_profiling, 'initial')

    # Load circuit
    circuit_name = os.path.split(manip_config['circuit_path'])[-1]
    circuit_config =  os.path.join(manip_config['circuit_path'], 'CircuitConfig') # [Using default name]
    sonata_config =  os.path.join(manip_config['circuit_path'], 'sonata', 'circuit_config.json') # [Using default name]
    N_split = max(manip_config.get('N_split_nodes', 1), 1)
    
    nodes, nodes_file, node_ids_split, edges, edges_file = load_circuit(sonata_config, N_split)
    
    # Prepare output parquet path
    parquet_path = os.path.join(os.path.split(edges_file)[0], 'parquet')
    if not os.path.exists(parquet_path):
        os.makedirs(parquet_path)
    
    # Start processing
    np.random.seed(manip_config.get('seed', 123456))
    
    parquet_file_list = []
    N_syn_in = []
    N_syn_out = []
    aux_dict = {} # Auxiliary dict to pass information from one split iteration to another
    for i_split, split_ids in enumerate(node_ids_split):
        
        # Load edge table containing all edge (=synapse) properties
        edges_table = edges.afferent_edges(split_ids, properties=sorted(edges.property_names))
        logging.info(f'Split {i_split + 1}/{N_split}: Loaded {edges_table.shape[0]} synapses with {edges_table.shape[1]} properties between {len(split_ids)} neurons')
        N_syn_in.append(edges_table.shape[0])
        resource_profiling(do_profiling, f'loaded-{i_split + 1}/{N_split}')
        
        # Apply connectome manipulation
        aux_dict.update({'N_split': N_split, 'i_split': i_split, 'split_ids': split_ids})
        edges_table_manip = apply_manipulation(edges_table, nodes, manip_config, aux_dict)
#         logging.log_assert(edges_table_manip['@target_node'].is_monotonic_increasing, 'ERROR: Target nodes not monotonically increasing!') # [TESTING/DEBUGGING]
        N_syn_out.append(edges_table_manip.shape[0])
        resource_profiling(do_profiling, f'manipulated-{i_split + 1}/{N_split}')
        
        # Write back connectome to .parquet file
        parquet_file_manip = os.path.splitext(os.path.join(parquet_path, os.path.split(edges_file)[1]))[0] + f'_{manip_config["manip"]["name"]}' + (f'.{i_split:04d}' if N_split > 1 else '') + '.parquet'
        edges_to_parquet(edges_table_manip, parquet_file_manip)
        parquet_file_list.append(parquet_file_manip)
        resource_profiling(do_profiling, f'saved-{i_split + 1}/{N_split}')
    
    logging.info(f'Total input/output synapse counts: {np.sum(N_syn_in)}/{np.sum(N_syn_out)}\n')
    
    # Convert .parquet file(s) to SONATA file
    edges_file_manip = os.path.splitext(edges_file)[0] + f'_{manip_config["manip"]["name"]}' + os.path.splitext(edges_file)[1]
    parquet_to_sonata(parquet_file_list, edges_file_manip, nodes_file)
    
    # Create new sonata config
    edge_fn = os.path.split(edges_file)[1]
    edge_fn_manip = os.path.split(edges_file_manip)[1]
    sonata_config_manip = os.path.splitext(sonata_config)[0] + f'_{manip_config["manip"]["name"]}' + os.path.splitext(sonata_config)[1]
    config_replacement = {edge_fn: edge_fn_manip}
    create_new_file_from_template(sonata_config_manip, sonata_config, config_replacement)
    
    # Write manipulation config to JSON file
    json_file = os.path.join(os.path.split(sonata_config)[0], f'manip_config_{manip_config["manip"]["name"]}.json')
    with open(json_file, 'w') as f:
        json.dump(manip_config, f, indent=2)
    logging.info(f'Creating file {json_file}')
    
    # Create new symlink (using rel. path) and circuit config
    with open(circuit_config, 'r') as file: # Read CircuitConfig
        config = file.read()
    nrn_path = list(filter(lambda x: x.find('nrnPath') >= 0, config.splitlines()))[0].replace('nrnPath', '').strip() # Extract path to edges file    

    symlink_src = os.path.relpath(edges_file_manip, os.path.split(nrn_path)[0])
    symlink_dst = os.path.join(os.path.split(nrn_path)[0], os.path.splitext(edge_fn_manip)[0] + '.sonata')
    if os.path.isfile(symlink_dst):
        os.remove(symlink_dst) # Remove if already exists
    os.symlink(symlink_src, symlink_dst)
    logging.info(f'Creating symbolic link ...{symlink_dst} -> {symlink_src}')
    
    circuit_config_manip = os.path.splitext(circuit_config)[0] + f'_{manip_config["manip"]["name"]}' + os.path.splitext(circuit_config)[1]
    config_replacement = {nrn_path: symlink_dst}
    create_new_file_from_template(circuit_config_manip, circuit_config, config_replacement)
    
    resource_profiling(do_profiling, 'final')
    