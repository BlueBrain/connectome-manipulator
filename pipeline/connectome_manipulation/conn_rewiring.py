# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

from model_building import model_building
from model_building import conn_prob
import logging
import numpy as np
import os.path
import pickle

""" Rewiring (interchange) of connections between pairs of neurons based on given conn. prob. model (keeping synapses & number of ingoing connections) """
def apply(edges_table, nodes, aux_dict, src_node_sel, tgt_node_sel, prob_model_file, delay_model_file=None):
    
    # Load connection probability model
    logging.log_assert(os.path.exists(prob_model_file), 'Conn. prob. model file not found!')
    logging.info(f'INFO: Loading conn. prob. model from {prob_model_file}')
    with open(prob_model_file, 'rb') as f:
        prob_model_dict = pickle.load(f)
    p_model = model_building.get_model(prob_model_dict['model'], prob_model_dict['model_inputs'], prob_model_dict['model_params'])
    
    # Determine model order
    if len(prob_model_dict['model_inputs']) == 0:
        model_order = 1 # Constant conn. prob. (no inputs)
    elif len(prob_model_dict['model_inputs']) == 1:
        model_order = 2 # Distance-dependent conn. prob. (1 input: distance)
    else:
        logging.log_assert(False, 'Model order could not be determined!')
    
    # Load delay model (optional)
    if not delay_model_file is None:
        logging.log_assert(os.path.exists(delay_model_file), 'Delay model file not found!')
        logging.info(f'INFO: Loading delay model from {delay_model_file}')
        with open(delay_model_file, 'rb') as f:
            delay_model_dict = pickle.load(f)
        d_model = model_building.get_model(delay_model_dict['model'], delay_model_dict['model_inputs'], delay_model_dict['model_params'])
        logging.log_assert(len(delay_model_dict['model_inputs']) == 2, 'Distance-dependent delay model with two inputs (d, type) expected!')
    else:
        d_model = None
    
    # Run connection rewiring
    src_node_ids = nodes.ids(src_node_sel)
    tgt_node_ids = nodes.ids(tgt_node_sel)
    logging.info('Sampling neurons for connection rewiring...')
    for tgt in tgt_node_ids:
        syn_sel_idx = np.isin(edges_table['@target_node'], tgt)
        if not np.any(syn_sel_idx):
            continue # Nothing to rewire (no synapses on target node)
        
        src, src_idx = np.unique(edges_table.loc[syn_sel_idx, '@source_node'], return_inverse=True)
        num_src = len(src) # Number of currently existing sources for given target node
        
        # TODO: Check that synapse classes of src neurons are the same as the ones from src_node_ids!
        #       (EXC (INH) synapses on tgt must be connected to EXC (INH) pre-synaptic neurons!!)
        
        # Sample new num_src presynaptic neurons from full list of source nodes according to conn. prob.
        # (keeping the same number of ingoing connections)
        if model_order == 1: # Constant conn. prob. (no inputs)
            p_src = np.full(len(src_node_ids), p_model())
        elif model_order == 2: # Distance-dependent conn. prob. (1 input: distance)
            d = conn_prob.compute_dist_matrix(nodes, src_node_ids, [tgt])
            p_src = p_model(d).flatten()
            p_src[np.isnan(p_src)] = 0.0
        else:
            logging.log_assert(False, f'Model order {model_order} not supported!')
        
        src_new = np.random.choice(src_node_ids, num_src, replace=False, p=p_src/np.sum(p_src))
        
        # Assign new source nodes = rewiring
        edges_table.loc[syn_sel_idx, '@source_node'] = src_new[src_idx]
        
        # Assign new distance-dependent delay (optional)
        if not d_model is None:
            # Determine distance from source neuron (soma) to synapse on target neuron
            src_new_pos = nodes.positions(src_new).to_numpy()
            syn_pos = edges_table.loc[syn_sel_idx, ['afferent_center_x', 'afferent_center_y', 'afferent_center_z']].to_numpy() # Synapse position on post-synaptic dendrite
            syn_dist = np.sqrt(np.sum((syn_pos - src_new_pos[src_idx, :])**2, 1))
            delay_new = np.random.normal(d_model(syn_dist, 'mean'), d_model(syn_dist, 'std'))
            assert np.all(delay_new > 0.0), 'ERROR: Delay must be larger than 0ms!'
            
            edges_table.loc[syn_sel_idx, 'delay'] = delay_new
    
    return edges_table
