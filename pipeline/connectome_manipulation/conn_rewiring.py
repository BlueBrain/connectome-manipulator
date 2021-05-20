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
from scipy.stats import truncnorm

""" Rewiring (interchange) of connections between pairs of neurons based on given conn. prob. model (keeping synapses and optionally, number of ingoing connections) """
def apply(edges_table, nodes, aux_dict, syn_class, prob_model_file, sel_src=None, sel_dest=None, delay_model_file=None, pos_map_file=None, keep_indegree=True, amount_pct=100.0):
    
    logging.log_assert(syn_class in ['EXC', 'INH'], f'Synapse class "{syn_class}" not supported (must be "EXC" or "INH")!')
    logging.log_assert(amount_pct >= 0.0 and amount_pct <= 100.0, 'amount_pct out of range!')
    
    # Load connection probability model
    logging.log_assert(os.path.exists(prob_model_file), 'Conn. prob. model file not found!')
    logging.info(f'Loading conn. prob. model from {prob_model_file}')
    with open(prob_model_file, 'rb') as f:
        prob_model_dict = pickle.load(f)
    p_model = model_building.get_model(prob_model_dict['model'], prob_model_dict['model_inputs'], prob_model_dict['model_params'])
    
    if len(prob_model_dict['model_inputs']) == 0:
        model_order = 1 # Constant conn. prob. (no inputs)
    elif len(prob_model_dict['model_inputs']) == 1:
        model_order = 2 # Distance-dependent conn. prob. (1 input: distance)
    elif len(prob_model_dict['model_inputs']) == 2:
        model_order = 3 # Bipolar distance-dependent conn. prob. (2 inputs: distance, z offset)
    elif len(prob_model_dict['model_inputs']) == 3:
        model_order = 4 # Offset-dependent conn. prob. (3 inputs: x/y/z offsets)
    else:
        logging.log_assert(False, 'Model order could not be determined!')
    logging.info(f'Model order {model_order} detected')
    
    # Load delay model (optional)
    if not delay_model_file is None:
        logging.log_assert(os.path.exists(delay_model_file), 'Delay model file not found!')
        logging.info(f'Loading delay model from {delay_model_file}')
        with open(delay_model_file, 'rb') as f:
            delay_model_dict = pickle.load(f)
        d_model = model_building.get_model(delay_model_dict['model'], delay_model_dict['model_inputs'], delay_model_dict['model_params'])
        logging.log_assert(len(delay_model_dict['model_inputs']) == 2, 'Distance-dependent delay model with two inputs (d, type) expected!')
    else:
        d_model = None
        logging.info(f'No delay model provided')
    
    # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
    if not pos_map_file is None:
        logging.log_assert(model_order >= 2, 'Position mapping only applicable for 2nd-order models and higher!')
        logging.log_assert((nodes[0].name == nodes[1].name) or (nodes[0].ids().min() > nodes[1].ids().max()) or (nodes[0].ids().max() < nodes[1].ids().min()), 'Position mapping only supported for same source/taget node population or non-overlapping id ranges!')
        logging.log_assert(os.path.exists(pos_map_file), 'Position mapping model file not found!')
        logging.info(f'Loading position map from {pos_map_file}')
        with open(pos_map_file, 'rb') as f:
            pos_map_dict = pickle.load(f)
        pos_map = model_building.get_model(pos_map_dict['model'], pos_map_dict['model_inputs'], pos_map_dict['model_params'])
    else:
        pos_map = None
        logging.info(f'No position mapping model provided')
    
    # Determine source/target nodes for rewiring
    src_class = nodes[0].get(sel_src, properties='synapse_class')
    src_node_ids = src_class[src_class == syn_class].index.to_numpy() # Select only source nodes with given synapse class (EXC/INH)
    logging.log_assert(len(src_node_ids) > 0, f'No {syn_class} source nodes found!')
    if model_order >= 2:
        src_pos = conn_prob.get_neuron_positions(nodes[0].positions if pos_map is None else pos_map, [src_node_ids])[0] # Get neuron positions (incl. position mapping, if provided)
    
    tgt_node_ids = nodes[1].ids(sel_dest)
    num_tgt = np.round(amount_pct * len(tgt_node_ids) / 100).astype(int)
    tgt_sel = np.random.permutation([True] * num_tgt + [False] * (len(tgt_node_ids) - num_tgt))
    tgt_node_ids = tgt_node_ids[tgt_sel] # Select subset of neurons (keeping order)
    
    logging.info(f'Rewiring afferent {syn_class} connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons (sel_src={sel_src}, sel_dest={sel_dest}, keep_indegree={keep_indegree})')
    
    # Run connection rewiring
    warning_syn_count_diff = [] # Keep track of synapse count mismatch to provide a warning
    for tgt in tgt_node_ids:
        syn_sel_idx = edges_table['@target_node'] == tgt
        
        # Rewire only synapses of given class (EXC/INH)
        if syn_class == 'EXC':
            syn_sel_idx = np.logical_and(syn_sel_idx, edges_table['syn_type_id'] >= 100)
        elif syn_class == 'INH':
            syn_sel_idx = np.logical_and(syn_sel_idx, edges_table['syn_type_id'] < 100)
        else:
            logging.log_assert(False, f'Synapse class {syn_class} not supported!')
        
        num_syn = np.sum(syn_sel_idx)
        if num_syn == 0:
            warning_syn_count_diff.append(num_syn)
            continue # Nothing to rewire (no synapses on target node)
        
        # Determine conn. prob. of all source nodes to be connected with target node
        if model_order == 1: # Constant conn. prob. (no inputs)
            p_src = np.full(len(src_node_ids), p_model())
        elif model_order == 2: # Distance-dependent conn. prob. (1 input: distance)
            tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_map is None else pos_map, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
            d = conn_prob.compute_dist_matrix(src_pos, tgt_pos)
            p_src = p_model(d).flatten()
        elif model_order == 3: # Bipolar distance-dependent conn. prob. (2 inputs: distance, z offset)
            tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_map is None else pos_map, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
            d = conn_prob.compute_dist_matrix(src_pos, tgt_pos)
            bip = conn_prob.compute_bip_matrix(src_pos, tgt_pos)
            p_src = p_model(d, bip).flatten()
        elif model_order == 4: # Offset-dependent conn. prob. (3 inputs: x/y/z offsets)
            tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_map is None else pos_map, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
            dx, dy, dz = conn_prob.compute_offset_matrices(src_pos, tgt_pos)
            p_src = p_model(dx, dy, dz).flatten()
        else:
            logging.log_assert(False, f'Model order {model_order} not supported!')
        
        p_src[np.isnan(p_src)] = 0.0 # Exclude invalid values
        p_src[src_node_ids == tgt] = 0.0 # Exclude autapses
        
        # Sample new presynaptic neurons from list of source nodes according to conn. prob.
        if keep_indegree: # Keep the same number of ingoing connections (and #synapses/connection)
            src, src_syn_idx = np.unique(edges_table.loc[syn_sel_idx, '@source_node'], return_inverse=True)
            num_src = len(src) # Number of currently existing sources for given target node
            logging.log_assert(len(src_node_ids) >= num_src, f'Not enough source neurons for target neuron {tgt} available for rewiring!')
            
            src_new = np.random.choice(src_node_ids, size=num_src, replace=False, p=p_src/np.sum(p_src)) # New source node IDs per connection
        else: # Number of ingoing connections (and #synapses/connection) NOT kept the same
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel] # New source node IDs per connection
            
            if len(src_new) == 0: # No connections to assign existing synapses to
                warning_syn_count_diff.append(len(src_new) - num_syn)
                
                # At least one connection required to assign synapses to => Select one source node (randomly, taking p_src into account)
                src_new = np.random.choice(src_node_ids, size=1, replace=False, p=p_src/np.sum(p_src))
                src_syn_idx = np.zeros(num_syn, dtype=int)
            elif num_syn < len(src_new): # Too many connections to be realized with existing synapses
                warning_syn_count_diff.append(len(src_new) - num_syn)
                
                # Reduce to num_syn connections with 1 syn/conn (random subsample, taking p_src into account)
                sub_sel = np.sort(np.random.choice(len(src_new), size=num_syn, replace=False, p=p_src[src_new_sel]/np.sum(p_src[src_new_sel])))
                src_new = src_new[sub_sel]
                src_syn_idx = np.arange(num_syn)
            else: # Connections can be realized with existing synapses => Assign synapses to connections
                num_syn_per_conn = num_syn / len(src_new) # Distribute equally
                src_syn_idx = np.floor(np.arange(num_syn) / num_syn_per_conn).astype(int)
                
        # Assign new source nodes = rewiring
        edges_table.loc[syn_sel_idx, '@source_node'] = src_new[src_syn_idx] # Source node IDs per connection expanded to synapses
        
        # Assign new distance-dependent delay, drawn from truncated normal distribution (optional)
        if not d_model is None:
            # Determine distance from source neuron (soma) to synapse on target neuron
            src_new_pos = nodes[0].positions(src_new).to_numpy()
            syn_pos = edges_table.loc[syn_sel_idx, ['afferent_center_x', 'afferent_center_y', 'afferent_center_z']].to_numpy() # Synapse position on post-synaptic dendrite
            syn_dist = np.sqrt(np.sum((syn_pos - src_new_pos[src_syn_idx, :])**2, 1))
            
            d_mean = d_model(syn_dist, 'mean')
            d_std = d_model(syn_dist, 'std')
            d_min = d_model(syn_dist, 'min')
            delay_new = truncnorm(a=(d_min - d_mean) / d_std, b=np.inf, loc=d_mean, scale=d_std).rvs()            
            edges_table.loc[syn_sel_idx, 'delay'] = delay_new
    
    if len(warning_syn_count_diff) > 0:
        warning_syn_count_diff = np.array(warning_syn_count_diff)
        cnt_str1 = f'{np.sum(warning_syn_count_diff==0)}x: Nothing to rewire, since no synapses' if np.sum(warning_syn_count_diff==0) > 0 else ''
        cnt_str2 = f'{np.sum(warning_syn_count_diff<0)}x: Increased #conn to 1 instead of 0' if np.sum(warning_syn_count_diff<0) > 0 else ''
        cnt_str3 = f'{np.sum(warning_syn_count_diff>0)}x: Decreased #conn by {np.mean(warning_syn_count_diff[warning_syn_count_diff>0]):.1f} on avg)' if np.sum(warning_syn_count_diff>0) > 0 else ''
        logging.warning(f'Wrong number of {syn_class} synapses to realize intended connections at {len(warning_syn_count_diff)} of {len(tgt_node_ids)} target neurons!\n({"; ".join(filter(None, [cnt_str1, cnt_str2, cnt_str3]))})')
    
    return edges_table
