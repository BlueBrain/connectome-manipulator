# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np
from helper_functions import get_gsyn_sum_per_conn, rescale_gsyn_per_conn

""" Add certain amount of synapses to existing connections, optionally keeping sum of g_syns per connection constant """
def apply(edges_table, nodes, aux_dict, sel_src, sel_dest, amount, rescale_gsyn=False, method='duplicate'):
    
    # Input checks
    logging.log_assert(isinstance(amount, dict) and 'type' in amount.keys() and 'value' in amount.keys(), 'Amount must be specified as dict with "type"/"value" entries!')
    logging.log_assert(amount['type'] in ['pct', 'pct_per_conn', 'minval_per_conn'], f'Synapse amount type "{amount["type"]}" not supported (must be "pct", "pct_per_conn", or "minval_per_conn")!')
    # amount: pct ... Overall increase by percentage of total number of existing synapses
    #         pct_per_conn ... Increase by percentage of existing synapses per connection
    #         minval_per_conn ... Increase until minimum target value of synapses per connection is reached
    logging.log_assert(method in ['duplicate'], f'Synapse addition method "{method}" not supported (must be "duplicate")!')
    # method: duplicate ... Duplicate existing synapses
    #         [NYI] derive ... Derive from existing synapses (duplicate and randomize certain properties)
    #         [NYI] randomize ... Fully randomized parameterization, based on model-based distributions
    #         [NYI] load ... Load from external connectome, e.g. structural connectome
    
    # Determine number of synapses to be added
    gids_src = nodes.ids(sel_src)
    gids_dest = nodes.ids(sel_dest)
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest))
    num_syn = np.sum(syn_sel_idx)
    
    if amount['type'] == 'pct': # Overall increase by percentage of total number of existing synapses between src and dest
        logging.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn / 100).astype(int)
    elif amount['type'] == 'pct_per_conn': # Increase by percentage of existing synapses per connection
        logging.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
        num_add = np.round(amount['value'] * num_syn_per_conn / 100).astype(int)
    elif amount['type'] == 'minval_per_conn': # Increase until minimum target value of synapses per connection is reached
        logging.log_assert(amount['value'] >= 0, f'Amount value for type "{amount["type"]}" out of range!')
        conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
        num_add = np.maximum(np.round(amount['value']).astype(int) - num_syn_per_conn, 0)
    else:
        logging.log_assert(False, f'Amount type "{amount["type"]}" not supported!')
    
    logging.info(f'Adding {np.sum(num_add)} synapses to {edges_table.shape[0]} total synapses from {sel_src} to {sel_dest} neurons (amount type={amount["type"]}, amount value={amount["value"]}, method={method}, rescale_gsyn={rescale_gsyn})')
        
    # Add num_add synapses between src and dest nodes
    if np.sum(num_add) > 0:
        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) BEFORE adding synapses
            gsyn_table = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)
        
        if method == 'duplicate': # Duplicate existing synapses
            if np.isscalar(num_add): # Overall number of synapses to add
                sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_add) # Random sampling from existing synapses with replacement
            else: # Number of synapses per connection to add
                sel_dupl = np.full(np.sum(num_add), -1) # Empty selection
                dupl_idx = np.hstack((0, np.cumsum(num_add))) # Index vector where to save selected indices
                for cidx, num in enumerate(num_add):
                    if num == 0: # Nothing to add for this connection
                        continue
                    conn_sel = np.zeros_like(syn_sel_idx, dtype=bool) # Empty selection mask
                    conn_sel[syn_sel_idx==True] = syn_conn_idx == cidx # Sub-select (mask) synapses belonging to given connection from all selected synapses
                    sel_dupl[dupl_idx[cidx]:dupl_idx[cidx+1]] = np.random.choice(np.where(conn_sel)[0], num) # Random sampling from existing synapses per connection with replacement
            new_edges = edges_table.iloc[sel_dupl]
        else:
            logging.log_assert(False, f'Method "{method}" not supported!')
        
        # Add new synapses to table, re-sort, and assign new index
        edges_table = edges_table.append(new_edges)
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)    
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required when merging files in block-based processing]
        
        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) AFTER adding synapses ...
            gsyn_table_manip = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)
            
            # ... and rescale g_syn so that the sum of g_syns per connections BEFORE and AFTER the manipulation is kept the same
            rescale_gsyn_per_conn(edges_table, gids_src, gids_dest, gsyn_table, gsyn_table_manip)        
    else:
        logging.warning(f'Nothing to add!')
    
    return edges_table
