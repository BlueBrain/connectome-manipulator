# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np

""" Add certain amount of synapses to existing connections, optionally keeping sum of g_syns per connection constant """
def apply(edges_table, nodes, aux_dict, sel_src, sel_dest, syn_class, amount, rescale_gsyn=False, method='duplicate'):
    
    # Input checks
    logging.log_assert(syn_class in ['EXC', 'INH'], f'Synapse class "{syn_class}" not supported (must be "EXC" or "INH")!')
    logging.log_assert(isinstance(amount, dict) and 'type' in amount.keys() and 'value' in amount.keys(), 'Amount must be specified as dict with "type"/"value" entries!')
    logging.log_assert(amount['type'] in ['pct', 'pct_per_conn', 'minval_per_conn'], f'Synapse amount type "{amount["type"]}" not supported (must be "pct", "pct_per_conn", or "minval_per_conn")!')
    # amount: pct ... Overall increase by percentage of total number of existing synapses
    #         [NYI] pct_per_conn ... Increase by percentage of existing synapses per connection
    #         [NYI] minval_per_conn ... Increase until minimum target value of synapses per connection is reached
    logging.log_assert(method in ['duplicate'], f'Synapse addition method "{method}" not supported (must be "duplicate")!')
    # method: duplicate ... Duplicate existing synapses
    #         [NYI] derive ... Derive from existing synapses (duplicate and randomize certain properties)
    #         [NYI] randomize ... Fully randomized parameterization, based on model-based distributions
    #         [NYI] load ... Load from external connectome, e.g. structural connectome
    
    # Select number of synapses to be added
    gids_src = nodes.ids(sel_src)
    gids_dest = nodes.ids(sel_dest)
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest))
    num_syn = np.sum(syn_sel_idx)
    
    if amount['type'] == 'pct': # Overall increase by percentage of total number of existing synapses between src and dest
        logging.log_assert(amount['value'] >= 0.0 and amount['value'] <= 100.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn / 100).astype(int)
    else:
        logging.log_assert(False, f'Amount type "{amount["type"]}" not supported!')
    
    logging.info(f'Adding {num_add} {syn_class} synapses to {edges_table.shape[0]} total synapses from {sel_src} to {sel_dest} neurons (amount type={amount["type"]}, amount value={amount["value"]}, method={method}, rescale_gsyn={rescale_gsyn})')
    
    if num_add > 0:
        # Create num_add EXC/INH synapses between src and dest nodes to be added
        if method == 'duplicate':
            # Duplicate only synapses of given class (EXC/INH)
            if syn_class == 'EXC':
                syn_sel_idx = np.logical_and(syn_sel_idx, edges_table['syn_type_id'] >= 100)
            elif syn_class == 'INH':
                syn_sel_idx = np.logical_and(syn_sel_idx, edges_table['syn_type_id'] < 100)
            else:
                logging.log_assert(False, f'Synapse class {syn_class} not supported!')
            
            sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_add) # Random sampling from existing EXC/INH synapses with replacement
            new_edges = edges_table.iloc[sel_dupl]
        else:
            logging.log_assert(False, f'Method "{method}" not supported!')
        
        # Add new synapses to table, re-sort, and assign new index
        edges_table = edges_table.append(new_edges)
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)    
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required in block-based processing]
    else:
        logging.warning(f'Nothing to add!')
    
    return edges_table
