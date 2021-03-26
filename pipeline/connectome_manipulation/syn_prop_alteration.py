# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np

""" Modify property values of a selected set of synapses, either by relative scaling or a new absolute value """
def apply(edges_table, nodes, aux_dict, sel_src, sel_dest, syn_prop, new_value, syn_filter={}, amount_pct=100.0):
    
    # Input checks
    available_properties = np.setdiff1d(edges_table.columns, ['@source_node', '@target_node']).tolist() # Source/target nodes excluded
    available_modes = ['setval', 'scale', 'shuffle', 'randval', 'randscale', 'randadd'] # Supported modes for generating new values
    logging.log_assert(syn_prop in edges_table.columns, f'syn_prop "{syn_prop}" not available! Must be one of: {edges_table.columns.to_list()}')
    logging.log_assert(np.all(np.isin(list(syn_filter.keys()), available_properties)), 'One or more filter properties not available!')
    logging.log_assert('mode' in new_value.keys(), 'Value "mode" must be specified!')
    logging.log_assert(new_value['mode'] in available_modes, f'Value type "{new_value["mode"]}" unknown! Must be one of: {available_modes}')
    logging.log_assert(amount_pct >= 0.0 and amount_pct <= 100.0, 'amount_pct out of range!')
    
    # Select pathway synapses
    gids_src = nodes.ids(sel_src)
    gids_dest = nodes.ids(sel_dest)
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest)) # All potential synapses to be removed

    # Filter based on synapse properties (optional)
    if len(syn_filter) > 0:
        logging.info(f'Applying synapse filter(s) on: {list(syn_filter.keys())}')
        for prop, val in syn_filter.items():
            syn_sel_idx = np.logical_and(syn_sel_idx, np.isin(edges_table[prop], val))
    
    # Apply alterations
    num_syn = np.sum(syn_sel_idx)
    num_alter = np.round(amount_pct * num_syn / 100).astype(int)
    
    logging.info(f'Altering "{syn_prop}" in {num_alter} ({amount_pct}%) of {num_syn} selected synapses from {sel_src} to {sel_dest} neurons based on "{new_value["mode"]}" mode')
    
    if num_alter < num_syn:
        sel_alter = np.random.permutation([True] * num_alter + [False] * (num_syn - num_alter))
        syn_sel_idx[syn_sel_idx == True] = sel_alter # Set actual indices of synapses to be altered
    
    prop_dtype = edges_table.dtypes[syn_prop].type # Property data type to cast new values to, so that data type is not changed!!
    if new_value['mode'] == 'setval': # Set to a fixed given value
        edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(new_value['value'])
    elif new_value['mode'] == 'scale': # Scale by a given factor
        edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(edges_table.loc[syn_sel_idx, syn_prop] * new_value['factor'])
    elif new_value['mode'] == 'shuffle': # Shuffle across synapses
        logging.log_assert(aux_dict['N_split'] == 1, f'"{new_value["mode"]}" mode not supported in block-based processing! Reduce number of splits to 1!')
        edges_table.loc[syn_sel_idx, syn_prop] = edges_table.loc[syn_sel_idx, syn_prop].values[np.random.permutation(np.sum(syn_sel_idx))]
    elif new_value['mode'] == 'randval': # Set random values from given distribution
        rng = eval('np.random.' + new_value['rng'])
        edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(rng(**new_value['kwargs'], size=np.sum(syn_sel_idx)))
    elif new_value['mode'] == 'randscale': # Scale by random factors from given distribution
        rng = eval('np.random.' + new_value['rng'])
        edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(edges_table.loc[syn_sel_idx, syn_prop] * rng(**new_value['kwargs'], size=np.sum(syn_sel_idx)))
    elif new_value['mode'] == 'randadd': # Add random values from given distribution
        rng = eval('np.random.' + new_value['rng'])
        edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(edges_table.loc[syn_sel_idx, syn_prop] + rng(**new_value['kwargs'], size=np.sum(syn_sel_idx)))
    else:
        logging.log_assert(False, f'Value mode "{new_value["mode"]}" not implemented!')
    
    return edges_table
