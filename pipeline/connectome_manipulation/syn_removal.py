# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np

""" Remove percentage of randomly selected synapses according to certain cell selection criteria, optionally keeping connections (i.e., at least 1 syn/conn) """
def apply(edges_table, nodes, aux_dict, sel_src, sel_dest, amount_pct=100.0, keep_conns=False):
    
    logging.log_assert(amount_pct >= 0.0 and amount_pct <= 100.0, 'amount_pct out of range!')
    
    gids_src = nodes.ids(sel_src)
    gids_dest = nodes.ids(sel_dest)
    
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest)) # All potential synapses to be removed
    
    if keep_conns: # Keep (at least) one synapse per connection
        rnd_perm = np.random.permutation(np.sum(syn_sel_idx))
        _, syn_idx_to_keep = np.unique(edges_table[syn_sel_idx].iloc[rnd_perm][['@source_node', '@target_node']], axis=0, return_index=True) # Randomize order, so that index of first occurrence is randomized
        
        syn_keep_idx = np.ones(np.sum(syn_sel_idx)).astype(bool)
        syn_keep_idx[syn_idx_to_keep] = False
        inv_perm = np.argsort(rnd_perm)
        syn_sel_idx[syn_sel_idx == True] = syn_keep_idx[inv_perm] # Restore original order
    
    num_syn = np.sum(syn_sel_idx)
    num_remove = np.round(amount_pct * num_syn / 100).astype(int)
    
    logging.info(f'Removing {num_remove} ({amount_pct}%) of {num_syn} synapses from {sel_src} to {sel_dest} neurons (keep_conns={keep_conns})')
    
    sel_remove = np.random.permutation([True] * num_remove + [False] * (num_syn - num_remove))
    syn_sel_idx[syn_sel_idx == True] = sel_remove # Set actual indices of synapses to be removed
    edges_table_manip = edges_table[~syn_sel_idx]
    
    return edges_table_manip
