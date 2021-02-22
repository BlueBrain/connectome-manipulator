# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np
import resource

""" Random synapse subsampling, keeping a certain percentage of synapses """
def apply(edges_table, nodes, aux_dict, keep_pct=100.0):
    
    logging.log_assert(keep_pct >= 0.0 and keep_pct <= 100.0, 'keep_pct out of range!')
    
    num_syn = edges_table.shape[0]
    num_keep = np.round(keep_pct * num_syn / 100).astype(int)
    
    logging.info(f'Synapse subsampling, keeping {num_keep} ({keep_pct}%) of {num_syn} synapses')
    
    syn_sel_idx = np.random.permutation([True] * num_keep + [False] * (num_syn - num_keep))
    edges_table = edges_table[syn_sel_idx]
    
    return edges_table
