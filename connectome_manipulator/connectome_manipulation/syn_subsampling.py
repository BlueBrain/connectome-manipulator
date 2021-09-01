'''TODO: improve description'''
# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import numpy as np

from connectome_manipulator import log


def apply(edges_table, _nodes, _aux_dict, keep_pct=100.0):
    """Random synapse subsampling, keeping a certain percentage of synapses."""
    log.log_assert(0.0 <= keep_pct <= 100.0, 'keep_pct out of range!')

    num_syn = edges_table.shape[0]
    num_keep = np.round(keep_pct * num_syn / 100).astype(int)

    log.info(f'Synapse subsampling, keeping {num_keep} ({keep_pct}%) of {num_syn} synapses')

    syn_sel_idx = np.random.permutation([True] * num_keep + [False] * (num_syn - num_keep))
    edges_table_manip = edges_table[syn_sel_idx].copy()

    return edges_table_manip
