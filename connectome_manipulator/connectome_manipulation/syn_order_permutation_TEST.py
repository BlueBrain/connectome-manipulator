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


def apply(edges_table, _nodes, _aux_dict):  # pragma: no cover
    """Randomly permute order of synapses to be written to fiel in edges_table [FOR TESTING PURPOSES]."""
    log.info('Permuting synapse order [TESTING]')

    num_syn = edges_table.shape[0]
    perm_idx = np.random.permutation(num_syn)
    edges_table = edges_table.iloc[perm_idx]

    log.info(f'PERMUTATION INDEX: {perm_idx[:5]}..{perm_idx[-5:]}')

    return edges_table
