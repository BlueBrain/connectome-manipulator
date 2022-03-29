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
from connectome_manipulator.connectome_manipulation.helper_functions import (
    get_gsyn_sum_per_conn, rescale_gsyn_per_conn)
from connectome_manipulator.access_functions import get_node_ids


def apply(edges_table, nodes, _aux_dict, sel_src=None, sel_dest=None, amount_pct=100.0, keep_conns=False, rescale_gsyn=False):
    """Remove percentage of randomly selected synapses according to certain cell selection criteria, optionally keeping connections (i.e., at least 1 syn/conn) and rescaling g_syns to keep sum of g_syns per connection constant (unless there is no synapse per connection left)."""
    log.log_assert(0.0 <= amount_pct <= 100.0, 'amount_pct out of range!')

    gids_src = get_node_ids(nodes[0], sel_src)
    gids_dest = get_node_ids(nodes[1], sel_dest)

    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest)) # All potential synapses to be removed

    if rescale_gsyn:
        # Determine connection strength (sum of g_syns per connection) BEFORE synapse removal
        gsyn_table = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)

    if keep_conns: # Keep (at least) one synapse per connection
        rnd_perm = np.random.permutation(np.sum(syn_sel_idx))
        _, syn_idx_to_keep = np.unique(edges_table[syn_sel_idx].iloc[rnd_perm][['@source_node', '@target_node']], axis=0, return_index=True) # Randomize order, so that index of first occurrence is randomized

        syn_keep_idx = np.ones(np.sum(syn_sel_idx)).astype(bool)
        syn_keep_idx[syn_idx_to_keep] = False
        inv_perm = np.argsort(rnd_perm)
        syn_sel_idx[syn_sel_idx] = syn_keep_idx[inv_perm] # Restore original order

    num_syn = np.sum(syn_sel_idx)
    num_remove = np.round(amount_pct * num_syn / 100).astype(int)

    log.info(f'Removing {num_remove} ({amount_pct}%) of {num_syn} synapses (sel_src={sel_src}, sel_dest={sel_dest}, keep_conns={keep_conns}, rescale_gsyn={rescale_gsyn})')

    sel_remove = np.random.permutation([True] * num_remove + [False] * (num_syn - num_remove))
    syn_sel_idx[syn_sel_idx] = sel_remove # Set actual indices of synapses to be removed
    edges_table_manip = edges_table[~syn_sel_idx].copy()

    if rescale_gsyn:
        # Determine connection strength (sum of g_syns per connection) AFTER synapse removal ...
        gsyn_table_manip = get_gsyn_sum_per_conn(edges_table_manip, gids_src, gids_dest)

        # ... and rescale g_syn so that the sum of g_syns per connections BEFORE and AFTER manipulation is kept the same (unless there is no synapse per connection left)
        rescale_gsyn_per_conn(edges_table_manip, gids_src, gids_dest, gsyn_table, gsyn_table_manip)

    return edges_table_manip
