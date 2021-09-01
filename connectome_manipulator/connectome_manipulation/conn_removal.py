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


def apply(edges_table, nodes, _aux_dict, sel_src=None, sel_dest=None, amount_pct=100.0, min_syn_per_conn=None, max_syn_per_conn=None):
    """Remove percentage of randomly selected connections (i.e., all synapses per connection) according to certain cell and syn/conn selection criteria."""
    log.log_assert(0.0 <= amount_pct <= 100.0, 'amount_pct out of range!')

    gids_src = nodes[0].ids(sel_src)
    gids_dest = nodes[1].ids(sel_dest)

    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest)) # All potential synapses to be removed
    conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
    conn_sel = np.ones(conns.shape[0]).astype(bool)

    # Apply syn/conn filters (optional)
    if min_syn_per_conn is not None:
        log.log_assert(min_syn_per_conn >= 1, 'min_syn_per_conn out of range!')
        conn_sel = np.logical_and(conn_sel, num_syn_per_conn >= min_syn_per_conn)

    if max_syn_per_conn is not None:
        log.log_assert(max_syn_per_conn >= 1, 'max_syn_per_conn out of range!')
        conn_sel = np.logical_and(conn_sel, num_syn_per_conn <= max_syn_per_conn)

    conn_sel_idx = np.where(conn_sel)[0]
    num_conn = len(conn_sel_idx)
    if num_conn == 0:
        log.warning('Selection empty, nothing to remove!')
    num_remove = np.round(amount_pct * num_conn / 100).astype(int)
    conn_idx_remove = np.random.choice(conn_sel_idx, num_remove, replace=False)
    syn_idx_remove = np.isin(syn_conn_idx, conn_idx_remove)

    if min_syn_per_conn is not None and max_syn_per_conn is not None:
        syn_per_conn_info = f'with {min_syn_per_conn}-{max_syn_per_conn} syns/conn '
    elif min_syn_per_conn is None and max_syn_per_conn is not None:
        syn_per_conn_info = f'with max {max_syn_per_conn} syns/conn '
    elif min_syn_per_conn is not None and max_syn_per_conn is None:
        syn_per_conn_info = f'with min {min_syn_per_conn} syns/conn '
    else:
        syn_per_conn_info = ''
    log.info(f'Removing {num_remove} ({amount_pct}%) of {num_conn} connections {syn_per_conn_info}(sel_src={sel_src}, sel_dest={sel_dest}, {np.sum(syn_idx_remove)} synapses)')

    syn_sel_idx[syn_sel_idx] = syn_idx_remove # Set actual indices of connections to be removed
    edges_table_manip = edges_table[~syn_sel_idx].copy()

    return edges_table_manip
