# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging
import numpy as np

""" Remove percentage of randomly selected connections (i.e., all synapses per connection) according to certain cell and syn/conn selection criteria """
def apply(edges_table, nodes, aux_dict, sel_src, sel_dest, amount_pct=100.0, min_syn_per_conn=None, max_syn_per_conn=None):
    
    logging.log_assert(amount_pct >= 0.0 and amount_pct <= 100.0, 'amount_pct out of range!')
    
    gids_src = nodes.ids(sel_src)
    gids_dest = nodes.ids(sel_dest)
    
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest)) # All potential synapses to be removed
    conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
    conn_sel = np.ones(conns.shape[0]).astype(bool)

    # Apply syn/conn filters (optional)
    if not min_syn_per_conn is None:
        logging.log_assert(min_syn_per_conn >= 1, 'min_syn_per_conn out of range!')
        conn_sel = np.logical_and(conn_sel, num_syn_per_conn >= min_syn_per_conn)

    if not max_syn_per_conn is None:
        logging.log_assert(max_syn_per_conn >= 1, 'max_syn_per_conn out of range!')
        conn_sel = np.logical_and(conn_sel, num_syn_per_conn <= max_syn_per_conn)
    
    conn_sel_idx = np.where(conn_sel)[0]
    num_conn = len(conn_sel_idx)
    if num_conn == 0:
        logging.warning('Selection empty, nothing to remove!')
    num_remove = np.round(amount_pct * num_conn / 100).astype(int)
    conn_idx_remove = np.random.choice(conn_sel_idx, num_remove, replace=False)
    syn_idx_remove = np.isin(syn_conn_idx, conn_idx_remove)
    
    if not min_syn_per_conn is None and not max_syn_per_conn is None:
        syn_per_conn_info = f'with {min_syn_per_conn}-{max_syn_per_conn} syns/conn '
    elif min_syn_per_conn is None and not max_syn_per_conn is None:
        syn_per_conn_info = f'with max {max_syn_per_conn} syns/conn '
    elif not min_syn_per_conn is None and max_syn_per_conn is None:
        syn_per_conn_info = f'with min {min_syn_per_conn} syns/conn '
    else:
        syn_per_conn_info = ''
    logging.info(f'Removing {num_remove} ({amount_pct}%) of {num_conn} connections {syn_per_conn_info}from {sel_src} to {sel_dest} neurons ({np.sum(syn_idx_remove)} synapses)')
    
    syn_sel_idx[syn_sel_idx == True] = syn_idx_remove # Set actual indices of connections to be removed
    edges_table_manip = edges_table[~syn_sel_idx].copy()
    
    return edges_table_manip
