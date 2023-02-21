"""TODO: improve description"""
# Helper functions for connectome manipulations

import numpy as np


def get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest):
    """Helper function to determine sum of g_syns per connection."""
    src_offset = min(gids_src)  # Index offset so that indexing always starts at zero
    dest_offset = min(gids_dest)  # Index offset so that indexing always starts at zero

    gsyn_table = np.full((max(gids_src) - src_offset + 1, max(gids_dest) - dest_offset + 1), 0.0)
    edge_indices = edges_table.index[
        np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )
    ]
    for edge_id in edge_indices:
        sidx = edges_table.at[edge_id, "@source_node"].astype(int) - src_offset
        didx = edges_table.at[edge_id, "@target_node"].astype(int) - dest_offset
        gsyn_table[sidx, didx] += edges_table.at[edge_id, "conductance"]

    return gsyn_table


def rescale_gsyn_per_conn(edges_table, gids_src, gids_dest, gsyn_table, gsyn_table_manip):
    """Helper function to rescale g_syn in case of changed number of synapses per connection, keeping sum of g_syns per connection constant."""
    src_offset = min(gids_src)  # Index offset so that indexing always starts at zero
    dest_offset = min(gids_dest)  # Index offset so that indexing always starts at zero

    edge_indices = edges_table.index[
        np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )
    ]
    for edge_id in edge_indices:
        sidx = edges_table.at[edge_id, "@source_node"].astype(int) - src_offset
        didx = edges_table.at[edge_id, "@target_node"].astype(int) - dest_offset
        if (
            gsyn_table[sidx, didx] > 0.0
            and gsyn_table_manip[sidx, didx] > 0.0
            and gsyn_table_manip[sidx, didx] != gsyn_table[sidx, didx]
        ):
            scale = gsyn_table[sidx, didx] / gsyn_table_manip[sidx, didx]
            edges_table.at[
                edge_id, "conductance"
            ] *= scale  # Re-scale conductance 'in-place' in edges_table
