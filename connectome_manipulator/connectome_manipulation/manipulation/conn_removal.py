# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Manipulation name: conn_removal

Description: Remove percentage of randomly selected connections (i.e., all synapses per connection)
according to certain cell and syn/conn selection criteria.

Optionally, a connection mask can be provided, in which case only connections within
that mask will be considered for removal (in addition to the other selecion criteria).
Such connection mask (.npz file) must be given as sparse adjacency matrix in CSC format,
exactly matching the size of the selected src/dest neurons and indexed in increasing order.
"""

import os

import numpy as np
import scipy.sparse as sps

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import get_node_ids
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class ConnectomeRemoval(Manipulation):
    """Manipulation name: conn_removal

    Description: Remove percentage of randomly selected connections (i.e., all synapses per connection)
    according to certain cell and syn/conn selection criteria.

    Optionally, a connection mask can be provided, in which case only connections within
    that mask will be considered for removal (in addition to the other selecion criteria).
    Such connection mask (.npz file) must be given as sparse adjacency matrix in CSC format,
    exactly matching the size of the selected src/dest neurons and indexed in increasing order.
    """

    @profiler.profileit(name="conn_removal")
    def apply(
        self,
        split_ids,
        sel_src=None,
        sel_dest=None,
        amount_pct=100.0,
        min_syn_per_conn=None,
        max_syn_per_conn=None,
        conn_mask_file=None,
        **kwargs,
    ):
        """Remove percentage of randomly selected connections (i.e., all synapses per connection) according to certain cell and syn/conn selection criteria."""
        # pylint: disable=arguments-differ
        log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")

        gids_src = get_node_ids(self.nodes[0], sel_src)
        gids_dest = get_node_ids(self.nodes[1], sel_dest)

        edges_table = self.writer.to_pandas()
        syn_sel_idx = np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )  # All potential synapses to be removed
        conns, syn_conn_idx, num_syn_per_conn = np.unique(
            edges_table[syn_sel_idx][["@source_node", "@target_node"]],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        conn_sel = np.ones(conns.shape[0]).astype(bool)

        # Connection mask (optional)
        if conn_mask_file is not None:
            log.log_assert(
                os.path.splitext(conn_mask_file)[-1].lower() == ".npz",
                f'Connection mask file "{conn_mask_file}" not in .npz format!',
            )
            log.log_assert(
                os.path.exists(conn_mask_file),
                f'Connection mask file "{conn_mask_file}" not found!',
            )
            conn_mask = sps.load_npz(conn_mask_file)
            log.log_assert(
                conn_mask.shape[0] == len(gids_src) and conn_mask.shape[1] == len(gids_dest),
                f"Size of connection mask does not match selected number of pre/post neurons (must be <{len(gids_src)}x{len(gids_dest)}>)!",
            )
            log.info(
                f"Loaded <{conn_mask.shape[0]}x{conn_mask.shape[1]}> connection mask with {conn_mask.count_nonzero()} entries"
            )

            # Create index table for converting neuron IDs to matrix indices
            src_conv = sps.csr_matrix(
                (
                    np.arange(len(gids_src), dtype=int),
                    (np.zeros(len(gids_src), dtype=int), gids_src),
                )
            )
            dest_conv = sps.csr_matrix(
                (
                    np.arange(len(gids_dest), dtype=int),
                    (np.zeros(len(gids_dest), dtype=int), gids_dest),
                )
            )
            conns_reindex = np.array(
                [
                    src_conv[0, conns[:, 0]].toarray().flatten(),
                    dest_conv[0, conns[:, 1]].toarray().flatten(),
                ]
            ).T

            # Apply mask
            conn_sel = np.logical_and(
                conn_sel, np.array(conn_mask[conns_reindex[:, 0], conns_reindex[:, 1]]).flatten()
            )

        # Apply syn/conn filters (optional)
        if min_syn_per_conn is not None:
            log.log_assert(min_syn_per_conn >= 1, "min_syn_per_conn out of range!")
            conn_sel = np.logical_and(conn_sel, num_syn_per_conn >= min_syn_per_conn)

        if max_syn_per_conn is not None:
            log.log_assert(max_syn_per_conn >= 1, "max_syn_per_conn out of range!")
            conn_sel = np.logical_and(conn_sel, num_syn_per_conn <= max_syn_per_conn)

        conn_sel_idx = np.where(conn_sel)[0]
        num_conn = len(conn_sel_idx)
        if num_conn == 0:
            log.debug("Selection empty, nothing to remove!")
        num_remove = np.round(amount_pct * num_conn / 100).astype(int)
        conn_idx_remove = np.random.choice(conn_sel_idx, num_remove, replace=False)
        syn_idx_remove = np.isin(syn_conn_idx, conn_idx_remove)

        if min_syn_per_conn is not None and max_syn_per_conn is not None:
            if min_syn_per_conn == max_syn_per_conn:
                syn_per_conn_info = f"with {min_syn_per_conn} syns/conn "
            else:
                syn_per_conn_info = f"with {min_syn_per_conn}-{max_syn_per_conn} syns/conn "
        elif min_syn_per_conn is None and max_syn_per_conn is not None:
            syn_per_conn_info = f"with max {max_syn_per_conn} syns/conn "
        elif min_syn_per_conn is not None and max_syn_per_conn is None:
            syn_per_conn_info = f"with min {min_syn_per_conn} syns/conn "
        else:
            syn_per_conn_info = ""
        log.info(
            f"Removing {num_remove} ({amount_pct}%) of {num_conn} connections {syn_per_conn_info}(sel_src={sel_src}, sel_dest={sel_dest}, {np.sum(syn_idx_remove)} synapses)"
        )

        syn_sel_idx[syn_sel_idx] = syn_idx_remove  # Set actual indices of connections to be removed
        edges_table_manip = edges_table[~syn_sel_idx].copy()

        self.writer.from_pandas(edges_table_manip)
