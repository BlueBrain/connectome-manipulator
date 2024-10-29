# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Synapse removal module."""

import numpy as np

from scipy.sparse import csc_matrix

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import get_node_ids
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class SynapseRemoval(Manipulation):
    """Connectome manipulation class for removing synapses:

    Removes a percentage of randomly selected synapses according to certain cell
    selection criteria, optionally keeping connections (i.e., at least 1 syn/conn)
    and rescaling g_syns to keep sum of g_syns per connection constant (unless there
    is no synapse per connection left).
    The manipulation can be applied through the :func:`apply` method.
    """

    @profiler.profileit(name="syn_removal")
    def apply(
        self,
        split_ids,
        sel_src=None,
        sel_dest=None,
        amount_pct=100.0,
        keep_conns=False,
        rescale_gsyn=False,
        **kwargs,
    ):
        """Applies a removal of synapses according to certain cell selection criteria.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
            sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
            amount_pct (float): Percentage of randomly sampled synapses to be removed
            keep_conns (bool): If selected, won't remove entire connections but will keep (at least) one synapse per connection
            rescale_gsyn (bool): If selected, rescales g_syn (synaptic "conductance" property) so that the sum of g_syns per connections before and after the manipulation is kept the same (unless the entire connection has been removed)
            **kwargs: Additional keyword arguments - Not used

        Note:
            Input/output edges (synapse) tables are accessed through the ``writer`` object:

            * Loading input edges: ``edges_table = self.writer.to_pandas()``
            * Writing output edges: ``self.writer.from_pandas(edges_table_manip)``
        """
        # pylint: disable=arguments-differ
        log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")

        gids_src = get_node_ids(self.nodes[0], sel_src)
        gids_dest = get_node_ids(self.nodes[1], sel_dest)

        edges_table = self.writer.to_pandas()
        syn_sel_idx = np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )  # All potential synapses to be removed

        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) BEFORE synapse removal
            gsyn_table = self._get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)

        if keep_conns:  # Keep (at least) one synapse per connection
            rnd_perm = np.random.permutation(np.sum(syn_sel_idx))
            _, syn_idx_to_keep = np.unique(
                edges_table[syn_sel_idx].iloc[rnd_perm][["@source_node", "@target_node"]],
                axis=0,
                return_index=True,
            )  # Randomize order, so that index of first occurrence is randomized

            syn_keep_idx = np.ones(np.sum(syn_sel_idx)).astype(bool)
            syn_keep_idx[syn_idx_to_keep] = False
            inv_perm = np.argsort(rnd_perm)
            syn_sel_idx[syn_sel_idx] = syn_keep_idx[inv_perm]  # Restore original order

        num_syn = np.sum(syn_sel_idx)
        num_remove = np.round(amount_pct * num_syn / 100).astype(int)

        log.info(
            f"Removing {num_remove} ({amount_pct}%) of {num_syn} synapses (sel_src={sel_src}, sel_dest={sel_dest}, keep_conns={keep_conns}, rescale_gsyn={rescale_gsyn})"
        )

        sel_remove = np.random.permutation([True] * num_remove + [False] * (num_syn - num_remove))
        syn_sel_idx[syn_sel_idx] = sel_remove  # Set actual indices of synapses to be removed
        edges_table_manip = edges_table[~syn_sel_idx].copy()

        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) AFTER synapse removal ...
            gsyn_table_manip = self._get_gsyn_sum_per_conn(edges_table_manip, gids_src, gids_dest)

            # ... and rescale g_syn so that the sum of g_syns per connections BEFORE and AFTER manipulation is kept the same (unless there is no synapse per connection left)
            self._rescale_gsyn_per_conn(
                edges_table_manip, gids_src, gids_dest, gsyn_table, gsyn_table_manip
            )

        self.writer.from_pandas(edges_table_manip)

    @staticmethod
    def _get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest):
        """Helper function to determine sum of g_syns per connection."""
        mask = np.logical_and(
            np.in1d(edges_table["@source_node"], gids_src),
            np.in1d(edges_table["@target_node"], gids_dest),
        )
        gsyn_tab = (
            edges_table.loc[mask].groupby(["@source_node", "@target_node"]).sum()["conductance"]
        )
        gsyn_mat = csc_matrix(
            (
                gsyn_tab.to_numpy(),
                (
                    gsyn_tab.index.get_level_values("@source_node").to_numpy(),
                    gsyn_tab.index.get_level_values("@target_node").to_numpy(),
                ),
            )
        )

        return gsyn_mat

    @staticmethod
    def _rescale_gsyn_per_conn(edges_table, gids_src, gids_dest, gsyn_mat, gsyn_mat_manip):
        """Helper function to rescale g_syn in case of changed number of synapses per connection, keeping sum of g_syns per connection constant."""
        mask = np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )
        edge_indices = edges_table.index[mask]

        for edge_id in edge_indices:
            sidx = edges_table.at[edge_id, "@source_node"].astype(int)
            didx = edges_table.at[edge_id, "@target_node"].astype(int)
            if (
                gsyn_mat[sidx, didx] > 0.0
                and gsyn_mat_manip[sidx, didx] > 0.0
                and gsyn_mat_manip[sidx, didx] != gsyn_mat[sidx, didx]
            ):
                scale = gsyn_mat[sidx, didx] / gsyn_mat_manip[sidx, didx]
                edges_table.at[
                    edge_id, "conductance"
                ] *= scale  # Re-scale conductance 'in-place' in edges_table
