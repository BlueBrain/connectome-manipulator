# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Connectome extraction module."""

import os

import json
import numpy as np

from connectome_manipulator import log, profiler
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class ConnectomeExtraction(Manipulation):
    """Connectome manipulation class for extracting connections:

    Extracts a subset of connections within a cell target, i.e., keeps
    only connections within that target and removed all other connections
    from the connectome. The manipulation can be applied through the :func:`apply`
    method. The nodes (neuron) table is always kept unchanged!
    """

    @profiler.profileit(name="conn_extraction")
    def apply(self, split_ids, target_name=None, node_sets_file=None, **kwargs):
        """Applies an extraction of connections within a given cell target.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            target_name (str): Cell target name (as defined in SONATA node sets file) to extract connectome from, i.e, keeping only connections within that target; an empty connectome will be returned if no `target_name` is provided
            node_sets_file (str): Optional file path to alternative SONATA node sets file (.json) containing `node_id` entries; can be provided in case the cell target is not part of the circuit's intrinsic node sets.
            **kwargs: Additional keyword arguments - Not used

        Note:
            Input/output edges (synapse) tables are accessed through the ``writer`` object:

            * Loading input edges: ``edges_table = self.writer.to_pandas()``
            * Writing output edges: ``self.writer.from_pandas(edges_table_manip)``
        """
        # pylint: disable=arguments-differ
        edges_table = self.writer.to_pandas()
        if target_name is None:
            log.info("No target name provided, returning empty connectome!")
            self.writer.from_pandas(edges_table.loc[[]].copy())
            return

        # Load cell targets
        assert (
            self.nodes[0] is self.nodes[1]
        ), "ERROR: Only one source/target node population supported!"
        # pylint: disable=W0212
        if target_name in self.nodes[0]._node_sets.content:
            target_gids = self.nodes[0].ids(target_name)
        else:
            log.log_assert(
                node_sets_file is not None,
                f'Target "{target_name}" unknown, node sets file required!',
            )

            # Load node sets file
            log.log_assert(
                os.path.exists(node_sets_file), f'Node sets file "{node_sets_file}" not found!'
            )
            with open(node_sets_file, "r") as f:
                node_sets = json.load(f)

            log.log_assert(
                target_name in node_sets.keys(),
                f'Target "{target_name}" not found in "{node_sets_file}"!',
            )
            log.log_assert(
                "node_id" in node_sets[target_name],
                f'Node IDs for target "{target_name}" not found in "{node_sets_file}"!',
            )
            target_gids = node_sets[target_name]["node_id"]
        log.info(f'Found {len(target_gids)} GIDs for target "{target_name}" to extract')

        # Extract connectome
        syn_sel_idx = np.logical_and(
            np.isin(edges_table["@source_node"], target_gids),
            np.isin(edges_table["@target_node"], target_gids),
        )
        edges_table_manip = edges_table[syn_sel_idx].copy()

        # TESTING/DEBUGGING #
        log.log_assert(
            not np.any(edges_table_manip.sum(1) == 0),
            f"Empty edges table entries found!\ntarget_gids={target_gids}\np.sum(syn_sel_idx)={np.sum(syn_sel_idx)}\nedges_table_manip.size={edges_table_manip.size}",
        )
        # ################# #

        self.writer.from_pandas(edges_table_manip)
