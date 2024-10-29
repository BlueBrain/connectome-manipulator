# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Synapse subsampling module."""

import numpy as np

from connectome_manipulator import log, profiler
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class SynapseSubsampling(Manipulation):
    """Connectome manipulation class for synapse subsampling:

    Subsamples synapses, keeping a certain percentage of synapses.
    The manipulation can be applied through the :func:`apply` method.
    For a more fine-grained control, please use the
    :func:`connectome_manipulator.connectome_manipulation.manipulation.syn_removal.SynapseRemoval`
    operation.
    """

    @profiler.profileit(name="syn_subsampling")
    def apply(self, split_ids, keep_pct=100.0, **kwargs):
        """Applies a random subsampling of synapses, keeping a certain percentage of synapses.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            keep_pct (float): Percentage of randomly sampled synapses to be kept
            **kwargs: Additional keyword arguments - Not used

        Note:
            Input/output edges (synapse) tables are accessed through the ``writer`` object:

            * Loading input edges: ``edges_table = self.writer.to_pandas()``
            * Writing output edges: ``self.writer.from_pandas(edges_table_manip)``

        """
        # pylint: disable=arguments-differ
        log.log_assert(0.0 <= keep_pct <= 100.0, "keep_pct out of range!")

        edges_table = self.writer.to_pandas()
        num_syn = edges_table.shape[0]
        num_keep = np.round(keep_pct * num_syn / 100).astype(int)

        log.info(f"Synapse subsampling, keeping {num_keep} ({keep_pct}%) of {num_syn} synapses")

        syn_sel_idx = np.random.permutation([True] * num_keep + [False] * (num_syn - num_keep))
        edges_table_manip = edges_table[syn_sel_idx].copy()

        self.writer.from_pandas(edges_table_manip)
