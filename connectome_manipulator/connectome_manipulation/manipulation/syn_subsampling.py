# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Manipulation name: syn_subsampling

Description: Random subsampling of synapses, keeping a certain percentage of synapses.
"""

import numpy as np

from connectome_manipulator import log, profiler
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class SynapseSubsampling(Manipulation):
    """Manipulation name: syn_subsampling

    Description: Random subsampling of synapses, keeping a certain percentage of synapses.
    """

    @profiler.profileit(name="syn_subsampling")
    def apply(self, split_ids, keep_pct=100.0, **kwargs):
        """Random subsampling of synapses, keeping a certain percentage of synapses."""
        # pylint: disable=arguments-differ
        log.log_assert(0.0 <= keep_pct <= 100.0, "keep_pct out of range!")

        edges_table = self.writer.to_pandas()
        num_syn = edges_table.shape[0]
        num_keep = np.round(keep_pct * num_syn / 100).astype(int)

        log.info(f"Synapse subsampling, keeping {num_keep} ({keep_pct}%) of {num_syn} synapses")

        syn_sel_idx = np.random.permutation([True] * num_keep + [False] * (num_syn - num_keep))
        edges_table_manip = edges_table[syn_sel_idx].copy()

        self.writer.from_pandas(edges_table_manip)
