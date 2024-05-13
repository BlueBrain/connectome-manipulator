# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Manipulation name: syn_prop_alteration

Description: Modification of synaptic property values of a selected set of synapses, as given by
- absolute value
- relative scaling
- shuffling across synapses
- random absolute value drawn from given distribution
- random relative scaling drawn from given distribution
- random additive value drawn from given distribution
"""

import numpy as np

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import get_node_ids
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class SynapsePropertAlteration(Manipulation):
    """Manipulation name: syn_prop_alteration

    Description: Modification of synaptic property values of a selected set of synapses.
    """

    @profiler.profileit(name="syn_prop_alteration")
    def apply(
        self,
        split_ids,
        syn_prop,
        new_value,
        sel_src=None,
        sel_dest=None,
        syn_filter=None,
        amount_pct=100.0,
        **kwargs,
    ):
        """Modification of synaptic property values of a selected set of synapses."""
        # pylint: disable=arguments-differ
        if syn_filter is None:
            syn_filter = {}

        # Input checks
        edges_table = self.writer.to_pandas()
        available_properties = np.setdiff1d(
            edges_table.columns, ["@source_node", "@target_node"]
        ).tolist()  # Source/target nodes excluded
        available_modes = [
            "setval",
            "scale",
            "offset",
            "shuffle",
            "randval",
            "randscale",
            "randadd",
        ]  # Supported modes for generating new values
        log.log_assert(
            syn_prop in available_properties,
            f'syn_prop "{syn_prop}" not available or allowed for alteration! Must be one of: {available_properties}',
        )
        log.log_assert(
            np.all(np.isin(list(syn_filter.keys()), available_properties)),
            "One or more filter properties not available!",
        )
        log.log_assert(
            "mode" in new_value.keys(),
            f'Value "mode" must be specified! Available modes: {available_modes}',
        )
        log.log_assert(
            new_value["mode"] in available_modes,
            f'Value type "{new_value["mode"]}" unknown! Must be one of: {available_modes}',
        )
        log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")

        # Select pathway synapses
        gids_src = get_node_ids(self.nodes[0], sel_src)
        gids_dest = get_node_ids(self.nodes[1], sel_dest)
        syn_sel_idx = np.logical_and(
            np.isin(edges_table["@source_node"], gids_src),
            np.isin(edges_table["@target_node"], gids_dest),
        )  # All potential synapses to be removed

        # Filter based on synapse properties (optional)
        if len(syn_filter) > 0:
            log.info(f"Applying synapse filter(s) on: {list(syn_filter.keys())}")
            for prop, val in syn_filter.items():
                syn_sel_idx = np.logical_and(syn_sel_idx, np.isin(edges_table[prop], val))

        # Apply alterations
        num_syn = np.sum(syn_sel_idx)
        num_alter = np.round(amount_pct * num_syn / 100).astype(int)

        log.info(
            f'Altering "{syn_prop}" in {num_alter} ({amount_pct}%) of {num_syn} selected synapses based on "{new_value["mode"]}" mode (sel_src={sel_src}, sel_dest={sel_dest})'
        )

        if num_alter < num_syn:
            sel_alter = np.random.permutation([True] * num_alter + [False] * (num_syn - num_alter))
            syn_sel_idx[syn_sel_idx] = sel_alter  # Set actual indices of synapses to be altered

        val_range = new_value.get("range", [-np.inf, np.inf])

        # Property data type to cast new values to, so that data type is not changed!!
        prop_dtype = edges_table.dtypes[syn_prop].type

        if new_value["mode"] == "setval":
            # Set to a fixed given value
            log.log_assert(
                new_value["value"] >= val_range[0] and new_value["value"] <= val_range[1],
                "Property value out of range!",
            )
            edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(new_value["value"])
        elif new_value["mode"] == "scale":
            # Scale by a given factor
            edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(
                np.clip(
                    edges_table.loc[syn_sel_idx, syn_prop] * new_value["factor"],
                    val_range[0],
                    val_range[1],
                )
            )
        elif new_value["mode"] == "offset":
            # Offset by a given value
            edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(
                np.clip(
                    edges_table.loc[syn_sel_idx, syn_prop] + new_value["value"],
                    val_range[0],
                    val_range[1],
                )
            )
        elif new_value["mode"] == "shuffle":
            # Shuffle across synapses
            log.log_assert(
                self.split_total == 1,
                f'"{new_value["mode"]}" mode not supported in block-based processing! Reduce number of splits to 1!',
            )
            edges_table.loc[syn_sel_idx, syn_prop] = edges_table.loc[syn_sel_idx, syn_prop].values[
                np.random.permutation(np.sum(syn_sel_idx))
            ]
        elif new_value["mode"] in ["randval", "randscale", "randadd"]:
            transform = {
                "randval": lambda _, y: y,
                "randscale": np.multiply,
                "randadd": np.add,
            }[new_value["mode"]]

            rng = getattr(np.random, new_value["rng"])
            random_values = rng(**new_value["kwargs"], size=np.sum(syn_sel_idx))
            new_values = np.clip(
                transform(edges_table.loc[syn_sel_idx, syn_prop], random_values),
                val_range[0],
                val_range[1],
            )

            # Set random values from given distribution
            rng = getattr(np.random, new_value["rng"])
            edges_table.loc[syn_sel_idx, syn_prop] = prop_dtype(new_values)
        else:
            log.log_assert(False, f'Value mode "{new_value["mode"]}" not implemented!')

        self.writer.from_pandas(edges_table)
