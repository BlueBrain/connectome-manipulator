# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for alteration of synapse property values."""

import numpy as np

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import get_node_ids
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class SynapsePropertAlteration(Manipulation):
    """Connectome manipulation class for altering synapse property values:

    Modifies the values of a selected synapse property and a selected set of synapses,
    as given by constant, multiplicative, additive, or random values drawn from chosen
    distributions. The manipulation can be applied through the :func:`apply` method.
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
        """Applies a modification of synapse property values on a selected set of synapses.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            syn_prop (str): Selected synapse property values of which should be altered; can be selected from all available properties in the given SONATA input edges table (except for "@source_node" and "@target_node")
            new_value (dict): Dictionary specifying the type of alteration; must contain a "mode" key with one of the available alteration modes, together with additional settings depending on the mode, see Notes; optionally, can also contain a "range" key with lower/upper value bounds, an "rng" key to specify a random number generator, and a "kwargs" dict with additional arguments for the random number generator
            sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
            sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
            syn_filter (dict): Optional filter dictionary with property names (keys) with a corresponding list of values (e.g., ``{"layer": [2, 3, 4], "synapse_class": ["EXC"]}``); only synapses corresponding to that selection of values will be considered for alteration
            amount_pct (float): Percentage of randomly sampled synapses to be altered
            **kwargs: Additional keyword arguments - Not used

        Note:
            Available alteration modes and corresponding settings:

            * "setval": Absolute value, as given by "value"
            * "scale": Multiplicative scaling, as given by "factor"
            * "offset": Additive offset, as given by "value"
            * "shuffle": Shuffling of values among synapses; only supported when using a single data split
            * "randval": Random absolute value drawn from a given distribution (*)
            * "randscale": Multiplicative scaling by a random value drawn from a given distribution (*)
            * "randadd": Additive offset by a random value drawn from a given distribution (*)

            (*) A random distribution can be specified in ``new_value`` by choosing a random number generator as "rng" from ``numpy.random`` (e.g., "normal"), together with "kwargs" which will be passed when drawing values from that generator (e.g., "loc" and "scale" in case of "normal")

        Note:
            Input/output edges (synapse) tables are accessed through the ``writer`` object:

            * Loading input edges: ``edges_table = self.writer.to_pandas()``
            * Writing output edges: ``self.writer.from_pandas(edges_table_manip)``
        """
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
