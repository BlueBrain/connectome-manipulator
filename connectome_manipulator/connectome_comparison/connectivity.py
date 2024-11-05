# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for comparing connectomes based on connectivity features:

Structural comparison of two connectomes in terms of connection probability matrices for selected
pathways (including #synapses per connection), as specified by the config. For each connectome,
the underlying connectivity matrices are computed by the :func:`compute` function and will be saved
to a data file first. The individual connectivity matrices, together with a difference map
between the two connectomes, are then plotted by means of the :func:`plot` function.
"""

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from connectome_manipulator.access_functions import (
    get_edges_population,
    get_node_ids,
    get_connections,
)


def compute(
    circuit,
    group_by=None,
    sel_src=None,
    sel_dest=None,
    skip_empty_groups=False,
    edges_popul_name=None,
    **_,
):
    """Computes the average connection probabilities and #synapses/connection between groups of neurons of a given circuit's connectome.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        group_by (str): Neuron property name based on which to group connections, e.g., "synapse_class", "layer", or "mtype"; if omitted, the overall average is computed
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        skip_empty_groups (bool): If selected, only group property values that exist within the given source/target selection are kept; otherwise, all group property values, even if not present in the given source/target selection, will be included
        edges_popul_name (str): Name of SONATA egdes population to extract data from

    Returns:
        dict: Dictionary containing the computed data elements; see Notes

    Note:
        The returned dictionary contains the following data elements that can be selected for plotting through the structural comparison configuration file, together with a common dictionary containing additional information. Each data element is a dictionary with "data" (numpy.ndarray of size <source-group-size x target-group-size>), "name" (str), and "unit" (str) items.

        * "nsyn_conn": Mean number of synapses per connection
        * "nsyn_conn_std": Standard deviation of the number of synapses per connection
        * "nsyn_conn_sem": Standard error of the mean of the number of synapses per connection
        * "nsyn_conn_min": Minimum number of synapses per connection
        * "nsyn_conn_max": Maximum number of synapses per connection
        * "conn_prob": Average connection probability
    """
    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    if group_by is None:
        src_group_sel = [sel_src]
        tgt_group_sel = [sel_dest]
    else:
        if (
            skip_empty_groups
        ):  # Take only group property values that exist within given src/tgt selection
            src_group_values = np.unique(
                src_nodes.get(get_node_ids(src_nodes, sel_src), properties=group_by)
            )
            tgt_group_values = np.unique(
                tgt_nodes.get(get_node_ids(tgt_nodes, sel_dest), properties=group_by)
            )
        else:  # Keep all group property values, even if not present in given src/tgt selection, to get the full matrix
            src_group_values = sorted(src_nodes.property_values(group_by))
            tgt_group_values = sorted(tgt_nodes.property_values(group_by))

        if sel_src is None:
            sel_src = {}
        else:
            assert isinstance(
                sel_src, dict
            ), "ERROR: Source node selection must be a dict or empty!"  # Otherwise, it cannot be merged with group selection
        if sel_dest is None:
            sel_dest = {}
        else:
            assert isinstance(
                sel_dest, dict
            ), "ERROR: Target node selection must be a dict or empty!"  # Otherwise, it cannot be merged with group selection

        src_group_sel = [
            {**sel_src, group_by: src_group_values[idx]} for idx in range(len(src_group_values))
        ]  # group_by will overwrite selection in case group property also exists in selection!
        tgt_group_sel = [
            {**sel_dest, group_by: tgt_group_values[idx]} for idx in range(len(tgt_group_values))
        ]  # group_by will overwrite selection in case group property also exists in selection!

    print(
        f"INFO: Computing connectivity (group_by={group_by}, sel_src={sel_src}, sel_dest={sel_dest}, N={len(src_group_values)}x{len(tgt_group_values)} groups)",
        flush=True,
    )

    syn_table = np.zeros((len(src_group_sel), len(tgt_group_sel)))  # Mean
    syn_table_std = np.zeros((len(src_group_sel), len(tgt_group_sel)))  # Std
    syn_table_sem = np.zeros((len(src_group_sel), len(tgt_group_sel)))  # SEM
    syn_table_min = np.zeros((len(src_group_sel), len(tgt_group_sel)))  # Min
    syn_table_max = np.zeros((len(src_group_sel), len(tgt_group_sel)))  # Max
    p_table = np.zeros((len(src_group_sel), len(tgt_group_sel)))
    pbar = progressbar.ProgressBar()
    for idx_pre in pbar(range(len(src_group_sel))):
        sel_pre = src_group_sel[idx_pre]
        for idx_post, _ in enumerate(tgt_group_sel):
            sel_post = tgt_group_sel[idx_post]
            pre_ids = get_node_ids(src_nodes, sel_pre)
            post_ids = get_node_ids(tgt_nodes, sel_post)
            conns = get_connections(edges, pre_ids, post_ids, with_nsyn=True)

            if conns.size > 0:
                scounts = conns[:, 2]  # Synapse counts per connection
                ccount = len(scounts)  # Connection count
                pre_count = len(pre_ids)
                post_count = len(post_ids)

                syn_table[idx_pre, idx_post] = np.mean(scounts)
                syn_table_std[idx_pre, idx_post] = np.std(scounts)
                syn_table_sem[idx_pre, idx_post] = np.std(scounts) / np.sqrt(ccount)
                syn_table_min[idx_pre, idx_post] = np.min(scounts)
                syn_table_max[idx_pre, idx_post] = np.max(scounts)
                p_table[idx_pre, idx_post] = 100.0 * ccount / (pre_count * post_count)

    syn_table_name = "Synapses per connection"
    syn_table_unit = "#syn/conn"
    p_table_name = "Connection probability"
    p_table_unit = "Conn. prob. (%)"

    return {
        "nsyn_conn": {"data": syn_table, "name": syn_table_name, "unit": "Mean " + syn_table_unit},
        "nsyn_conn_std": {
            "data": syn_table_std,
            "name": syn_table_name,
            "unit": "Std of " + syn_table_unit,
        },
        "nsyn_conn_sem": {
            "data": syn_table_sem,
            "name": syn_table_name,
            "unit": "SEM of " + syn_table_unit,
        },
        "nsyn_conn_min": {
            "data": syn_table_min,
            "name": syn_table_name,
            "unit": "Min " + syn_table_unit,
        },
        "nsyn_conn_max": {
            "data": syn_table_max,
            "name": syn_table_name,
            "unit": "Max " + syn_table_unit,
        },
        "conn_prob": {"data": p_table, "name": p_table_name, "unit": p_table_unit},
        "common": {"src_group_values": src_group_values, "tgt_group_values": tgt_group_values},
    }


def plot(
    res_dict, common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, group_by=None, **_
):  # pragma:no cover
    """Plots a connectivity matrix or a difference matrix.

    Args:
        res_dict (dict): Results dictionary, containing selected data for plotting; must contain a "data" item with a connectivity matrix of type numpy.ndarray of size  <#source-group-values x #target-group-values>, as well as "name" and "unit" items containing strings.
        common_dict (dict): Common dictionary, containing additional information; must contain "src_group_values" and "tgt_group_values" items containing lists of source/target values of the grouped property, matching the size of the connectivity matrix in ``res_dict``
        fig_title (str): Optional figure title
        vmin (float): Minimum plot range
        vmax (float): Maximum plot range
        isdiff (bool): Flag indicating that ``res_dict`` contains a difference matrix; in this case, a symmetric plot range is required and a divergent colormap will be used
        group_by (str): Neuron property name based on which to group connections, e.g., "synapse_class", "layer", or "mtype"; if omitted, the overall average is computed
    """
    if isdiff:  # Difference plot
        assert -1 * vmin == vmax, "ERROR: Symmetric plot range required!"
        cmap = "PiYG"  # Symmetric (diverging) colormap
    else:  # Regular plot
        cmap = "hot_r"  # Regular colormap

    plt.imshow(res_dict["data"], interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    if fig_title is None:
        plt.title(res_dict["name"])
    else:
        plt.title(fig_title)

    if group_by:
        plt.xlabel(f"Postsynaptic {group_by}")
        plt.ylabel(f"Presynaptic {group_by}")

    n_grp = np.maximum(len(common_dict["src_group_values"]), len(common_dict["tgt_group_values"]))
    font_size = max(13 - n_grp / 6, 1)  # Font scaling
    if len(common_dict["src_group_values"]) > 0:
        plt.yticks(
            range(len(common_dict["src_group_values"])),
            common_dict["src_group_values"],
            rotation=0,
            fontsize=font_size,
        )

    if len(common_dict["tgt_group_values"]) > 0:
        if max(len(str(grp)) for grp in common_dict["tgt_group_values"]) > 1:
            rot_x = 90
        else:
            rot_x = 0
        plt.xticks(
            range(len(common_dict["tgt_group_values"])),
            common_dict["tgt_group_values"],
            rotation=rot_x,
            fontsize=font_size,
        )

    cb = plt.colorbar()
    cb.set_label(res_dict["unit"])
