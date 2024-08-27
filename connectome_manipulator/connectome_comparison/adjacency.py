# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for comparing connectomes based on adjacency matrices:

Structural comparison of two connectomes in terms of adjacency matrices for selected pathways
(including synapse counts per connection), as specified by the config. For each connectome,
the underlying adjacency/count matrices are computed by the :func:`compute` function and will be saved
to a data file first. The individual adjacency/count matrices, together with a difference map
between the two connectomes, are then plotted by means of the :func:`plot` function.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix
from connectome_manipulator.access_functions import (
    get_edges_population,
    get_node_ids,
    get_connections,
)


def compute(circuit, sel_src=None, sel_dest=None, edges_popul_name=None, **_):
    """Extracts adjacency and count matrices from a given circuit's connectome.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        edges_popul_name (str): Name of SONATA egdes population to extract data from

    Returns:
        dict: Dictionary containing the extracted data elements; see Notes

    Note:
        The returned dictionary contains the following data elements that can be selected for plotting through the structural comparison configuration file, together with a common dictionary containing additional information. Each data element is a dictionary with "data" (scipy.sparse.csc_matrix), "name" (str), and "unit" (str) items.

        * "adj": Adjacency matrix
        * "adj_cnt": Synaptome matrix, containing the numbers of synapses per connection
    """
    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    src_node_ids = get_node_ids(src_nodes, sel_src)
    tgt_node_ids = get_node_ids(tgt_nodes, sel_dest)

    assert (
        len(src_node_ids) > 0 and len(tgt_node_ids) > 0
    ), "ERROR: Empty src/tgt node selection(s)!"

    # Map source/target node ids to continuous range of indices for plotting
    src_gid_min = min(src_nodes.ids())
    src_gid_max = max(src_nodes.ids())
    tgt_gid_min = min(tgt_nodes.ids())
    tgt_gid_max = max(tgt_nodes.ids())

    src_plot_ids = np.full(src_gid_max - src_gid_min + 1, -1).astype(int)
    src_gid_offset = src_gid_min
    src_plot_ids[src_node_ids - src_gid_offset] = np.arange(len(src_node_ids))

    def src_gid_to_idx(gids):
        return src_plot_ids[gids - src_gid_offset]

    tgt_plot_ids = np.full(tgt_gid_max - tgt_gid_min + 1, -1).astype(int)
    tgt_gid_offset = tgt_gid_min
    tgt_plot_ids[tgt_node_ids - tgt_gid_offset] = np.arange(len(tgt_node_ids))

    def tgt_gid_to_idx(gids):
        return tgt_plot_ids[gids - tgt_gid_offset]

    print(
        f"INFO: Creating {len(src_node_ids)}x{len(tgt_node_ids)} adjacency matrix (sel_src={sel_src}, sel_dest={sel_dest})",
        flush=True,
    )

    conns = get_connections(edges, src_node_ids, tgt_node_ids, with_nsyn=True)
    if len(conns) == 0:  # No connections, creating empty matrix
        count_matrix = csc_matrix((len(src_node_ids), len(tgt_node_ids)), dtype=int)
    else:
        count_matrix = csc_matrix(
            (conns[:, 2], (src_gid_to_idx(conns[:, 0]), tgt_gid_to_idx(conns[:, 1]))),
            shape=(len(src_node_ids), len(tgt_node_ids)),
            dtype=int,
        )

    adj_matrix = count_matrix > 0

    return {
        "adj": {"data": adj_matrix, "name": "Adjacency", "unit": None},
        "adj_cnt": {"data": count_matrix, "name": "Adjacency count", "unit": "Synapse count"},
        "common": {"src_gids": src_node_ids, "tgt_gids": tgt_node_ids},
    }


def plot(
    res_dict, _common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, **_
):  # pragma:no cover
    """Plots an adjacency/count matrix or a difference matrix.

    Args:
        res_dict (dict): Results dictionary, containing selected data for plotting; must contain a "data" item with a sparse matrix of type scipy.sparse.csc_matrix, as well as "name" and "unit" items containing strings.
        _common_dict (dict): Common dictionary, containing additional information - Not used
        fig_title (str): Optional figure title
        vmin (float): Minimum plot range
        vmax (float): Maximum plot range
        isdiff (bool): Flag indicating that ``res_dict`` contains a difference matrix; in this case, a symmetric plot range is required and a divergent colormap will be used
    """
    if isdiff:  # Difference plot
        assert -1 * vmin == vmax, "ERROR: Symmetric plot range required!"
        cmap = "PiYG"  # Symmetric (diverging) colormap
    else:  # Regular plot
        assert vmin == 0, "ERROR: Plot range including 0 required!"
        cmap = "hot_r"  # Regular colormap [color at 0 should be white (not actually drawn), to match figure background!]

    mat = res_dict["data"].tocoo()  # Convert to COO, for easy access to row/col and data!!
    col_idx = mat.data
    plt.scatter(
        mat.col,
        mat.row,
        marker=",",
        s=0.1,
        edgecolors="none",
        alpha=0.5,
        c=col_idx,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if not res_dict["data"].dtype == bool:
        cb = plt.colorbar()
        cb.set_label(res_dict["unit"])

    if fig_title is None:
        plt.title(res_dict["name"])
    else:
        plt.title(fig_title)

    plt.xlabel("Postsynaptic neurons")
    plt.ylabel("Presynaptic neurons")

    plt.axis("image")
    plt.xlim((-0.5, res_dict["data"].shape[1] - 0.5))
    plt.ylim((-0.5, res_dict["data"].shape[0] - 0.5))
    plt.gca().invert_yaxis()
