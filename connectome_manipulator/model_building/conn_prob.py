# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for building stochastic connection probability models of various model orders"""

from functools import partial
import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from voxcell import VoxelData

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import (
    get_node_ids,
    get_edges_population,
    get_node_positions,
    get_connections,
    get_cv_data,
)

JET = plt.get_cmap("jet")
HOT = plt.get_cmap("hot")


def extract(
    circuit,
    order,
    sel_src=None,
    sel_dest=None,
    sample_size=None,
    edges_popul_name=None,
    CV_dict=None,
    **kwargs,
):
    """Extracts the connection probabilities between samples of neurons.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        order (str): Model order, such as "1" (constant), "2" (distance-dependent), "3" (bipolar distance-dependent), "4" (offset-dependent), "4R" (reduced offset-dependent), "5" (position-dependent), "5R" (reduced position dependent)
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        sample_size (int): Size of random subsample of data to extract data from
        edges_popul_name (str): Name of SONATA egdes population to extract data from
        CV_dict (dict): Optional cross-validation dictionary, containing "n_folds" (int), "fold_idx" (int), "training_set" (bool) keys; will be automatically provided by the framework if "CV_folds" are specified
        **kwargs: Additional keyword arguments depending on the model order; see Notes

    Returns:
        dict: Dictionary containing the extracted connection probability data depending on the model order

    Note:
        For (optional) keyword arguments, see details in the respective helper functions:

        * Order 1: :func:`extract_1st_order`
        * Order 2: :func:`extract_2nd_order`
        * Order 3: :func:`extract_3rd_order`
        * Order 4: :func:`extract_4th_order`
        * Order 4R: :func:`extract_4th_order_reduced`
        * Order 5: :func:`extract_5th_order`
        * Order 5R: :func:`extract_5th_order_reduced`
    """
    log.info(
        f"Running order-{order} data extraction (sel_src={sel_src}, sel_dest={sel_dest}, sample_size={sample_size} neurons, CV_dict={CV_dict})..."
    )

    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)

    if map_file := kwargs.get("pos_map_file") is not None:
        log.log_assert(
            (src_nodes.name == tgt_nodes.name)
            or (isinstance(map_file, list) and len(map_file) == 2),
            f'Separate source/target position mappings required for different node populations "{src_nodes.name}" and "{tgt_nodes.name}"!',
        )

    if sample_size is None or sample_size <= 0:
        sample_size = np.inf  # Select all nodes
    if sample_size < len(node_ids_src) or sample_size < len(node_ids_dest):
        log.warning(
            "Sub-sampling neurons! Consider running model building with a different random sub-samples!"
        )
    sample_size_src = min(sample_size, len(node_ids_src))
    sample_size_dest = min(sample_size, len(node_ids_dest))
    log.log_assert(sample_size_src > 0 and sample_size_dest > 0, "ERROR: Empty nodes selection!")
    node_ids_src_sel = node_ids_src[
        np.random.permutation(
            [True] * sample_size_src + [False] * (len(node_ids_src) - sample_size_src)
        )
    ]
    node_ids_dest_sel = node_ids_dest[
        np.random.permutation(
            [True] * sample_size_dest + [False] * (len(node_ids_dest) - sample_size_dest)
        )
    ]

    # Cross-validation (optional)
    node_ids_src_sel, node_ids_dest_sel = get_cv_data(
        [node_ids_src_sel, node_ids_dest_sel], CV_dict
    )

    if not isinstance(order, str):
        order = str(order)

    if order == "1":
        return extract_1st_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == "2":
        return extract_2nd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == "3":
        return extract_3rd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == "4":
        return extract_4th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == "4R":
        return extract_4th_order_reduced(
            nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs
        )
    elif order == "5":
        return extract_5th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == "5R":
        return extract_5th_order_reduced(
            nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs
        )
    else:
        log.log_assert(False, f"ERROR: Order-{order} data extraction not supported!")
    return None


def build(order, **kwargs):
    """Builds a stochastic connection probability model from (binned) data.

    Args:
        order (str): Model order, such as "1" (constant), "2" (distance-dependent), "3" (bipolar distance-dependent), "4" (offset-dependent), "4R" (reduced offset-dependent), "5" (position-dependent), "5R" (reduced position dependent)
        **kwargs: Additional keyword arguments depending on the model order; see Notes

    Returns:
        Type depends on the model order; see Notes: Fitted stochastic connection probability model

    Note:
        For (optional) keyword arguments and return types, see details in the respective helper functions:

        * Order 1: :func:`build_1st_order`
        * Order 2: :func:`build_2nd_order`
        * Order 3: :func:`build_3rd_order`
        * Order 4: :func:`build_4th_order`
        * Order 4R: :func:`build_4th_order_reduced`
        * Order 5: :func:`build_5th_order`
        * Order 5R: :func:`build_5th_order_reduced`
    """
    log.info(f"Running order-{order} model building...")

    if not isinstance(order, str):
        order = str(order)

    if order == "1":
        return build_1st_order(**kwargs)
    elif order == "2":
        return build_2nd_order(**kwargs)
    elif order == "3":
        return build_3rd_order(**kwargs)
    elif order == "4":
        return build_4th_order(**kwargs)
    elif order.upper() == "4R":
        return build_4th_order_reduced(**kwargs)
    elif order == "5":
        return build_5th_order(**kwargs)
    elif order == "5R":
        return build_5th_order_reduced(**kwargs)
    else:
        log.log_assert(False, f"ERROR: Order-{order} model building not supported!")
    return None


def plot(order, **kwargs):
    """Visualizes extracted data vs. actual model output.

    Args:
        order (str): Model order, such as "1" (constant), "2" (distance-dependent), "3" (bipolar distance-dependent), "4" (offset-dependent), "4R" (reduced offset-dependent), "5" (position-dependent), "5R" (reduced position dependent)
        **kwargs: Additional keyword arguments depending on the model order; see Notes

    Note:
        For (optional) keyword arguments, see details in the respective helper functions:

        * Order 1: :func:`plot_1st_order`
        * Order 2: :func:`plot_2nd_order`
        * Order 3: :func:`plot_3rd_order`
        * Order 4: :func:`plot_4th_order`
        * Order 4R: :func:`plot_4th_order_reduced`
        * Order 5: :func:`plot_5th_order`
        * Order 5R: :func:`plot_5th_order_reduced`
    """
    log.info(f"Running order-{order} data/model visualization...")

    if not isinstance(order, str):
        order = str(order)

    if order == "1":
        return plot_1st_order(**kwargs)
    elif order == "2":
        return plot_2nd_order(**kwargs)
    elif order == "3":
        return plot_3rd_order(**kwargs)
    elif order == "4":
        return plot_4th_order(**kwargs)
    elif order == "4R":
        return plot_4th_order_reduced(**kwargs)
    elif order == "5":
        return plot_5th_order(**kwargs)
    elif order == "5R":
        return plot_5th_order_reduced(**kwargs)
    else:
        log.log_assert(False, f"ERROR: Order-{order} data/model visualization not supported!")
    return None


###################################################################################################
# Helper functions
###################################################################################################


def pos_accessor(pos_map, gids):
    """Access function"""
    if pos_map is None:
        return None
    else:
        return pos_map.apply(gids=gids)


def get_pos_mapping_fcts(pos_map_file):
    """Get access functions to one or two (src/tgt) position mappings from model (.json) or voxel data (.nrrd) file(s)."""

    def get_mapping(file):
        """Returns single mapping from .json or .nrrd file."""
        pos_acc = None
        vox_map = None
        if str.lower(os.path.splitext(file)[-1]) == ".json":
            _, pos_acc = load_pos_mapping_model(file)  # Model access
        elif str.lower(os.path.splitext(file)[-1]) == ".nrrd":
            vox_map = VoxelData.load_nrrd(file)  # Direct access to voxel data
            log.info(f"Loading position mapping (voxel data) from {file}")
            log.log_assert(vox_map.ndim == 3, "3D voxel data required!")
        else:
            log.log_assert(False, "Position mapping file error (must be .json or .nrrd)!")
        return pos_acc, vox_map

    if pos_map_file is None:
        pos_acc_src = pos_acc_tgt = None
        vox_map_src = vox_map_tgt = None
    elif isinstance(pos_map_file, list):
        log.log_assert(
            len(pos_map_file) == 2, "Two position mapping files (source/target) expected!"
        )
        log.log_assert(
            str.lower(os.path.splitext(pos_map_file[0])[-1])
            == str.lower(os.path.splitext(pos_map_file[1])[-1]),
            "Same file type for source/target position mappings required!",
        )
        pos_acc_src, vox_map_src = get_mapping(pos_map_file[0])
        pos_acc_tgt, vox_map_tgt = get_mapping(pos_map_file[1])
    else:  # Same mapping for src/tgt
        pos_acc_src, vox_map_src = get_mapping(pos_map_file)
        pos_acc_tgt = pos_acc_src
        vox_map_tgt = vox_map_src
    if pos_acc_src is None and pos_acc_tgt is None:
        pos_acc = None
    else:
        pos_acc = [pos_acc_src, pos_acc_tgt]
    if vox_map_src is None and vox_map_tgt is None:
        vox_map = None
    else:
        vox_map = [vox_map_src, vox_map_tgt]

    if pos_acc is None and vox_map is None:
        log.debug("No position mapping provided")

    return pos_acc, vox_map


def load_pos_mapping_model(pos_map_file):
    """Load a position mapping model from file (incl. access function)."""
    if pos_map_file is None:
        pos_map = None
        pos_acc = None
    else:
        log.log_assert(os.path.exists(pos_map_file), "Position mapping model file not found!")
        log.info(f"Loading position mapping model from {pos_map_file}")
        pos_map = model_types.AbstractModel.model_from_file(pos_map_file)
        log.log_assert(
            pos_map.input_names == ["gids"],
            'ERROR: Position mapping model error (must take "gids" as input)!',
        )
        pos_acc = partial(pos_accessor, pos_map)

    return pos_map, pos_acc


def get_neuron_positions(nodes, node_ids, pos_acc=None, vox_map=None):
    """Get neuron positions, optionally using a position mapping.

    Two types of mappings are supported:
    - pos_acc: Position access function indexed by node ID
    - vox_map: Voxel map accessed by node position
    """
    if pos_acc:  # Position mapping model provided
        nrn_pos = get_neuron_positions_by_id(pos_acc, node_ids)
        log.log_assert(
            vox_map is None, "Voxel map not supported when providing position access functions!"
        )
    else:
        nrn_pos = [
            get_node_positions(nodes[i], node_ids[i], vox_map[i] if vox_map else None)[1]
            for i in range(len(nodes))
        ]
    return nrn_pos


def get_neuron_positions_by_id(pos_fct, node_ids_list):
    """Get neuron positions indexed by node ID (using position access/mapping function).

    node_ids_list should be list of node_ids lists!
    """
    if not isinstance(pos_fct, list):
        pos_fct = [pos_fct for i in node_ids_list]
    else:
        log.log_assert(
            len(pos_fct) == len(node_ids_list),
            'ERROR: "pos_fct" must be scalar or a list with same length as "node_ids_list"!',
        )

    nrn_pos = [np.array(pos_fct[i](node_ids_list[i])) for i in range(len(node_ids_list))]

    return nrn_pos


def extract_dependent_p_conn(
    src_node_ids, tgt_node_ids, edges, dep_matrices, dep_bins, min_count_per_bin=None
):
    """Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins."""
    num_dep = len(dep_matrices)
    log.log_assert(len(dep_bins) == num_dep, "ERROR: Dependencies/bins mismatch!")
    log.log_assert(
        np.all(
            [
                dep_matrices[dim].shape == (len(src_node_ids), len(tgt_node_ids))
                for dim in range(num_dep)
            ]
        ),
        "ERROR: Matrix dimension mismatch!",
    )

    # Extract adjacency
    conns = get_connections(edges, src_node_ids, tgt_node_ids)
    if len(conns) > 0:
        adj_mat = csr_matrix(
            (np.full(conns.shape[0], True), conns.T.tolist()),
            shape=(max(src_node_ids) + 1, max(tgt_node_ids) + 1),
        )
    else:
        adj_mat = csr_matrix((max(src_node_ids) + 1, max(tgt_node_ids) + 1))  # Empty matrix
    if np.any(adj_mat.diagonal()):
        log.debug("Autaptic connection(s) found!")

    # Extract connection probability
    num_bins = [len(b) - 1 for b in dep_bins]
    bin_indices = [list(range(n)) for n in num_bins]
    count_all = np.full(
        num_bins, -1
    )  # Count of all pairs of neurons for each combination of dependencies
    count_conn = np.full(
        num_bins, -1
    )  # Count of connected pairs of neurons for each combination of dependencies

    log.debug(
        f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...'
    )
    pbar = progressbar.ProgressBar(maxval=np.prod(num_bins) - 1)
    for idx in pbar(itertools.product(*bin_indices)):
        dep_sel = np.full((len(src_node_ids), len(tgt_node_ids)), True)
        for dim in range(num_dep):
            lower = dep_bins[dim][idx[dim]]
            upper = dep_bins[dim][idx[dim] + 1]
            dep_sel = np.logical_and(
                dep_sel,
                np.logical_and(
                    dep_matrices[dim] >= lower,
                    (
                        (dep_matrices[dim] < upper)
                        if idx[dim] < num_bins[dim] - 1
                        else (dep_matrices[dim] <= upper)
                    ),
                ),
            )  # Including last edge
        sidx, tidx = np.nonzero(dep_sel)
        count_all[idx] = np.sum(dep_sel)
        # count_conn[idx] = np.sum(adj_mat[src_node_ids[sidx], tgt_node_ids[tidx]]) # ERROR in scipy/sparse/compressed.py if len(sidx) >= 2**31: "ValueError: could not convert integer scalar"
        # [WORKAROUND]: Split indices into parts of 2**31-1 length and sum them separately
        sidx_split = np.split(sidx, np.arange(0, len(sidx), 2**31 - 1)[1:])
        tidx_split = np.split(tidx, np.arange(0, len(tidx), 2**31 - 1)[1:])
        count_split = 0
        for s, t in zip(sidx_split, tidx_split):
            count_split = count_split + np.sum(adj_mat[src_node_ids[s], tgt_node_ids[t]])
        count_conn[idx] = count_split
    p_conn = np.array(count_conn / count_all)
    # p_conn[np.isnan(p_conn)] = 0.0

    # Check bin counts below threshold and ignore
    if min_count_per_bin is None:
        min_count_per_bin = 0  # No threshold
    bad_bins = np.logical_and(count_all > 0, count_all < min_count_per_bin)
    if np.sum(bad_bins) > 0:
        log.warning(
            f"Found {np.sum(bad_bins)} of {count_all.size} ({100.0 * np.sum(bad_bins) / count_all.size:.1f}%) bins with less than th={min_count_per_bin} pairs of neurons ... IGNORING! (Consider increasing sample size and/or bin size and/or smoothing!)"
        )
        p_conn[bad_bins] = np.nan  # 0.0

    return p_conn, count_conn, count_all


def get_value_ranges(max_range, num_coords, pos_range=False):
    """Returns ranges of values for given max. ranges (strictly positive incl. zero, symmetric around zero, or arbitrary)"""
    if np.isscalar(pos_range):
        pos_range = [pos_range for i in range(num_coords)]
    else:
        if num_coords == 1:  # Special case
            pos_range = [pos_range]
        log.log_assert(
            len(pos_range) == num_coords, f"ERROR: pos_range must have {num_coords} elements!"
        )

    if np.isscalar(max_range):
        max_range = [max_range for i in range(num_coords)]
    else:
        if num_coords == 1:  # Special case
            max_range = [max_range]
        log.log_assert(
            len(max_range) == num_coords, f"ERROR: max_range must have {num_coords} elements!"
        )

    val_ranges = []
    for ridx, (r, p) in enumerate(zip(max_range, pos_range)):
        if np.isscalar(r):
            log.log_assert(r > 0.0, f"ERROR: Maximum range of coord {ridx} must be larger than 0!")
            if p:  # Positive range
                val_ranges.append([0, r])
            else:  # Symmetric range
                val_ranges.append([-r, r])
        else:  # Arbitrary range
            log.log_assert(len(r) == 2 and r[0] < r[1], f"ERROR: Range of coord {ridx} invalid!")
            if p:
                log.log_assert(r[0] == 0, f"ERROR: Range of coord {ridx} must include 0!")
            val_ranges.append(r)

    if num_coords == 1:  # Special case
        return val_ranges[0]
    else:
        return val_ranges


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   1st order model (Erdos-Renyi)
###################################################################################################


def extract_1st_order(_nodes, edges, src_node_ids, tgt_node_ids, min_count_per_bin=10, **_):
    """Extracts the average connection probability (1st order) from a sample of pairs of neurons.

    Args:
        _nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes - Not used
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        min_count_per_bin (int): Minimum number of samples; otherwise, no estimate will be made

    Returns:
        dict: Dictionary containing the extracted 1st-order connection probability data
    """
    p_conn, conn_count, _ = extract_dependent_p_conn(
        src_node_ids, tgt_node_ids, edges, [], [], min_count_per_bin
    )

    src_cell_count = len(src_node_ids)
    tgt_cell_count = len(tgt_node_ids)
    log.info(
        f"Found {conn_count} connections between {src_cell_count}x{tgt_cell_count} neurons (p = {p_conn:.3f})"
    )

    return {"p_conn": p_conn, "src_cell_count": src_cell_count, "tgt_cell_count": tgt_cell_count}


def build_1st_order(p_conn, **_):
    """Builds a stochastic 1st order connection probability model (Erdos-Renyi).

    Args:
        p_conn (float): Constant connection probability, as returned by :func:`extract_1st_order`

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb1stOrderModel: Resulting stochastic 1st order connectivity model
    """
    # Create model
    model = model_types.ConnProb1stOrderModel(p_conn=float(p_conn))
    log.debug("Model description:\n%s", model)

    return model


def plot_1st_order(out_dir, p_conn, src_cell_count, tgt_cell_count, model, **_):  # pragma: no cover
    """Visualizes 1st order extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn (float): Constant connection probability, as returned by :func:`extract_1st_order`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_1st_order`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_1st_order`
        model (connectome_manipulator.model_building.model_types.ConnProb1stOrderModel): Fitted stochastic 1st order connectivity model, as returned by :func:`build_1st_order`
    """
    model_params = model.get_param_dict()
    model_str = f'f(x) = {model_params["p_conn"]:.3f}'

    # Draw figure
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(
        0.5,
        p_conn,
        width=1,
        facecolor="tab:blue",
        label=f"Data: N = {src_cell_count}x{tgt_cell_count} cells",
    )
    plt.plot(
        [-0.5, 1.5],
        np.ones(2) * model.get_conn_prob(),
        "--",
        color="tab:red",
        label=f"Model: {model_str}",
    )
    plt.text(0.5, 0.99 * p_conn, f"p = {p_conn:.3f}", color="k", ha="center", va="top")
    plt.xticks([])
    plt.ylabel("Conn. prob.")
    plt.title("Average conn. prob. (1st-order)", fontweight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################


def extract_2nd_order(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    bin_size_um=100,
    max_range_um=None,
    pos_map_file=None,
    min_count_per_bin=10,
    **_,
):
    """Extracts the binned, distance-dependent connection probabilities (2nd order) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        bin_size_um (float): Distance bin size in um
        max_range_um (float): Maximum distance range in um
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin

    Returns:
        dict: Dictionary containing the extracted 2nd-order connection probability data
    """
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute distance matrix
    dist_mat = model_types.ConnProb2ndOrderExpModel.compute_dist_matrix(src_nrn_pos, tgt_nrn_pos)

    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    log.log_assert(
        max_range_um > 0 and bin_size_um > 0,
        "ERROR: Max. range and bin size must be larger than 0um!",
    )
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um

    p_conn_dist, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids, tgt_node_ids, edges, [dist_mat], [dist_bins], min_count_per_bin
    )

    return {
        "p_conn_dist": p_conn_dist,
        "count_conn": count_conn,
        "count_all": count_all,
        "dist_bins": dist_bins,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_2nd_order(
    p_conn_dist, dist_bins, count_all, model_specs=None, rel_fit_err_th=None, strict_fit=False, **_
):
    """Builds a stochastic 2nd order connection probability model (exponential distance-dependent).

    Args:
        p_conn_dist (numpy.ndarray): Binned connection probabilities, as retuned by :func:`extract_2nd_order`
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract_2nd_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_2nd_order`
        model_specs (dict): Model specifications; see Notes
        rel_fit_err_th (float): Threshold for rel. standard error of the coefficients; exceeding the threshold will return an invalid model
        strict_fit (bool): Flag to enforce strict model fitting, which means that first data bin must contain valid data (otherwise, there is a risk of a bad extrapolation at low distances)

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb2ndOrder[Complex]ExpModel: Resulting stochastic 2nd order connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; either "SimpleExponential" (2 parameters) or "ComplexExponential" (5 parameters)
        * p0 (list-like): Initial guess for parameter fit, as used in :func:`scipy.optimize.curve_fit`
        * bounds (list-like): Lower and upper bounds on parameters, as used in :func:`scipy.optimize.curve_fit`
    """
    if model_specs is None:
        model_specs = {"type": "SimpleExponential"}

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    X = dist_bins[:-1][np.isfinite(p_conn_dist)] + bin_offset
    y = p_conn_dist[np.isfinite(p_conn_dist)]

    if model_specs.get("type") == "SimpleExponential":  # Exponential curve fit with 2 parameters
        # Fit simple model
        def exp_model(x, a, b):
            return a * np.exp(-b * np.array(x))

        if np.sum(y) == 0.0:  # Special case: No connections at all, skipping model fit
            a_opt = b_opt = 0.0
        else:
            p0 = model_specs.get("p0", [0.0, 0.0])
            bounds = model_specs.get("bounds", [0.0, np.inf])
            invalid_model = False
            try:
                (a_opt, b_opt), pcov, *_ = curve_fit(exp_model, X, y, p0=p0, bounds=bounds)
            except (
                ValueError,
                RuntimeError,
            ) as e:  # Raised if input data invalid or optimization fails
                log.error(e)
                invalid_model = True

            if not invalid_model and not 0.0 <= a_opt <= 1.0:
                log.error('"Scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model:
                rel_err = np.sqrt(np.diag(pcov)) / np.array(
                    [a_opt, b_opt]
                )  # Rel. standard error of the coefficients
                log.debug(f"Rel. error of simple 2nd-order model fit: {rel_err}")
                if rel_fit_err_th is not None and (
                    not all(np.isfinite(rel_err)) or max(rel_err) > rel_fit_err_th
                ):
                    log.error(
                        f"Rel. error of model fit exceeds error threshold of {rel_fit_err_th} (or could not be determined)!"
                    )
                    invalid_model = True

            if not invalid_model and strict_fit:
                if not np.isfinite(p_conn_dist[0]):
                    # Strict fit: Must contain data in the first bin (otherwise, resulting in potentially bad extrapolation at low distances)
                    log.error("Strict fit violation: Lowest-distance bin empty!")
                    invalid_model = True

            if invalid_model:
                a_opt = b_opt = np.nan

        # Create simple model
        model = model_types.ConnProb2ndOrderExpModel(scale=float(a_opt), exponent=float(b_opt))

    elif (
        model_specs.get("type") == "ComplexExponential"
    ):  # Complex (dual) exponential curve fit with 5 parameters [capturing deflection towards distance zero and slowly decaying offset at large distances]
        # Fit complex model
        def exp_model(x, a, b, c, d, e):
            return a * np.exp(-b * np.array(x) ** c) + d * np.exp(-e * np.array(x))

        if np.sum(y) == 0.0:  # Special case: No connections at all, skipping model fit
            a_opt = b_opt = c_opt = d_opt = e_opt = 0.0
        else:
            p0 = model_specs.get("p0", [0.0, 0.0, 1.0, 0.0, 0.0])
            bounds = model_specs.get(
                "bounds", [[0.0, 0.0, 1.0, 0.0, 0.0], [np.inf, np.inf, 2.0, np.inf, np.inf]]
            )
            invalid_model = False
            try:
                (a_opt, b_opt, c_opt, d_opt, e_opt), pcov, *_ = curve_fit(
                    exp_model, X, y, p0=p0, bounds=bounds
                )
            except (
                ValueError,
                RuntimeError,
            ) as e:  # Raised if input data invalid or optimization fails
                log.error(e)
                invalid_model = True

            if not invalid_model and not 0.0 <= a_opt <= 1.0:
                log.error('Proximal "scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model and not 0.0 <= d_opt <= 1.0:
                log.error('Distal "scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model:
                rel_err = np.sqrt(np.diag(pcov)) / np.array(
                    [a_opt, b_opt, c_opt, d_opt, e_opt]
                )  # Rel. standard error of the coefficients
                log.debug(f"Rel. error of complex 2nd-order model fit: {rel_err}")
                if rel_fit_err_th is not None and (
                    not all(np.isfinite(rel_err)) or max(rel_err) > rel_fit_err_th
                ):
                    log.error(
                        f"Rel. error of model fit exceeds error threshold of {rel_fit_err_th} (or could not be determined)!"
                    )
                    invalid_model = True

            if not invalid_model and strict_fit:
                if not np.isfinite(p_conn_dist[0]):
                    # Strict fit: Must contain data in the first bin (otherwise, resulting in potentially bad extrapolation at low distances)
                    log.error("Strict fit violation: Lowest-distance bin empty!")
                    invalid_model = True

            if invalid_model:
                a_opt = b_opt = c_opt = d_opt = e_opt = np.nan

        # Create complex model
        model = model_types.ConnProb2ndOrderComplexExpModel(
            prox_scale=float(a_opt),
            prox_exp=float(b_opt),
            prox_exp_pow=float(c_opt),
            dist_scale=float(d_opt),
            dist_exp=float(e_opt),
        )

    else:
        log.log_assert(False, "ERROR: Model type not specified or unknown!")

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_dist * count_all).astype(int)
    p_conn_model = model.get_conn_prob(distance=dist_bins[:-1] + bin_offset)
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_2nd_order(
    out_dir,
    p_conn_dist,
    count_conn,
    count_all,
    dist_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    pos_map_file=None,
    **_,
):  # pragma: no cover
    """Visualizes 2nd order extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_dist (numpy.ndarray): Binned connection probabilities, as retuned by :func:`extract_2nd_order`
        count_conn (numpy.ndarray): Count of all connected pairs of neurons (i.e., all actual connections) in each bin, as retuned by :func:`extract_2nd_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_2nd_order`
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract_2nd_order`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_2nd_order`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_2nd_order`
        model (connectome_manipulator.model_building.model_types.ConnProb2ndOrder[Complex]ExpModel): Fitted stochastic 2nd order connectivity model, as returned by :func:`extract_2nd_order`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
    """
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_str = str(model).split("\n")[1].split("=")[-1].strip()

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    plt.plot(
        dist_bins[:-1] + bin_offset,
        p_conn_dist,
        ".-",
        label=f"Data: N = {src_cell_count}x{tgt_cell_count} cells",
    )
    plt.plot(dist_model, model.get_conn_prob(dist_model), "--", label="Model: " + model_str)
    plt.grid()
    plt.xlabel("Distance [$\\mu$m]")
    plt.ylabel("Conn. prob.")
    plt.title("Data vs. model fit")
    plt.legend(fontsize=6)

    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500  # (um)
    r_markers = [200, 400]  # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model.get_conn_prob(vdist)
    plt.imshow(
        pdist,
        interpolation="bilinear",
        extent=(-plot_range, plot_range, -plot_range, plot_range),
        cmap=HOT,
        vmin=0.0,
    )
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor="w", linestyle="--", fill=False))
        plt.text(0, r, f"{r} $\\mu$m", color="w", ha="center", va="bottom")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$\\Delta$x")
    plt.ylabel("$\\Delta$z")
    plt.title("2D model")
    plt.colorbar(label="Conn. prob.")

    plt.suptitle(
        f"Distance-dependent connection probability model (2nd order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Data counts
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(dist_bins[:-1] + bin_offset, count_all, width=1.5 * bin_offset, label="All pair count")
    plt.bar(
        dist_bins[:-1] + bin_offset, count_conn, width=1.0 * bin_offset, label="Connection count"
    )
    plt.gca().set_yscale("log")
    plt.grid()
    plt.xlabel("Distance [$\\mu$m]")
    plt.ylabel("Count")
    plt.title(
        f"Distance-dependent connection counts (N = {src_cell_count}x{tgt_cell_count} cells)\n<Position mapping: {pos_map_file}>"
    )
    plt.legend()
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_counts.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   3rd order (bipolar distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################


def extract_3rd_order(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    bin_size_um=100,
    max_range_um=None,
    pos_map_file=None,
    no_dist_mapping=False,
    min_count_per_bin=10,
    bip_coord=2,
    **_,
):
    """Extracts the binned, bipolar distance-dependent connection probability (3rd order) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        bin_size_um (float): Distance bin size in um
        max_range_um (float): Maximum distance range in um
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        no_dist_mapping (bool): Flag to disable position mapping for computing distances, i.e., position mapping will only be used to determine the bipolar coordinate if selected
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin
        bip_coord (int): Index to select bipolar coordinate axis (0..x, 1..y, 2..z), usually perpendicular to layers

    Returns:
        dict: Dictionary containing the extracted 3rd-order connection probability data
    """
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos_raw, tgt_nrn_pos_raw = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], None, None
    )
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute distance matrix
    if no_dist_mapping:  # Don't use position mapping for computing distances
        dist_mat = model_types.ConnProb3rdOrderExpModel.compute_dist_matrix(
            src_nrn_pos_raw, tgt_nrn_pos_raw
        )
    else:  # Use position mapping for computing distances
        dist_mat = model_types.ConnProb3rdOrderExpModel.compute_dist_matrix(
            src_nrn_pos, tgt_nrn_pos
        )

    # Compute bipolar matrix (always using position mapping, if provided; along z-axis (by default); post-synaptic neuron below (delta_z < 0) or above (delta_z > 0) pre-synaptic neuron)
    bip_mat = model_types.ConnProb3rdOrderExpModel.compute_bip_matrix(
        src_nrn_pos, tgt_nrn_pos, bip_coord
    )

    # Extract bipolar distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    log.log_assert(
        max_range_um > 0 and bin_size_um > 0,
        "ERROR: Max. range and bin size must be larger than 0um!",
    )
    num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
    bip_bins = [np.min(bip_mat), 0, np.max(bip_mat)]

    p_conn_dist_bip, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids,
        tgt_node_ids,
        edges,
        [dist_mat, bip_mat],
        [dist_bins, bip_bins],
        min_count_per_bin,
    )

    return {
        "p_conn_dist_bip": p_conn_dist_bip,
        "count_conn": count_conn,
        "count_all": count_all,
        "dist_bins": dist_bins,
        "bip_bins": bip_bins,
        "bip_coord_data": bip_coord,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_3rd_order(
    p_conn_dist_bip,
    dist_bins,
    count_all,
    bip_coord_data,
    model_specs=None,
    rel_fit_err_th=None,
    strict_fit=False,
    **_,
):
    """Builds a stochastic 3rd order connection probability model (bipolar exponential distance-dependent).

    Args:
        p_conn_dist_bip (numpy.ndarray): Binned bipolar connection probabilities, as retuned by :func:`extract_3rd_order`
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract_3rd_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_3rd_order`
        bip_coord_data (int): Index of bipolar coordinate axis, as returned by :func:`extract_3rd_order`
        model_specs (dict): Model specifications; see Notes
        rel_fit_err_th (float): Threshold for rel. standard error of the coefficients; exceeding the threshold will return an invalid model
        strict_fit (bool): Flag to enforce strict model fitting, which means that first data bin must contain valid data (otherwise, there is a risk of a bad extrapolation at low distances)

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb3rdOrder[Complex]ExpModel: Resulting stochastic 3rd order connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; either "SimpleExponential" (2 parameters) or "ComplexExponential" (5 parameters)
        * p0 (list-like): Initial guess for parameter fit, as used in :func:`scipy.optimize.curve_fit`
        * bounds (list-like): Lower and upper bounds on parameters, as used in :func:`scipy.optimize.curve_fit`
    """
    if model_specs is None:
        model_specs = {"type": "SimpleExponential"}

    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    X = dist_bins[:-1][np.all(np.isfinite(p_conn_dist_bip), 1)] + bin_offset
    y = p_conn_dist_bip[np.all(np.isfinite(p_conn_dist_bip), 1), :]

    if model_specs.get("type") == "SimpleExponential":  # Exponential curve fit with 2x2 parameters
        # Fit simple model
        def exp_model(x, a, b):
            return a * np.exp(-b * np.array(x))

        if np.sum(y) == 0.0:  # Special case: No connections at all, skipping model fit
            aN_opt = bN_opt = 0.0
            aP_opt = bP_opt = 0.0
        else:
            p0 = model_specs.get("p0", [0.0, 0.0])
            bounds = model_specs.get("bounds", [0.0, np.inf])
            invalid_model = False
            try:
                (aN_opt, bN_opt), pcovN, *_ = curve_fit(exp_model, X, y[:, 0], p0=p0, bounds=bounds)
                (aP_opt, bP_opt), pcovP, *_ = curve_fit(exp_model, X, y[:, 1], p0=p0, bounds=bounds)
            except (
                ValueError,
                RuntimeError,
            ) as e:  # Raised if input data invalid or optimization fails
                log.error(e)
                invalid_model = True

            if not invalid_model and (not 0.0 <= aN_opt <= 1.0 or not 0.0 <= aP_opt <= 1.0):
                log.error('"Scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model:
                rel_err = np.hstack([np.sqrt(np.diag(pcovN)), np.sqrt(np.diag(pcovP))]) / np.array(
                    [aN_opt, bN_opt, aP_opt, bP_opt]
                )  # Rel. standard error of the coefficients
                log.debug(f"Rel. error of simple 3rd-order model fit: {rel_err}")
                if rel_fit_err_th is not None and (
                    not all(np.isfinite(rel_err)) or max(rel_err) > rel_fit_err_th
                ):
                    log.error(
                        f"Rel. error of model exceeds error threshold of {rel_fit_err_th} (or could not be determined)!"
                    )
                    invalid_model = True

            if not invalid_model and strict_fit:
                if not all(np.isfinite(p_conn_dist_bip[0, :])):
                    # Strict fit: Must contain data in the first bin (otherwise, resulting in potentially bad extrapolation at low distances)
                    log.error("Strict fit violation: Lowest-distance bin empty!")
                    invalid_model = True

            if invalid_model:
                aN_opt = bN_opt = np.nan
                aP_opt = bP_opt = np.nan

        # Create simple model
        model = model_types.ConnProb3rdOrderExpModel(
            scale_N=float(aN_opt),
            exponent_N=float(bN_opt),
            scale_P=float(aP_opt),
            exponent_P=float(bP_opt),
            bip_coord=int(bip_coord_data),
        )

    elif (
        model_specs.get("type") == "ComplexExponential"
    ):  # Complex (dual) exponential curve fit with 2x5 parameters [capturing deflection towards distance zero and slowly decaying offset at large distances]
        # Fit complex model
        def exp_model(x, a, b, c, d, e):
            return a * np.exp(-b * np.array(x) ** c) + d * np.exp(-e * np.array(x))

        if np.sum(y) == 0.0:  # Special case: No connections at all, skipping model fit
            aN_opt = bN_opt = cN_opt = dN_opt = eN_opt = 0.0
            aP_opt = bP_opt = cP_opt = dP_opt = eP_opt = 0.0
        else:
            p0 = model_specs.get("p0", [0.0, 0.0, 1.0, 0.0, 0.0])
            bounds = model_specs.get(
                "bounds", [[0.0, 0.0, 1.0, 0.0, 0.0], [np.inf, np.inf, 2.0, np.inf, np.inf]]
            )
            invalid_model = False
            try:
                (aN_opt, bN_opt, cN_opt, dN_opt, eN_opt), pcovN, *_ = curve_fit(
                    exp_model, X, y[:, 0], p0=p0, bounds=bounds
                )
                (aP_opt, bP_opt, cP_opt, dP_opt, eP_opt), pcovP, *_ = curve_fit(
                    exp_model, X, y[:, 1], p0=p0, bounds=bounds
                )
            except (
                ValueError,
                RuntimeError,
            ) as e:  # Raised if input data invalid or optimization fails
                log.error(e)
                invalid_model = True

            if not invalid_model and (not 0.0 <= aN_opt <= 1.0 or not 0.0 <= aP_opt <= 1.0):
                log.error('Proximal "scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model and (not 0.0 <= dN_opt <= 1.0 or not 0.0 <= dP_opt <= 1.0):
                log.error('Distal "scale" must be between 0 and 1!')
                invalid_model = True

            if not invalid_model:
                rel_err = np.hstack([np.sqrt(np.diag(pcovN)), np.sqrt(np.diag(pcovP))]) / np.array(
                    [aN_opt, bN_opt, cN_opt, dN_opt, eN_opt, aP_opt, bP_opt, cP_opt, dP_opt, eP_opt]
                )  # Rel. standard error of the coefficients
                log.debug(f"Rel. error of complex 3rd-order model fit: {rel_err}")
                if rel_fit_err_th is not None and (
                    not all(np.isfinite(rel_err)) or max(rel_err) > rel_fit_err_th
                ):
                    log.error(
                        f"Rel. error of model fit exceeds error threshold of {rel_fit_err_th} (or could not be determined)!"
                    )
                    invalid_model = True

            if not invalid_model and strict_fit:
                if not all(np.isfinite(p_conn_dist_bip[0, :])):
                    # Strict fit: Must contain data in the first bin (otherwise, resulting in potentially bad extrapolation at low distances)
                    log.error("Strict fit violation: Lowest-distance bin empty!")
                    invalid_model = True

            if invalid_model:
                aN_opt = bN_opt = cN_opt = dN_opt = eN_opt = np.nan
                aP_opt = bP_opt = cP_opt = dP_opt = eP_opt = np.nan

        # Create complex model
        model = model_types.ConnProb3rdOrderComplexExpModel(
            prox_scale_N=float(aN_opt),
            prox_exp_N=float(bN_opt),
            prox_exp_pow_N=float(cN_opt),
            dist_scale_N=float(dN_opt),
            dist_exp_N=float(eN_opt),
            prox_scale_P=float(aP_opt),
            prox_exp_P=float(bP_opt),
            prox_exp_pow_P=float(cP_opt),
            dist_scale_P=float(dP_opt),
            dist_exp_P=float(eP_opt),
            bip_coord=int(bip_coord_data),
        )
    else:
        log.log_assert(False, "ERROR: Model type not specified or unknown!")

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_dist_bip * count_all).astype(int)
    p_conn_model = np.array(
        [model.get_conn_prob(distance=dist_bins[:-1] + bin_offset, bip=bip) for bip in [-1, 1]]
    ).T
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_3rd_order(
    out_dir,
    p_conn_dist_bip,
    count_conn,
    count_all,
    dist_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    bip_coord_data,
    pos_map_file=None,
    **_,
):  # pragma: no cover
    """Visualizes 3rd order extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_dist_bip (numpy.ndarray): Binned bipolar connection probabilities, as retuned by :func:`extract_3rd_order`
        count_conn (numpy.ndarray): Count of all connected pairs of neurons (i.e., all actual connections) in each bin, as retuned by :func:`extract_3rd_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_3rd_order`
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract_3rd_order`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_3rd_order`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_3rd_order`
        model (connectome_manipulator.model_building.model_types.ConnProb3rdOrder[Complex]ExpModel): Fitted stochastic 3rd order connectivity model, as returned by :func:`extract_3rd_order`
        bip_coord_data (int): Index of bipolar coordinate axis, as returned by :func:`extract_3rd_order`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
    """
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_strN = str(model).split("\n")[1].split("=")[-1].strip()
    model_strP = str(model).split("\n")[2].split("=")[-1].strip()

    coord_names = ["x", "y", "z"]
    a0_name = coord_names[bip_coord_data]  # Bipolar coordinate name for plotting on bipolar axis
    a1_name = [c for c in coord_names if c != a0_name][
        0
    ]  # Other (arbitrary) coordinate name for plotting on perpendicular axis

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    bip_dist = np.concatenate(
        (-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset)
    )
    bip_data = np.concatenate((p_conn_dist_bip[::-1, 0], [np.nan], p_conn_dist_bip[:, 1]))
    plt.plot(bip_dist, bip_data, ".-", label=f"Data: N = {src_cell_count}x{tgt_cell_count} cells")
    plt.plot(
        -dist_model,
        model.get_conn_prob(dist_model, np.sign(-dist_model)),
        "--",
        label="Model: " + model_strN,
    )
    plt.plot(
        dist_model,
        model.get_conn_prob(dist_model, np.sign(dist_model)),
        "--",
        label="Model: " + model_strP,
    )
    plt.grid()
    plt.xlabel(f"sign($\\Delta${a0_name}) * Distance [$\\mu$m]")
    plt.ylabel("Conn. prob.")
    plt.title("Data vs. model fit")
    plt.legend(loc="upper left", fontsize=6)

    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500  # (um)
    r_markers = [200, 400]  # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model.get_conn_prob(vdist, np.sign(zv))
    plt.imshow(
        pdist,
        interpolation="bilinear",
        extent=(-plot_range, plot_range, -plot_range, plot_range),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor="w", linestyle="--", fill=False))
        plt.text(0, r, f"{r} $\\mu$m", color="w", ha="center", va="bottom")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"$\\Delta${a1_name}")
    plt.ylabel(f"$\\Delta${a0_name}")
    plt.title("2D model")
    plt.colorbar(label="Conn. prob.")

    plt.suptitle(
        f"Bipolar distance-dependent connection probability model (3rd order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Data counts
    plt.figure(figsize=(6, 4), dpi=300)
    bip_dist = np.concatenate(
        (-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset)
    )
    bip_count_all = np.concatenate((count_all[::-1, 0], [np.nan], count_all[:, 1]))
    bip_count_conn = np.concatenate((count_conn[::-1, 0], [np.nan], count_conn[:, 1]))
    plt.bar(bip_dist, bip_count_all, width=1.5 * bin_offset, label="All pair count")
    plt.bar(bip_dist, bip_count_conn, width=1.0 * bin_offset, label="Connection count")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.xlabel(f"sign($\\Delta${a0_name}) * Distance [$\\mu$m]")
    plt.ylabel("Count")
    plt.title(
        f"Bipolar distance-dependent connection counts (N = {src_cell_count}x{tgt_cell_count} cells)\n<Position mapping: {pos_map_file}>"
    )
    plt.legend()
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_counts.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   4th order (offset-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'type' (such as 'LinearInterpolation')
#                    and optionally, 'kwargs' may be provided
###################################################################################################


def extract_4th_order(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    bin_size_um=100,
    max_range_um=None,
    pos_map_file=None,
    min_count_per_bin=10,
    **_,
):
    """Extracts the binned, offset-dependent connection probability (4th order) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        bin_size_um (float/list-like): Offset bin size in um; can be scalar (same value for x/y/z dimension) or list-like with three individual values for x/y/z dimensions
        max_range_um (float/list-like): Maximum offset range in um; can be scalar (same +/- value for all dimensions) or list-like with three elements for x/y/z dimensions each of which can be either a scalar (same +/- ranges) or a two-element list with individual +/- ranges
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin

    Returns:
        dict: Dictionary containing the extracted 4th-order connection probability data
    """
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute dx/dy/dz offset matrices
    dx_mat, dy_mat, dz_mat = model_types.ConnProb4thOrderLinInterpnModel.compute_offset_matrices(
        src_nrn_pos, tgt_nrn_pos
    )

    # Extract offset-dependent connection probabilities
    if max_range_um is None:
        dx_range, dy_range, dz_range = zip(
            [np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)],
            [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)],
        )
    else:
        dx_range, dy_range, dz_range = get_value_ranges(max_range_um, 3, pos_range=False)

    if np.isscalar(bin_size_um):  # Single scalar range value to be used for all dimensions
        log.log_assert(bin_size_um > 0.0, "ERROR: Offset bin size must be larger than 0um!")
        bin_size_dx = bin_size_dy = bin_size_dz = bin_size_um
    else:  # Three values for x/y/z dimensions
        log.log_assert(
            len(bin_size_um) == 3, "ERROR: Offset bin sizes in x/y/z dimension expected!"
        )
        log.log_assert(
            np.all([b > 0.0 for b in bin_size_um]),
            "ERROR: Offset bin size must be larger than 0um!",
        )
        bin_size_dx, bin_size_dy, bin_size_dz = bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_offset, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids,
        tgt_node_ids,
        edges,
        [dx_mat, dy_mat, dz_mat],
        [dx_bins, dy_bins, dz_bins],
        min_count_per_bin,
    )

    return {
        "p_conn_offset": p_conn_offset,
        "count_conn": count_conn,
        "count_all": count_all,
        "dx_bins": dx_bins,
        "dy_bins": dy_bins,
        "dz_bins": dz_bins,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_4th_order(
    p_conn_offset,
    dx_bins,
    dy_bins,
    dz_bins,
    count_all,
    model_specs=None,
    smoothing_sigma_um=None,
    **_,
):
    """Builds a stochastic 4th order connection probability model (offset-dependent, based on linear interpolation).

    Args:
        p_conn_offset (numpy.ndarray): Binned offset-dependent connection probabilities, as retuned by :func:`extract_4th_order`
        dx_bins (numpy.ndarray): Offset bin edges along x-axis, as returned by :func:`extract_4th_order`
        dy_bins (numpy.ndarray): Offset bin edges along y-axis, as returned by :func:`extract_4th_order`
        dz_bins (numpy.ndarray): Offset bin edges along z-axis, as returned by :func:`extract_4th_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_4th_order`
        model_specs (dict): Model specifications; see Notes
        smoothing_sigma_um (float/list-like): Sigma in um for Gaussian smoothing; can be scalar (same value for x/y/z dimension) or list-like with three individual values for x/y/z dimensions

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb4thOrderLinInterpnModel: Resulting stochastic 4th order connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; only "LinearInterpolation" supported which does not require any additional specs
    """
    if model_specs is None:
        model_specs = {"type": "LinearInterpolation"}

    bin_sizes = [np.diff(dx_bins[:2])[0], np.diff(dy_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    dx_bin_offset = 0.5 * bin_sizes[0]
    dy_bin_offset = 0.5 * bin_sizes[1]
    dz_bin_offset = 0.5 * bin_sizes[2]

    dx_pos = dx_bins[:-1] + dx_bin_offset  # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset  # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset  # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_offset = p_conn_offset.copy()
    p_conn_offset[np.isnan(p_conn_offset)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 3  # Same value for all coordinates
        else:
            log.log_assert(
                len(smoothing_sigma_um) == 3, "ERROR: Smoothing sigma for 3 dimensions required!"
            )
        log.log_assert(
            np.all(np.array(smoothing_sigma_um) >= 0.0),
            "ERROR: Smoothing sigma must be non-negative!",
        )
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.debug(
            f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms'
        )
        p_conn_offset = gaussian_filter(p_conn_offset, sigmas, mode="constant")

    model_inputs = ["dx", "dy", "dz"]  # Must be the same for all interpolation types!
    if (
        model_specs.get("type") == "LinearInterpolation"
    ):  # Linear interpolation model => Removing dimensions with only single value from interpolation
        log.log_assert(
            len(model_specs.get("kwargs", {})) == 0,
            f'ERROR: No parameters expected for "{model_specs.get("type")}" model!',
        )

        # Create model
        index = pd.MultiIndex.from_product([dx_pos, dy_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_offset.flatten(), index=index, columns=["p"])
        model = model_types.ConnProb4thOrderLinInterpnModel(p_conn_table=df)

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("type")}" unknown!')

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_offset * count_all).astype(int)
    dxv, dyv, dzv = np.meshgrid(dx_pos, dy_pos, dz_pos, indexing="ij")
    p_conn_model = model.get_conn_prob(dx=dxv, dy=dyv, dz=dzv)
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_4th_order(
    out_dir,
    p_conn_offset,
    dx_bins,
    dy_bins,
    dz_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    pos_map_file=None,
    plot_model_ovsampl=3,
    plot_model_extsn=0,
    **_,
):  # pragma: no cover
    """Visualizes 4th order extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_offset (numpy.ndarray): Binned offset-dependent connection probabilities, as retuned by :func:`extract_4th_order`
        dx_bins (numpy.ndarray): Offset bin edges along x-axis, as returned by :func:`extract_4th_order`
        dy_bins (numpy.ndarray): Offset bin edges along y-axis, as returned by :func:`extract_4th_order`
        dz_bins (numpy.ndarray): Offset bin edges along z-axis, as returned by :func:`extract_4th_order`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_4th_order`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_4th_order`
        model (connectome_manipulator.model_building.model_types.ConnProb4thOrderLinInterpnModel): Fitted stochastic 4th order connectivity model, as returned by :func:`extract_4th_order`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        plot_model_ovsampl (int): Oversampling factor w.r.t. data binning for plotting model output (must be >=1)
        plot_model_extsn (int): Range extension in multiples of original data bins in each direction for plotting model output (must be >=0)
    """
    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(
        isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1,
        "ERROR: Model plot oversampling must be an integer factor >= 1!",
    )
    log.log_assert(
        isinstance(plot_model_extsn, int) and plot_model_extsn >= 0,
        "ERROR: Model plot extension must be an integer number of bins >= 0!",
    )
    dx_bin_size_model = np.diff(dx_bins[:2])[0] / plot_model_ovsampl
    dy_bin_size_model = np.diff(dy_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dx_bins_model = np.arange(
        dx_bins[0] - plot_model_extsn * dx_bin_size_model * plot_model_ovsampl,
        dx_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dx_bin_size_model,
        dx_bin_size_model,
    )
    dy_bins_model = np.arange(
        dy_bins[0] - plot_model_extsn * dy_bin_size_model * plot_model_ovsampl,
        dy_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dy_bin_size_model,
        dy_bin_size_model,
    )
    dz_bins_model = np.arange(
        dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl,
        dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model,
        dz_bin_size_model,
    )

    # Sample positions (at bin centers)
    dx_pos_model = dx_bins_model[:-1] + 0.5 * dx_bin_size_model
    dy_pos_model = dy_bins_model[:-1] + 0.5 * dy_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    dxv, dyv, dzv = np.meshgrid(dx_pos_model, dy_pos_model, dz_pos_model, indexing="ij")
    model_pos = np.array([dxv.flatten(), dyv.flatten(), dzv.flatten()]).T  # Regular grid
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2])
    model_val_xyz = model_val.reshape([len(dx_pos_model), len(dy_pos_model), len(dz_pos_model)])

    # 3D connection probability (data vs. model)
    num_p_bins = 100
    p_bins = np.linspace(0, max(np.max(p_conn_offset), np.max(model_val)), num_p_bins + 1)
    p_color_map = plt.cm.ScalarMappable(
        cmap=JET, norm=plt.Normalize(vmin=p_bins[0], vmax=p_bins[-1])
    )
    p_colors = p_color_map.to_rgba(np.linspace(p_bins[0], p_bins[-1], num_p_bins))

    fig = plt.figure(figsize=(16, 6), dpi=300)
    # (Data)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for pidx in range(num_p_bins):
        p_sel_idx = np.where(
            np.logical_and(p_conn_offset > p_bins[pidx], p_conn_offset <= p_bins[pidx + 1])
        )
        plt.plot(
            dx_bins[p_sel_idx[0]] + dx_bin_offset,
            dy_bins[p_sel_idx[1]] + dy_bin_offset,
            dz_bins[p_sel_idx[2]] + dz_bin_offset,
            "o",
            color=p_colors[pidx, :],
            alpha=0.01 + 0.99 * pidx / (num_p_bins - 1),
            markeredgecolor="none",
        )
    #     ax.view_init(30, 60)
    ax.set_xlim((dx_bins[0], dx_bins[-1]))
    ax.set_ylim((dy_bins[0], dy_bins[-1]))
    ax.set_zlim((dz_bins[0], dz_bins[-1]))
    ax.set_xlabel("$\\Delta$x [$\\mu$m]")
    ax.set_ylabel("$\\Delta$y [$\\mu$m]")
    ax.set_zlabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(p_color_map, label="Conn. prob.")
    plt.title(f"Data: N = {src_cell_count}x{tgt_cell_count} cells")

    # (Model)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    for pidx in range(num_p_bins):
        p_sel_idx = np.logical_and(model_val > p_bins[pidx], model_val <= p_bins[pidx + 1])
        plt.plot(
            model_pos[p_sel_idx, 0],
            model_pos[p_sel_idx, 1],
            model_pos[p_sel_idx, 2],
            ".",
            color=p_colors[pidx, :],
            alpha=0.01 + 0.99 * pidx / (num_p_bins - 1),
            markeredgecolor="none",
        )
    #     ax.view_init(30, 60)
    ax.set_xlim((dx_bins[0], dx_bins[-1]))
    ax.set_ylim((dy_bins[0], dy_bins[-1]))
    ax.set_zlim((dz_bins[0], dz_bins[-1]))
    ax.set_xlabel("$\\Delta$x [$\\mu$m]")
    ax.set_ylabel("$\\Delta$y [$\\mu$m]")
    ax.set_zlabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(p_color_map, label="Conn. prob.")
    plt.title(f"Model: {model.__class__.__name__}")

    plt.suptitle(
        f"Offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model_3d.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Max. intensity projection (data vs. model)
    plt.figure(figsize=(12, 6), dpi=300)
    # (Data)
    plt.subplot(2, 3, 1)
    plt.imshow(
        np.max(p_conn_offset, 1).T,
        interpolation="none",
        extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$x [$\\mu$m]")
    plt.ylabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")

    plt.subplot(2, 3, 2)
    plt.imshow(
        np.max(p_conn_offset, 0).T,
        interpolation="none",
        extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$y [$\\mu$m]")
    plt.ylabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")
    plt.title("Data")

    plt.subplot(2, 3, 3)
    plt.imshow(
        np.max(p_conn_offset, 2).T,
        interpolation="none",
        extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$x [$\\mu$m]")
    plt.ylabel("$\\Delta$y [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")

    # (Model)
    plt.subplot(2, 3, 4)
    plt.imshow(
        np.max(model_val_xyz, 1).T,
        interpolation="none",
        extent=(dx_bins_model[0], dx_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$x [$\\mu$m]")
    plt.ylabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")

    plt.subplot(2, 3, 5)
    plt.imshow(
        np.max(model_val_xyz, 0).T,
        interpolation="none",
        extent=(dy_bins_model[0], dy_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$y [$\\mu$m]")
    plt.ylabel("$\\Delta$z [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")
    plt.title("Model")

    plt.subplot(2, 3, 6)
    plt.imshow(
        np.max(model_val_xyz, 2).T,
        interpolation="none",
        extent=(dx_bins_model[0], dx_bins_model[-1], dy_bins_model[-1], dy_bins_model[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("$\\Delta$x [$\\mu$m]")
    plt.ylabel("$\\Delta$y [$\\mu$m]")
    plt.colorbar(label="Max. conn. prob.")

    plt.suptitle(
        f"Offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model_2d.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity:
#   Reduced 4th order (offset-dependent), modified from [Gal et al. 2020]
#     => Radial/axial offsets only
#     => Position mapping model (flatmap) supported
#     => model_specs with 'type' (such as 'LinearInterpolation')
#                    and optionally, 'kwargs' may be provided
###################################################################################################


def extract_4th_order_reduced(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    bin_size_um=100,
    max_range_um=None,
    pos_map_file=None,
    min_count_per_bin=10,
    axial_coord=2,
    **_,
):
    """Extracts the binned, offset-dependent connection probability (reduced 4th order) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        bin_size_um (float/list-like): Offset bin size in um; can be scalar (same value for radial/axial dimension) or list-like with two individual values for radial/axial dimensions
        max_range_um (float/list-like): Maximum offset range in um; can be scalar (same +/- value for all dimensions) or list-like with two elements for radial/axial dimensions each of which can be either a scalar (same +/- ranges) or a two-element list with individual +/- ranges; in any case, the lower radial offset range must always be zero
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin
        axial_coord (int): Index to select axial coordinate (0..x, 1..y, 2..z), usually perpendicular to layers

    Returns:
        dict: Dictionary containing the extracted 4th-order (reduced) connection probability data
    """
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute dr/dz offset matrices
    dr_mat, dz_mat = model_types.ConnProb4thOrderLinInterpnReducedModel.compute_offset_matrices(
        src_nrn_pos, tgt_nrn_pos, axial_coord
    )

    # Extract offset-dependent connection probabilities
    if max_range_um is None:
        dr_range, dz_range = zip([0, np.nanmin(dz_mat)], [np.nanmax(dr_mat), np.nanmax(dz_mat)])
    else:
        dr_range, dz_range = get_value_ranges(max_range_um, 2, pos_range=[True, False])

    if np.isscalar(bin_size_um):  # Single scalar range value to be used for all dimensions
        log.log_assert(bin_size_um > 0.0, "ERROR: Offset bin size must be larger than 0um!")
        bin_size_dr = bin_size_dz = bin_size_um
    else:  # Two values for r/z directions
        log.log_assert(len(bin_size_um) == 2, "ERROR: Offset bin sizes in r/z directions expected!")
        log.log_assert(
            np.all([b > 0.0 for b in bin_size_um]),
            "ERROR: Offset bin size must be larger than 0um!",
        )
        bin_size_dr, bin_size_dz = bin_size_um

    num_bins_dr = np.ceil((dr_range[1] - dr_range[0]) / bin_size_dr).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dr_bins = np.arange(0, num_bins_dr + 1) * bin_size_dr + dr_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_offset, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids, tgt_node_ids, edges, [dr_mat, dz_mat], [dr_bins, dz_bins], min_count_per_bin
    )

    return {
        "p_conn_offset": p_conn_offset,
        "count_conn": count_conn,
        "count_all": count_all,
        "dr_bins": dr_bins,
        "dz_bins": dz_bins,
        "axial_coord_data": axial_coord,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_4th_order_reduced(
    p_conn_offset,
    dr_bins,
    dz_bins,
    count_all,
    axial_coord_data,
    model_specs=None,
    smoothing_sigma_um=None,
    **_,
):
    """Builds a stochastic 4th order reduced connection probability model (offset-dependent, based on linear interpolation).

    Args:
        p_conn_offset (numpy.ndarray): Binned offset-dependent connection probabilities, as retuned by :func:`extract_4th_order_reduced`
        dr_bins (numpy.ndarray): Offset bin edges along radial axis, as returned by :func:`extract_4th_order_reduced`
        dz_bins (numpy.ndarray): Offset bin edges along axial axis, as returned by :func:`extract_4th_order_reduced`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_4th_order_reduced`
        axial_coord_data (int):  Index of axial coordinate axis, as returned by :func:`extract_4th_order_reduced`
        model_specs (dict): Model specifications; see Notes
        smoothing_sigma_um (float/list-like): Sigma in um for Gaussian smoothing; can be scalar (same value for radial/axial dimension) or list-like with two individual values for radial/axial dimensions

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb4thOrderLinInterpnReducedModel: Resulting stochastic 4th order reduced connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; only "LinearInterpolation" supported which does not require any additional specs
    """
    if model_specs is None:
        model_specs = {"type": "LinearInterpolation"}

    bin_sizes = [np.diff(dr_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    dr_bin_offset = 0.5 * bin_sizes[0]
    dz_bin_offset = 0.5 * bin_sizes[1]

    dr_pos = dr_bins[:-1] + dr_bin_offset  # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset  # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_offset = p_conn_offset.copy()
    p_conn_offset[np.isnan(p_conn_offset)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 2  # Same value for all coordinates
        else:
            log.log_assert(
                len(smoothing_sigma_um) == 2, "ERROR: Smoothing sigma for 2 dimensions required!"
            )
        log.log_assert(
            np.all(np.array(smoothing_sigma_um) >= 0.0),
            "ERROR: Smoothing sigma must be non-negative!",
        )
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.debug(
            f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms'
        )
        p_reflect = np.vstack(
            [p_conn_offset[::-1, :], p_conn_offset]
        )  # Mirror along radial axis at dr==0, to avoid edge effect
        p_reflect = gaussian_filter(p_reflect, sigmas, mode="constant")
        p_conn_offset = p_reflect[p_conn_offset.shape[0] :, :]  # Cut original part of the data

    model_inputs = ["dr", "dz"]  # Must be the same for all interpolation types!
    if (
        model_specs.get("type") == "LinearInterpolation"
    ):  # Linear interpolation model => Removing dimensions with only single value from interpolation
        log.log_assert(
            len(model_specs.get("kwargs", {})) == 0,
            f'ERROR: No parameters expected for "{model_specs.get("type")}" model!',
        )

        # Create model
        index = pd.MultiIndex.from_product([dr_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_offset.flatten(), index=index, columns=["p"])
        model = model_types.ConnProb4thOrderLinInterpnReducedModel(
            p_conn_table=df, axial_coord=int(axial_coord_data)
        )

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("type")}" unknown!')

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_offset * count_all).astype(int)
    drv, dzv = np.meshgrid(dr_pos, dz_pos, indexing="ij")
    p_conn_model = model.get_conn_prob(dr=drv, dz=dzv)
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_4th_order_reduced(
    out_dir,
    p_conn_offset,
    dr_bins,
    dz_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    axial_coord_data,
    pos_map_file=None,
    plot_model_ovsampl=3,
    plot_model_extsn=0,
    **_,
):  # pragma: no cover
    """Visualizes 4th order reduced extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_offset (numpy.ndarray): Binned offset-dependent connection probabilities, as retuned by :func:`extract_4th_order_reduced`
        dr_bins (numpy.ndarray): Offset bin edges along radial axis, as returned by :func:`extract_4th_order_reduced`
        dz_bins (numpy.ndarray): Offset bin edges along axial axis, as returned by :func:`extract_4th_order_reduced`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_4th_order_reduced`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_4th_order_reduced`
        model (connectome_manipulator.model_building.model_types.ConnProb4thOrderLinInterpnReducedModel): Fitted stochastic 4th order reduced connectivity model, as returned by :func:`extract_4th_order_reduced`
        axial_coord_data (int):  Index of axial coordinate axis, as returned by :func:`extract_4th_order_reduced`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        plot_model_ovsampl (int): Oversampling factor w.r.t. data binning for plotting model output (must be >=1)
        plot_model_extsn (int): Range extension in multiples of original data bins in each direction for plotting model output (must be >=0)
    """
    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(
        isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1,
        "ERROR: Model plot oversampling must be an integer factor >= 1!",
    )
    log.log_assert(
        isinstance(plot_model_extsn, int) and plot_model_extsn >= 0,
        "ERROR: Model plot extension must be an integer number of bins >= 0!",
    )
    dr_bin_size_model = np.diff(dr_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dr_bins_model = np.arange(
        dr_bins[0],
        dr_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dr_bin_size_model,
        dr_bin_size_model,
    )
    dz_bins_model = np.arange(
        dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl,
        dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model,
        dz_bin_size_model,
    )

    # Sample positions (at bin centers)
    dr_pos_model = dr_bins_model[:-1] + 0.5 * dr_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    drv, dzv = np.meshgrid(dr_pos_model, dz_pos_model, indexing="ij")
    model_pos = np.array([drv.flatten(), dzv.flatten()]).T  # Regular grid
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1])
    model_val = model_val.reshape([len(dr_pos_model), len(dz_pos_model)])

    # Connection probability (data vs. model)
    coord_names = ["x", "y", "z"]
    az_name = coord_names[axial_coord_data]  # Axial axis name
    ar_name = "r"  # Radial axis name

    plt.figure(figsize=(12, 4), dpi=300)

    # (Data)
    log.log_assert(dr_bins[0] == 0, "ERROR: Radial bin range error!")
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.hstack([p_conn_offset.T[:, ::-1], p_conn_offset.T]),
        interpolation="nearest",
        extent=(-dr_bins[-1], dr_bins[-1], dz_bins[-1], dz_bins[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(np.zeros(2), plt.ylim(), color="lightgrey", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel(f"$\\Delta${ar_name} [$\\mu$m]")
    plt.ylabel(f"$\\Delta${az_name} [$\\mu$m]")
    plt.colorbar(label="Conn. prob.")
    plt.title(f"Data: N = {src_cell_count}x{tgt_cell_count} cells")

    # (Model)
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.hstack([model_val.T[:, ::-1], model_val.T]),
        interpolation="nearest",
        extent=(-dr_bins_model[-1], dr_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]),
        cmap=HOT,
        vmin=0.0,
    )
    plt.plot(np.zeros(2), plt.ylim(), color="lightgrey", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel(f"$\\Delta${ar_name} [$\\mu$m]")
    plt.ylabel(f"$\\Delta${az_name} [$\\mu$m]")
    plt.colorbar(label="Conn. prob.")
    plt.title(f"Model: {model.__class__.__name__}")

    plt.suptitle(
        f"Reduced offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   5th order (position-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'type' (such as 'LinearInterpolation')
#                    and optionally, 'kwargs' may be provided
###################################################################################################


def extract_5th_order(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    position_bin_size_um=1000,
    position_max_range_um=None,
    offset_bin_size_um=100,
    offset_max_range_um=None,
    pos_map_file=None,
    min_count_per_bin=10,
    **_,
):
    """Extracts the binned, position-dependent connection probability (5th order) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        position_bin_size_um (float/list-like): Position bin size in um; can be scalar (same value for x/y/z dimension) or list-like with three individual values for x/y/z dimensions
        position_max_range_um (float/list-like): Maximum position range in um; can be scalar (same +/- value for all dimensions) or list-like with three elements for x/y/z dimensions each of which can be either a scalar (same +/- ranges) or a two-element list with individual +/- ranges
        offset_bin_size_um (float/list-like): Offset bin size in um; can be scalar (same value for x/y/z dimension) or list-like with three individual values for x/y/z dimensions
        offset_max_range_um (float/list-like): Maximum offset range in um; can be scalar (same +/- value for all dimensions) or list-like with three elements for x/y/z dimensions each of which can be either a scalar (same +/- ranges) or a two-element list with individual +/- ranges
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin

    Returns:
        dict: Dictionary containing the extracted 5th-order connection probability data
    """
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute PRE position & POST-PRE offset matrices
    x_mat, y_mat, z_mat = model_types.ConnProb5thOrderLinInterpnModel.compute_position_matrices(
        src_nrn_pos, tgt_nrn_pos
    )
    dx_mat, dy_mat, dz_mat = model_types.ConnProb5thOrderLinInterpnModel.compute_offset_matrices(
        src_nrn_pos, tgt_nrn_pos
    )

    # Extract position- & offset-dependent connection probabilities
    if position_max_range_um is None:
        x_range, y_range, z_range = zip(
            np.minimum(np.nanmin(src_nrn_pos, 0), np.nanmin(tgt_nrn_pos, 0)),
            np.maximum(np.nanmax(src_nrn_pos, 0), np.nanmax(tgt_nrn_pos, 0)),
        )
    else:
        x_range, y_range, z_range = get_value_ranges(position_max_range_um, 3, pos_range=False)

    if np.isscalar(position_bin_size_um):  # Single scalar range value to be used for all dimensions
        log.log_assert(
            position_bin_size_um > 0.0, "ERROR: Position bin size must be larger than 0um!"
        )
        bin_size_x = bin_size_y = bin_size_z = position_bin_size_um
    else:  # Three values for x/y/z dimensions
        log.log_assert(
            len(position_bin_size_um) == 3, "ERROR: Position bin sizes in x/y/z dimension expected!"
        )
        log.log_assert(
            np.all([b > 0.0 for b in position_bin_size_um]),
            "ERROR: Position bin size must be larger than 0um!",
        )
        bin_size_x, bin_size_y, bin_size_z = position_bin_size_um

    num_bins_x = np.ceil((x_range[1] - x_range[0]) / bin_size_x).astype(int)
    num_bins_y = np.ceil((y_range[1] - y_range[0]) / bin_size_y).astype(int)
    num_bins_z = np.ceil((z_range[1] - z_range[0]) / bin_size_z).astype(int)

    x_bins = np.arange(0, num_bins_x + 1) * bin_size_x + x_range[0]
    y_bins = np.arange(0, num_bins_y + 1) * bin_size_y + y_range[0]
    z_bins = np.arange(0, num_bins_z + 1) * bin_size_z + z_range[0]

    if offset_max_range_um is None:
        dx_range, dy_range, dz_range = zip(
            [np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)],
            [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)],
        )
    else:
        dx_range, dy_range, dz_range = get_value_ranges(offset_max_range_um, 3, pos_range=False)

    if np.isscalar(offset_bin_size_um):  # Single scalar range value to be used for all dimensions
        log.log_assert(offset_bin_size_um > 0.0, "ERROR: Offset bin size must be larger than 0um!")
        bin_size_dx = bin_size_dy = bin_size_dz = offset_bin_size_um
    else:  # Three values for x/y/z dimensions
        log.log_assert(
            len(offset_bin_size_um) == 3, "ERROR: Offset bin sizes in x/y/z dimension expected!"
        )
        log.log_assert(
            np.all([b > 0.0 for b in offset_bin_size_um]),
            "ERROR: Offset bin size must be larger than 0um!",
        )
        bin_size_dx, bin_size_dy, bin_size_dz = offset_bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_position, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids,
        tgt_node_ids,
        edges,
        [x_mat, y_mat, z_mat, dx_mat, dy_mat, dz_mat],
        [x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins],
        min_count_per_bin,
    )

    return {
        "p_conn_position": p_conn_position,
        "count_conn": count_conn,
        "count_all": count_all,
        "x_bins": x_bins,
        "y_bins": y_bins,
        "z_bins": z_bins,
        "dx_bins": dx_bins,
        "dy_bins": dy_bins,
        "dz_bins": dz_bins,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_5th_order(
    p_conn_position,
    x_bins,
    y_bins,
    z_bins,
    dx_bins,
    dy_bins,
    dz_bins,
    count_all,
    model_specs=None,
    smoothing_sigma_um=None,
    **_,
):
    """Builds a stochastic 5th order connection probability model (position-dependent, based on linear interpolation).

    Args:
        p_conn_position (numpy.ndarray): Binned position- and offset-dependent connection probabilities, as retuned by :func:`extract_5th_order`
        x_bins (numpy.ndarray): Position bin edges along x-axis, as returned by :func:`extract_5th_order`
        y_bins (numpy.ndarray): Position bin edges along y-axis, as returned by :func:`extract_5th_order`
        z_bins (numpy.ndarray): Position bin edges along z-axis, as returned by :func:`extract_5th_order`
        dx_bins (numpy.ndarray): Offset bin edges along x-axis, as returned by :func:`extract_5th_order`
        dy_bins (numpy.ndarray): Offset bin edges along y-axis, as returned by :func:`extract_5th_order`
        dz_bins (numpy.ndarray): Offset bin edges along z-axis, as returned by :func:`extract_5th_order`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_5th_order`
        model_specs (dict): Model specifications; see Notes
        smoothing_sigma_um (float/list-like): Sigma in um for Gaussian smoothing; can be scalar (same value for x/y/z/dx/dy/dz dimension) or list-like with six individual values for x/y/z/dx/dy/dz dimensions

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb5thOrderLinInterpnModel: Resulting stochastic 5th order connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; only "LinearInterpolation" supported which does not require any additional specs
    """
    if model_specs is None:
        model_specs = {"type": "LinearInterpolation"}

    bin_sizes = [
        np.diff(x_bins[:2])[0],
        np.diff(y_bins[:2])[0],
        np.diff(z_bins[:2])[0],
        np.diff(dx_bins[:2])[0],
        np.diff(dy_bins[:2])[0],
        np.diff(dz_bins[:2])[0],
    ]

    x_bin_offset = 0.5 * bin_sizes[0]
    y_bin_offset = 0.5 * bin_sizes[1]
    z_bin_offset = 0.5 * bin_sizes[2]

    x_pos = x_bins[:-1] + x_bin_offset  # Positions at bin centers
    y_pos = y_bins[:-1] + y_bin_offset  # Positions at bin centers
    z_pos = z_bins[:-1] + z_bin_offset  # Positions at bin centers

    dx_bin_offset = 0.5 * bin_sizes[3]
    dy_bin_offset = 0.5 * bin_sizes[4]
    dz_bin_offset = 0.5 * bin_sizes[5]

    dx_pos = dx_bins[:-1] + dx_bin_offset  # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset  # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset  # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_position = p_conn_position.copy()
    p_conn_position[np.isnan(p_conn_position)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 6  # Same value for all coordinates
        else:
            log.log_assert(
                len(smoothing_sigma_um) == 6, "ERROR: Smoothing sigma for 6 dimensions required!"
            )
        log.log_assert(
            np.all(np.array(smoothing_sigma_um) >= 0.0),
            "ERROR: Smoothing sigma must be non-negative!",
        )
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.debug(
            f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms'
        )
        p_conn_position = gaussian_filter(p_conn_position, sigmas, mode="constant")

    model_inputs = [
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
    ]  # Must be the same for all interpolation types!
    if (
        model_specs.get("type") == "LinearInterpolation"
    ):  # Linear interpolation model => Removing dimensions with only single value from interpolation
        log.log_assert(
            len(model_specs.get("kwargs", {})) == 0,
            f'ERROR: No parameters expected for "{model_specs.get("type")}" model!',
        )

        # Create model
        index = pd.MultiIndex.from_product(
            [x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos], names=model_inputs
        )
        df = pd.DataFrame(p_conn_position.flatten(), index=index, columns=["p"])
        model = model_types.ConnProb5thOrderLinInterpnModel(p_conn_table=df)

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("type")}" unknown!')

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_position * count_all).astype(int)
    xv, yv, zv, dxv, dyv, dzv = np.meshgrid(
        x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos, indexing="ij"
    )
    p_conn_model = model.get_conn_prob(x=xv, y=yv, z=zv, dx=dxv, dy=dyv, dz=dzv)
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_5th_order(
    out_dir,
    p_conn_position,
    x_bins,
    y_bins,
    z_bins,
    dx_bins,
    dy_bins,
    dz_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    pos_map_file=None,
    plot_model_ovsampl=3,
    plot_model_extsn=0,
    **_,
):  # pragma: no cover
    """Visualizes 5th order extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_position (numpy.ndarray): Binned position- and offset-dependent connection probabilities, as retuned by :func:`extract_5th_order`
        x_bins (numpy.ndarray): Position bin edges along x-axis, as returned by :func:`extract_5th_order`
        y_bins (numpy.ndarray): Position bin edges along y-axis, as returned by :func:`extract_5th_order`
        z_bins (numpy.ndarray): Position bin edges along z-axis, as returned by :func:`extract_5th_order`
        dx_bins (numpy.ndarray): Offset bin edges along x-axis, as returned by :func:`extract_5th_order`
        dy_bins (numpy.ndarray): Offset bin edges along y-axis, as returned by :func:`extract_5th_order`
        dz_bins (numpy.ndarray): Offset bin edges along z-axis, as returned by :func:`extract_5th_order`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_5th_order`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_5th_order`
        model (connectome_manipulator.model_building.model_types.ConnProb5thOrderLinInterpnModel): Fitted stochastic 5th order connectivity model, as returned by :func:`extract_5th_order`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        plot_model_ovsampl (int): Oversampling factor w.r.t. data binning for plotting model output (must be >=1)
        plot_model_extsn (int): Range extension in multiples of original data bins in each direction for plotting model output (must be >=0)
    """
    x_bin_offset = 0.5 * np.diff(x_bins[:2])[0]
    y_bin_offset = 0.5 * np.diff(y_bins[:2])[0]
    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]

    x_pos_model = x_bins[:-1] + x_bin_offset  # Positions at bin centers
    y_pos_model = y_bins[:-1] + y_bin_offset  # Positions at bin centers
    z_pos_model = z_bins[:-1] + z_bin_offset  # Positions at bin centers

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(
        isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1,
        "ERROR: Model plot oversampling must be an integer factor >= 1!",
    )
    log.log_assert(
        isinstance(plot_model_extsn, int) and plot_model_extsn >= 0,
        "ERROR: Model plot extension must be an integer number of bins >= 0!",
    )
    dx_bin_size_model = np.diff(dx_bins[:2])[0] / plot_model_ovsampl
    dy_bin_size_model = np.diff(dy_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dx_bins_model = np.arange(
        dx_bins[0] - plot_model_extsn * dx_bin_size_model * plot_model_ovsampl,
        dx_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dx_bin_size_model,
        dx_bin_size_model,
    )
    dy_bins_model = np.arange(
        dy_bins[0] - plot_model_extsn * dy_bin_size_model * plot_model_ovsampl,
        dy_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dy_bin_size_model,
        dy_bin_size_model,
    )
    dz_bins_model = np.arange(
        dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl,
        dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model,
        dz_bin_size_model,
    )

    # Sample positions (at bin centers)
    dx_pos_model = dx_bins_model[:-1] + 0.5 * dx_bin_size_model
    dy_pos_model = dy_bins_model[:-1] + 0.5 * dy_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    xv, yv, zv, dxv, dyv, dzv = np.meshgrid(
        x_pos_model,
        y_pos_model,
        z_pos_model,
        dx_pos_model,
        dy_pos_model,
        dz_pos_model,
        indexing="ij",
    )
    model_pos = np.array(
        [xv.flatten(), yv.flatten(), zv.flatten(), dxv.flatten(), dyv.flatten(), dzv.flatten()]
    ).T  # Regular grid
    model_val = model.get_conn_prob(
        model_pos[:, 0],
        model_pos[:, 1],
        model_pos[:, 2],
        model_pos[:, 3],
        model_pos[:, 4],
        model_pos[:, 5],
    )
    model_val_xyz = model_val.reshape(
        [
            len(x_pos_model),
            len(y_pos_model),
            len(z_pos_model),
            len(dx_pos_model),
            len(dy_pos_model),
            len(dz_pos_model),
        ]
    )

    # 3D connection probability (data vs. model)
    num_p_bins = 100
    p_bins = np.linspace(0, max(np.max(p_conn_position), np.max(model_val)), num_p_bins + 1)
    p_color_map = plt.cm.ScalarMappable(
        cmap=JET, norm=plt.Normalize(vmin=p_bins[0], vmax=p_bins[-1])
    )
    p_colors = p_color_map.to_rgba(np.linspace(p_bins[0], p_bins[-1], num_p_bins))

    for ix, xpm in enumerate(x_pos_model):
        for iy, ypm in enumerate(y_pos_model):
            for iz, zpm in enumerate(z_pos_model):
                p_conn_sel = p_conn_position[ix, iy, iz, :, :, :]
                model_val_sel = model_val_xyz[ix, iy, iz, :, :, :]

                fig = plt.figure(figsize=(16, 6), dpi=300)
                # (Data)
                ax = fig.add_subplot(1, 2, 1, projection="3d")
                for pidx in range(num_p_bins):
                    p_sel_idx = np.where(
                        np.logical_and(p_conn_sel > p_bins[pidx], p_conn_sel <= p_bins[pidx + 1])
                    )
                    plt.plot(
                        dx_bins[p_sel_idx[0]] + dx_bin_offset,
                        dy_bins[p_sel_idx[1]] + dy_bin_offset,
                        dz_bins[p_sel_idx[2]] + dz_bin_offset,
                        "o",
                        color=p_colors[pidx, :],
                        alpha=0.01 + 0.99 * pidx / (num_p_bins - 1),
                        markeredgecolor="none",
                    )
                ax.view_init(30, 60)
                ax.set_xlim((dx_bins[0], dx_bins[-1]))
                ax.set_ylim((dy_bins[0], dy_bins[-1]))
                ax.set_zlim((dz_bins[0], dz_bins[-1]))
                ax.set_xlabel("$\\Delta$x [$\\mu$m]")
                ax.set_ylabel("$\\Delta$y [$\\mu$m]")
                ax.set_zlabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(p_color_map, label="Conn. prob.")
                plt.title(f"Data: N = {src_cell_count}x{tgt_cell_count} cells")

                # (Model)
                ax = fig.add_subplot(1, 2, 2, projection="3d")
                for pidx in range(num_p_bins):
                    p_sel_idx = np.where(
                        np.logical_and(
                            model_val_sel > p_bins[pidx], model_val_sel <= p_bins[pidx + 1]
                        )
                    )
                    plt.plot(
                        dx_pos_model[p_sel_idx[0]].T,
                        dy_pos_model[p_sel_idx[1]].T,
                        dz_pos_model[p_sel_idx[2]].T,
                        ".",
                        color=p_colors[pidx, :],
                        alpha=0.01 + 0.99 * pidx / (num_p_bins - 1),
                        markeredgecolor="none",
                    )
                ax.view_init(30, 60)
                ax.set_xlim((dx_bins[0], dx_bins[-1]))
                ax.set_ylim((dy_bins[0], dy_bins[-1]))
                ax.set_zlim((dz_bins[0], dz_bins[-1]))
                ax.set_xlabel("$\\Delta$x [$\\mu$m]")
                ax.set_ylabel("$\\Delta$y [$\\mu$m]")
                ax.set_zlabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(p_color_map, label="Conn. prob.")
                plt.title(f"Model: {model.__class__.__name__}")

                plt.suptitle(
                    f"Position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>\nX={xpm:.0f}$\\mu$m, Y={ypm:.0f}$\\mu$m, Z={zpm:.0f}$\\mu$m"
                )
                plt.tight_layout()
                out_fn = os.path.abspath(
                    os.path.join(out_dir, f"data_vs_model_3d_x{ix}y{iy}z{iz}.png")
                )
                log.info(f"Saving {out_fn}...")
                plt.savefig(out_fn)

    # Max. intensity projection (data vs. model)
    for ix, xpm in enumerate(x_pos_model):
        for iy, ypm in enumerate(y_pos_model):
            for iz, zpm in enumerate(z_pos_model):
                p_conn_sel = p_conn_position[ix, iy, iz, :, :, :]
                model_val_sel = model_val_xyz[ix, iy, iz, :, :, :]

                plt.figure(figsize=(12, 6), dpi=300)
                # (Data)
                plt.subplot(2, 3, 1)
                plt.imshow(
                    np.max(p_conn_sel, 1).T,
                    interpolation="none",
                    extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(p_conn_sel, 1)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$x [$\\mu$m]")
                plt.ylabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")

                plt.subplot(2, 3, 2)
                plt.imshow(
                    np.max(p_conn_sel, 0).T,
                    interpolation="none",
                    extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(p_conn_sel, 0)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$y [$\\mu$m]")
                plt.ylabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")
                plt.title("Data")

                plt.subplot(2, 3, 3)
                plt.imshow(
                    np.max(p_conn_sel, 2).T,
                    interpolation="none",
                    extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(p_conn_sel, 2)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$x [$\\mu$m]")
                plt.ylabel("$\\Delta$y [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")

                # (Model)
                plt.subplot(2, 3, 4)
                plt.imshow(
                    np.max(model_val_sel, 1).T,
                    interpolation="none",
                    extent=(
                        dx_bins_model[0],
                        dx_bins_model[-1],
                        dz_bins_model[-1],
                        dz_bins_model[0],
                    ),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(model_val_sel, 1)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$x [$\\mu$m]")
                plt.ylabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")

                plt.subplot(2, 3, 5)
                plt.imshow(
                    np.max(model_val_sel, 0).T,
                    interpolation="none",
                    extent=(
                        dy_bins_model[0],
                        dy_bins_model[-1],
                        dz_bins_model[-1],
                        dz_bins_model[0],
                    ),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(model_val_sel, 0)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$y [$\\mu$m]")
                plt.ylabel("$\\Delta$z [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")
                plt.title("Model")

                plt.subplot(2, 3, 6)
                plt.imshow(
                    np.max(model_val_sel, 2).T,
                    interpolation="none",
                    extent=(
                        dx_bins_model[0],
                        dx_bins_model[-1],
                        dy_bins_model[-1],
                        dy_bins_model[0],
                    ),
                    cmap=HOT,
                    vmin=0.0,
                    vmax=0.1 if np.max(np.max(model_val_sel, 2)) == 0.0 else None,
                )
                plt.plot(plt.xlim(), np.zeros(2), "w", linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), "w", linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel("$\\Delta$x [$\\mu$m]")
                plt.ylabel("$\\Delta$y [$\\mu$m]")
                plt.colorbar(label="Max. conn. prob.")

                plt.suptitle(
                    f"Position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>\nX={xpm:.0f}$\\mu$m, Y={ypm:.0f}$\\mu$m, Z={zpm:.0f}$\\mu$m"
                )
                plt.tight_layout()
                out_fn = os.path.abspath(
                    os.path.join(out_dir, f"data_vs_model_2d_x{ix}y{iy}z{iz}.png")
                )
                log.info(f"Saving {out_fn}...")
                plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity:
#   Reduced 5th order (position-dependent), modified from [Gal et al. 2020]
#     => Axial position only
#     => Radial/axial offsets only
#     => Position mapping model (flatmap) supported
#     => model_specs with 'type' (such as 'LinearInterpolation')
#                    and optionally, 'kwargs' may be provided
###################################################################################################


def extract_5th_order_reduced(
    nodes,
    edges,
    src_node_ids,
    tgt_node_ids,
    position_bin_size_um=1000,
    position_max_range_um=None,
    offset_bin_size_um=100,
    offset_max_range_um=None,
    pos_map_file=None,
    min_count_per_bin=10,
    axial_coord=2,
    **_,
):
    """Extracts the binned, position-dependent connection probability (5th order reduced) from a sample of pairs of neurons.

    Args:
        nodes (list): Two-element list containing source and target neuron populations of type bluepysnap.nodes.Nodes
        edges (bluepysnap.edges.Edges): SONATA egdes population to extract connection probabilities from
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        position_bin_size_um (float): Axial position bin size in um
        position_max_range_um (float/list-like): Maximum axial position range in um; can be scalar (same +/- value) or list-like with two elements for individual +/- ranges
        offset_bin_size_um (float/list-like): Offset bin size in um; can be scalar (same value for radial/axial dimension) or list-like with two individual values for radial/axial dimensions
        offset_max_range_um (float/list-like): Maximum offset range in um; can be scalar (same +/- value for all dimensions) or list-like with two elements for radial/axial dimensions each of which can be either a scalar (same +/- ranges) or a two-element list with individual +/- ranges; in any case, the lower radial offset range must always be zero
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        min_count_per_bin (int): Minimum number of samples per bin; otherwise, no estimate will be made for a given bin
        axial_coord (int): Index to select axial coordinate (0..x, 1..y, 2..z), usually perpendicular to layers

    Returns:
        dict: Dictionary containing the extracted 5th-order (reduced) connection probability data
    """
    # pylint: disable=W0613
    # Get source/target neuron positions (optionally: two types of mappings)
    pos_acc, vox_map = get_pos_mapping_fcts(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions(
        nodes, [src_node_ids, tgt_node_ids], pos_acc, vox_map
    )

    # Compute PRE position & POST-PRE offset matrices
    z_mat = model_types.ConnProb5thOrderLinInterpnReducedModel.compute_position_matrix(
        src_nrn_pos, tgt_nrn_pos, axial_coord
    )
    dr_mat, dz_mat = model_types.ConnProb5thOrderLinInterpnReducedModel.compute_offset_matrices(
        src_nrn_pos, tgt_nrn_pos, axial_coord
    )

    # Extract position- & offset-dependent connection probabilities
    if position_max_range_um is None:
        z_range = [
            np.minimum(
                np.nanmin(src_nrn_pos[:, axial_coord]), np.nanmin(tgt_nrn_pos[:, axial_coord])
            ),
            np.maximum(
                np.nanmax(src_nrn_pos[:, axial_coord]), np.nanmax(tgt_nrn_pos[:, axial_coord])
            ),
        ]
    else:
        z_range = get_value_ranges(position_max_range_um, 1, pos_range=False)

    log.log_assert(
        np.isscalar(position_bin_size_um) and position_bin_size_um > 0.0,
        "ERROR: Position bin size must be a scalar larger than 0um!",
    )
    bin_size_z = position_bin_size_um
    num_bins_z = np.ceil((z_range[1] - z_range[0]) / bin_size_z).astype(int)
    z_bins = np.arange(0, num_bins_z + 1) * bin_size_z + z_range[0]

    if offset_max_range_um is None:
        dr_range, dz_range = zip([0, np.nanmin(dz_mat)], [np.nanmax(dr_mat), np.nanmax(dz_mat)])
    else:
        dr_range, dz_range = get_value_ranges(offset_max_range_um, 2, pos_range=[True, False])

    if np.isscalar(offset_bin_size_um):  # Single scalar range value to be used for all dimensions
        log.log_assert(offset_bin_size_um > 0.0, "ERROR: Offset bin size must be larger than 0um!")
        bin_size_dr = bin_size_dz = offset_bin_size_um
    else:  # Two values for r/z dimensions
        log.log_assert(
            len(offset_bin_size_um) == 2, "ERROR: Offset bin sizes in r/z directions expected!"
        )
        log.log_assert(
            np.all([b > 0.0 for b in offset_bin_size_um]),
            "ERROR: Offset bin size must be larger than 0um!",
        )
        bin_size_dr, bin_size_dz = offset_bin_size_um

    num_bins_dr = np.ceil((dr_range[1] - dr_range[0]) / bin_size_dr).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dr_bins = np.arange(0, num_bins_dr + 1) * bin_size_dr + dr_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_position, count_conn, count_all = extract_dependent_p_conn(
        src_node_ids,
        tgt_node_ids,
        edges,
        [z_mat, dr_mat, dz_mat],
        [z_bins, dr_bins, dz_bins],
        min_count_per_bin,
    )

    return {
        "p_conn_position": p_conn_position,
        "count_conn": count_conn,
        "count_all": count_all,
        "z_bins": z_bins,
        "dr_bins": dr_bins,
        "dz_bins": dz_bins,
        "axial_coord_data": axial_coord,
        "src_cell_count": len(src_node_ids),
        "tgt_cell_count": len(tgt_node_ids),
    }


def build_5th_order_reduced(
    p_conn_position,
    z_bins,
    dr_bins,
    dz_bins,
    count_all,
    axial_coord_data,
    model_specs=None,
    smoothing_sigma_um=None,
    **_,
):
    """Builds a stochastic 5th order reduced connection probability model (position-dependent, based on linear interpolation).

    Args:
        p_conn_position (numpy.ndarray): Binned position- and offset-dependent connection probabilities, as retuned by :func:`extract_5th_order_reduced`
        z_bins (numpy.ndarray): Position bin edges along axial axis, as returned by :func:`extract_5th_order_reduced`
        dr_bins (numpy.ndarray): Offset bin edges along radial axis, as returned by :func:`extract_5th_order_reduced`
        dz_bins (numpy.ndarray): Offset bin edges along axial axis, as returned by :func:`extract_5th_order_reduced`
        count_all (numpy.ndarray): Count of all pairs of neurons (i.e., all possible connections) in each bin, as retuned by :func:`extract_5th_order_reduced`
        axial_coord_data (int):  Index of axial coordinate axis, as returned by :func:`extract_5th_order_reduced`
        model_specs (dict): Model specifications; see Notes
        smoothing_sigma_um (float/list-like): Sigma in um for Gaussian smoothing; can be scalar (same value for z/dr/dz dimension) or list-like with three individual values for z/dr/dz dimensions

    Returns:
        connectome_manipulator.model_building.model_types.ConnProb5thOrderLinInterpnReducedModel: Resulting stochastic 5th order reduced connectivity model

    Note:
        Info on possible keys contained in `model_specs` dict:

        * type (str): Type of the fitted model; only "LinearInterpolation" supported which does not require any additional specs
    """
    if model_specs is None:
        model_specs = {"type": "LinearInterpolation"}

    bin_sizes = [np.diff(z_bins[:2])[0], np.diff(dr_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    z_bin_offset = 0.5 * bin_sizes[0]
    z_pos = z_bins[:-1] + z_bin_offset  # Positions at bin centers

    dr_bin_offset = 0.5 * bin_sizes[1]
    dz_bin_offset = 0.5 * bin_sizes[2]

    dr_pos = dr_bins[:-1] + dr_bin_offset  # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset  # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_position = p_conn_position.copy()
    p_conn_position[np.isnan(p_conn_position)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 3  # Same value for all coordinates
        else:
            log.log_assert(
                len(smoothing_sigma_um) == 3, "ERROR: Smoothing sigma for 3 dimensions required!"
            )
        log.log_assert(
            np.all(np.array(smoothing_sigma_um) >= 0.0),
            "ERROR: Smoothing sigma must be non-negative!",
        )
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.debug(
            f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms'
        )
        p_reflect = np.concatenate(
            [p_conn_position[:, ::-1, :], p_conn_position], axis=1
        )  # Mirror along radial axis at dr==0, to avoid edge effect
        p_reflect = gaussian_filter(p_reflect, sigmas, mode="constant")
        p_conn_position = p_reflect[
            :, p_conn_position.shape[1] :, :
        ]  # Cut original part of the data

    model_inputs = ["z", "dr", "dz"]  # Must be the same for all interpolation types!
    if (
        model_specs.get("type") == "LinearInterpolation"
    ):  # Linear interpolation model => Removing dimensions with only single value from interpolation
        log.log_assert(
            len(model_specs.get("kwargs", {})) == 0,
            f'ERROR: No parameters expected for "{model_specs.get("type")}" model!',
        )

        # Create model
        index = pd.MultiIndex.from_product([z_pos, dr_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_position.flatten(), index=index, columns=["p"])
        model = model_types.ConnProb5thOrderLinInterpnReducedModel(
            p_conn_table=df, axial_coord=int(axial_coord_data)
        )

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("type")}" unknown!')

    log.debug("Model description:\n%s", model)  # pylint: disable=E0606

    # Check model prediction of total number of connections
    conn_count_data = np.nansum(p_conn_position * count_all).astype(int)
    zv, drv, dzv = np.meshgrid(z_pos, dr_pos, dz_pos, indexing="ij")
    p_conn_model = model.get_conn_prob(z=zv, dr=drv, dz=dzv)
    conn_count_model = np.nansum(p_conn_model * count_all).astype(int)
    log.info(
        f"Model prediction of total number of connections: {conn_count_model} (model) vs. {conn_count_data} (data); DIFF {conn_count_model - conn_count_data} ({100.0 * (conn_count_model - conn_count_data) / conn_count_data:.2f}%)"
    )

    return model


def plot_5th_order_reduced(
    out_dir,
    p_conn_position,
    z_bins,
    dr_bins,
    dz_bins,
    src_cell_count,
    tgt_cell_count,
    model,
    axial_coord_data,
    pos_map_file=None,
    plot_model_ovsampl=4,
    plot_model_extsn=0,
    **_,
):  # pragma: no cover
    """Visualizes 5th order reduced extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        p_conn_position (numpy.ndarray): Binned position- and offset-dependent connection probabilities, as retuned by :func:`extract_5th_order_reduced`
        z_bins (numpy.ndarray): Position bin edges along axial axis, as returned by :func:`extract_5th_order_reduced`
        dr_bins (numpy.ndarray): Offset bin edges along radial axis, as returned by :func:`extract_5th_order_reduced`
        dz_bins (numpy.ndarray): Offset bin edges along axial axis, as returned by :func:`extract_5th_order_reduced`
        src_cell_count (int): Number of source (pre-synaptic) neurons, as returned by :func:`extract_5th_order_reduced`
        tgt_cell_count (int): Number or target (post-synaptic) neurons, as returned by :func:`extract_5th_order_reduced`
        model (connectome_manipulator.model_building.model_types.ConnProb5thOrderLinInterpnReducedModel): Fitted stochastic 5th order reduced connectivity model, as returned by :func:`extract_5th_order_reduced`
        axial_coord_data (int):  Index of axial coordinate axis, as returned by :func:`extract_5th_order_reduced`
        pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
        plot_model_ovsampl (int): Oversampling factor w.r.t. data binning for plotting model output (must be >=1)
        plot_model_extsn (int): Range extension in multiples of original data bins in each direction for plotting model output (must be >=0)
    """
    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]
    z_pos_model = z_bins[:-1] + z_bin_offset  # Positions at bin centers

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(
        isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1,
        "ERROR: Model plot oversampling must be an integer factor >= 1!",
    )
    log.log_assert(
        isinstance(plot_model_extsn, int) and plot_model_extsn >= 0,
        "ERROR: Model plot extension must be an integer number of bins >= 0!",
    )
    dr_bin_size_model = np.diff(dr_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dr_bins_model = np.arange(
        dr_bins[0],
        dr_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dr_bin_size_model,
        dr_bin_size_model,
    )
    dz_bins_model = np.arange(
        dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl,
        dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model,
        dz_bin_size_model,
    )

    # Sample positions (at bin centers)
    dr_pos_model = dr_bins_model[:-1] + 0.5 * dr_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    zv, drv, dzv = np.meshgrid(z_pos_model, dr_pos_model, dz_pos_model, indexing="ij")
    model_pos = np.array([zv.flatten(), drv.flatten(), dzv.flatten()]).T  # Regular grid
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2])
    model_val = model_val.reshape([len(z_pos_model), len(dr_pos_model), len(dz_pos_model)])

    # Connection probability (data vs. model)
    coord_names = ["x", "y", "z"]
    az_name = coord_names[axial_coord_data]  # Axial axis name
    ar_name = "r"  # Radial axis name

    p_max = np.nanmax(p_conn_position)
    p_max_model = np.nanmax(model_val)
    plt.figure(figsize=(12, 4 * len(z_pos_model)), dpi=300)
    for zidx, zval in enumerate(z_pos_model):
        # (Data)
        log.log_assert(dr_bins[0] == 0, "ERROR: Radial bin range error!")
        plt.subplot(len(z_pos_model), 2, zidx * 2 + 1)
        plt.imshow(
            np.hstack(
                [
                    np.squeeze(p_conn_position[zidx, ::-1, :]).T,
                    np.squeeze(p_conn_position[zidx, :, :]).T,
                ]
            ),
            interpolation="nearest",
            extent=(-dr_bins[-1], dr_bins[-1], dz_bins[-1], dz_bins[0]),
            cmap=HOT,
            vmin=0.0,
            vmax=0.1 if p_max == 0.0 else p_max,
        )
        plt.plot(np.zeros(2), plt.ylim(), color="lightgrey", linewidth=0.5)
        plt.text(
            np.min(plt.xlim()),
            np.max(plt.ylim()),
            f"{az_name}={zval}um",
            color="lightgrey",
            ha="left",
            va="top",
        )
        plt.gca().invert_yaxis()
        plt.xlabel(f"$\\Delta${ar_name} [$\\mu$m]")
        plt.ylabel(f"$\\Delta${az_name} [$\\mu$m]")
        plt.colorbar(label="Conn. prob.")
        if zidx == 0:
            plt.title(f"Data: N = {src_cell_count}x{tgt_cell_count} cells")

        # (Model)
        plt.subplot(len(z_pos_model), 2, zidx * 2 + 2)
        plt.imshow(
            np.hstack(
                [np.squeeze(model_val[zidx, ::-1, :]).T, np.squeeze(model_val[zidx, :, :]).T]
            ),
            interpolation="nearest",
            extent=(-dr_bins_model[-1], dr_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]),
            cmap=HOT,
            vmin=0.0,
            vmax=0.1 if p_max_model == 0.0 else p_max_model,
        )
        plt.plot(np.zeros(2), plt.ylim(), color="lightgrey", linewidth=0.5)
        plt.text(
            np.min(plt.xlim()),
            np.max(plt.ylim()),
            f"{az_name}={zval}um",
            color="lightgrey",
            ha="left",
            va="top",
        )
        plt.gca().invert_yaxis()
        plt.xlabel(f"$\\Delta${ar_name} [$\\mu$m]")
        plt.ylabel(f"$\\Delta${az_name} [$\\mu$m]")
        plt.colorbar(label="Conn. prob.")
        if zidx == 0:
            plt.title(f"Model: {model.__class__.__name__}")

    plt.suptitle(
        f"Reduced position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>"
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)
