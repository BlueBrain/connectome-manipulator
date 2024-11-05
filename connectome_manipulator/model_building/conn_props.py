# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for building connection/synapse properties models"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from scipy.optimize import curve_fit
from scipy.stats import norm

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_edges_population, get_node_ids, get_cv_data

MAX_UNIQUE_COUNT = 100  # To be used in discrete distributions

# Ideas for improvement:
#   *Detect actual distributions of synaptic properties (incl. data type!)
#   *Capture cross-correlations between synaptic properties


def extract(
    circuit,
    min_sample_size_per_group=None,
    max_sample_size_per_group=None,
    hist_bins=51,  # [In case of constant distributions, better to use odd bin count so that center bin symmetrically centered around constant value; otherwise, rounding problems at bin edges possible]
    sel_props=None,
    sel_src=None,
    sel_dest=None,
    edges_popul_name=None,
    CV_dict=None,
    **_,
):
    """Extracts statistics (like mean, std, min, max, histogram, ...) for synapse properties of samples of connections between each pair of m-types.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        min_sample_size_per_group (int): Minimum number of samples (connections) required based on which to estimate statistics for a given pair of m-types; otherwise, no statistics will be estimated
        max_sample_size_per_group (int): Maximum number of samples (connections) based on which to estimate statistics for a given pair of m-types; if more samples are available, a random subset is used
        hist_bins (int): Number of bins for extracting histograms
        sel_props (list-like): List of synaptic property names to extract statistics from; None to select default properties
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        edges_popul_name (str): Name of SONATA egdes population to extract data from
        CV_dict (dict): Optional cross-validation dictionary, containing "n_folds" (int), "fold_idx" (int), "training_set" (bool) keys; will be automatically provided by the framework if "CV_folds" are specified

    Returns:
        dict: Dictionary containing the extracted data elements

    Note:
        The following statistics will be extracted:

        * "mean" (float): Data mean
        * "std" (float): Standard deviation
        * "min" (float): Minimum value
        * "max" (float): Maximum value
        * "hist" (tuple): Histogram with ``hist_bins`` bins; stored as (counts, edges) tuple as returned by :func:`numpy.histogram`
        * "norm_loc" (float): Location estimate for truncnorm distributions
        * "norm_scale" (float): Scale estimate for truncnorm distributions
        * "val" (list-like): List of unique values for discrete distributions (*)
        * "cnt" (list-like): List of unique value counts for discrete distributions (*)
        * "p" (list-like): List of unique value probabilities for discrete distributions (*)
        * "shared_within" (bool): Flag indicating that all synapses within a connection share the same value for a given property; not applicable for #synapses/connection

        (*) Discrete statistics (i.e., "val"/"cnt"/"p") are only stored if there are not too many (less or equal ``MAX_UNIQUE_COUNT=100``).

    Note:
        Statistics for #synapses/connection will always be extracted, even if no properties are selected under ``sel_props`` (i.e., empty list).
    """
    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)

    # Cross-validation (optional)
    node_ids_src, node_ids_dest = get_cv_data([node_ids_src, node_ids_dest], CV_dict)

    node_ids = [node_ids_src, node_ids_dest]

    m_types = [
        np.unique(n.get(ids, properties="mtype")).tolist() for n, ids in zip(nodes, node_ids)
    ]
    m_type_class = [
        [
            nodes[i]
            .get(
                np.intersect1d(nodes[i].ids({"mtype": m}), node_ids[i]), properties="synapse_class"
            )
            .iloc[0]
            for m in m_types[i]
        ]
        for i in range(len(nodes))
    ]
    m_type_layer = [
        [
            nodes[i]
            .get(np.intersect1d(nodes[i].ids({"mtype": m}), node_ids[i]), properties="layer")
            .iloc[0]
            for m in m_types[i]
        ]
        for i in range(len(nodes))
    ]
    if sel_props is None:  # Select all usable properties
        syn_props = list(
            filter(
                lambda x: not np.any(
                    [excl in x for excl in ["@", "delay", "afferent", "efferent", "spine_length"]]
                ),
                edges.property_names,
            )
        )
    else:
        syn_props = list(sel_props)
        syn_props_check = np.array([prop in edges.property_names for prop in syn_props])
        if len(syn_props_check) > 0:
            log.log_assert(
                np.all(syn_props_check),
                f"Selected synapse properties not found: {np.array(syn_props)[~syn_props_check]}",
            )

    log.debug(
        f"Estimating statistics for {len(syn_props)} properties (plus #synapses/connection) between {len(m_types[0])}x{len(m_types[1])} m-types (min_sample_size_per_group={min_sample_size_per_group}, max_sample_size_per_group={max_sample_size_per_group}, CV_dict={CV_dict})"
    )

    # Statistics for #syn/conn
    # (incl. fitted norm loc/scale for truncnorm distribution,
    #  incl. unique values/counts/probabilities for discrete distribution)
    syns_per_conn_data = {
        "mean": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "std": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "min": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "max": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "hist": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
        "norm_loc": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "norm_scale": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "val": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
        "cnt": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
        "p": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
    }

    # Statistics for synapse/connection properties
    # (incl. fitted norm loc/scale for truncnorm distribution,
    #  incl. unique values/counts/probabilities for discrete distribution;
    #  incl. shared_within to indicate that all synapses within a connection share same value)
    conn_prop_data = {
        "mean": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "std": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "min": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "max": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "hist": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object),
        "norm_loc": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "norm_scale": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
        "val": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object),
        "cnt": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object),
        "p": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object),
        "shared_within": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan),
    }

    # Extract statistics
    conn_counts = {"min": np.inf, "max": -np.inf, "sel": 0}  # Count connections for reporting
    pbar = progressbar.ProgressBar()
    for sidx in pbar(range(len(m_types[0]))):
        sids = np.intersect1d(src_nodes.ids({"mtype": m_types[0][sidx]}), node_ids_src)
        for tidx in range(len(m_types[1])):
            tids = np.intersect1d(tgt_nodes.ids({"mtype": m_types[1][tidx]}), node_ids_dest)
            edges_sel = edges.pathway_edges(
                sids, tids, ["@source_node", "@target_node"] + syn_props
            )
            if edges_sel.shape[0] == 0:  # No synapses between pair of m-types
                continue

            _, syn_conn_idx, num_syn_per_conn = np.unique(
                edges_sel[["@source_node", "@target_node"]],
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            conn_counts["min"] = min(conn_counts["min"], len(num_syn_per_conn))
            conn_counts["max"] = max(conn_counts["max"], len(num_syn_per_conn))
            conn_sel = range(len(num_syn_per_conn))  # Select all connections
            if (
                min_sample_size_per_group is not None
                and min_sample_size_per_group > 0
                and len(conn_sel) < min_sample_size_per_group
            ):
                # Not enough connections available
                continue

            if max_sample_size_per_group is not None and 0 < max_sample_size_per_group < len(
                conn_sel
            ):
                # Subsample connections
                conn_sel = sorted(
                    np.random.choice(conn_sel, max_sample_size_per_group, replace=False)
                )
            conn_counts["sel"] += 1

            syns_per_conn_data["mean"][sidx, tidx] = np.mean(num_syn_per_conn[conn_sel])
            syns_per_conn_data["std"][sidx, tidx] = np.std(num_syn_per_conn[conn_sel])
            syns_per_conn_data["min"][sidx, tidx] = np.min(num_syn_per_conn[conn_sel])
            syns_per_conn_data["max"][sidx, tidx] = np.max(num_syn_per_conn[conn_sel])
            hist_counts, bin_edges = np.histogram(num_syn_per_conn[conn_sel], bins=hist_bins)
            syns_per_conn_data["hist"][sidx, tidx] = (hist_counts, bin_edges)
            bin_centers = [np.mean(bin_edges[i : i + 2]) for i in range(len(hist_counts))]
            loc, scale, _ = _norm_fitting(
                bin_centers,
                hist_counts,
                def_mn=syns_per_conn_data["mean"][sidx, tidx],
                def_sd=syns_per_conn_data["std"][sidx, tidx],
            )
            syns_per_conn_data["norm_loc"][sidx, tidx] = loc
            syns_per_conn_data["norm_scale"][sidx, tidx] = scale
            v, c = np.unique(num_syn_per_conn[conn_sel], return_counts=True)
            if len(v) == 1 or (
                len(v) < len(num_syn_per_conn[conn_sel]) and len(v) <= MAX_UNIQUE_COUNT
            ):  # Store discrete values only if not too many [otherwise, it is probably not a discrete distribution]
                syns_per_conn_data["val"][sidx, tidx] = v
                syns_per_conn_data["cnt"][sidx, tidx] = c
                syns_per_conn_data["p"][sidx, tidx] = c / np.sum(c)
            else:
                syns_per_conn_data["val"][sidx, tidx] = []
                syns_per_conn_data["cnt"][sidx, tidx] = []
                syns_per_conn_data["p"][sidx, tidx] = []
                # log.warning(f'Discrete #synapses/connection values not stored for {m_types[0][sidx]}-{m_types[1][tidx]} ({len(v)} of {len(num_syn_per_conn[conn_sel])} unique values)!')

            # Collect property statistics over connections (if shared values) or synapses (if non-shared values)
            for pidx, p in enumerate(syn_props):
                is_shared = _check_shared_property(edges_sel, p, conn_sel, syn_conn_idx)
                prop_values = _get_property_values(edges_sel, p, conn_sel, syn_conn_idx, is_shared)

                conn_prop_data["shared_within"][sidx, tidx, pidx] = is_shared
                conn_prop_data["mean"][sidx, tidx, pidx] = np.mean(
                    prop_values, dtype=np.float64
                )  # [float64 required, otherwise rounding problems!!!]
                conn_prop_data["std"][sidx, tidx, pidx] = np.std(
                    prop_values, dtype=np.float64
                )  # [float64 required, otherwise rounding problems!!!]
                conn_prop_data["min"][sidx, tidx, pidx] = np.min(prop_values).astype(np.float64)
                conn_prop_data["max"][sidx, tidx, pidx] = np.max(prop_values).astype(np.float64)
                hist_counts, bin_edges = np.histogram(prop_values, bins=hist_bins)
                conn_prop_data["hist"][sidx, tidx, pidx] = (hist_counts, bin_edges)
                bin_centers = [np.mean(bin_edges[i : i + 2]) for i in range(len(hist_counts))]
                loc, scale, _ = _norm_fitting(
                    bin_centers,
                    hist_counts,
                    def_mn=conn_prop_data["mean"][sidx, tidx, pidx],
                    def_sd=conn_prop_data["std"][sidx, tidx, pidx],
                )
                conn_prop_data["norm_loc"][sidx, tidx, pidx] = loc
                conn_prop_data["norm_scale"][sidx, tidx, pidx] = scale
                v, c = np.unique(prop_values, return_counts=True)
                if len(v) == 1 or (
                    len(v) < len(prop_values) and len(v) <= MAX_UNIQUE_COUNT
                ):  # Store discrete values only if not too many [otherwise, it is probably not a discrete distribution]
                    conn_prop_data["val"][sidx, tidx, pidx] = v
                    conn_prop_data["cnt"][sidx, tidx, pidx] = c
                    conn_prop_data["p"][sidx, tidx, pidx] = c / np.sum(c)
                else:
                    conn_prop_data["val"][sidx, tidx, pidx] = []
                    conn_prop_data["cnt"][sidx, tidx, pidx] = []
                    conn_prop_data["p"][sidx, tidx, pidx] = []
                    # log.warning(f'Discrete {p} values not stored for {m_types[0][sidx]}-{m_types[1][tidx]} ({len(v)} of {len(prop_values)} unique values)!')

    log.debug(
        f'Between {conn_counts["min"]} and {conn_counts["max"]} connections per pathway found. {conn_counts["sel"]} of {len(m_types[0])}x{len(m_types[1])} pathways selected.'
    )

    return {
        "syns_per_conn_data": syns_per_conn_data,
        "conn_prop_data": conn_prop_data,
        "m_types": m_types,
        "m_type_class": m_type_class,
        "m_type_layer": m_type_layer,
        "syn_props": syn_props,
        "hist_bins": hist_bins,
    }


def build(
    syns_per_conn_data,
    conn_prop_data,
    m_types,
    m_type_class,
    m_type_layer,
    syn_props,
    distr_types={},
    data_types={},
    data_bounds={},
    shared_within={},
    **_,
):
    """Fit model distribution to data, incl. missing values interpolated at different levels of granularity.

    Args:
        syns_per_conn_data (dict): Dictionary with entries for all statistics (see Notes under :func:`extract`) estimated for #synapses/connection, each of which contains a numpy.ndarray of size <#source-mtypes x #target-mtypes> for all pairs of m-types, as returned by :func:`extract`
        conn_prop_data (dict): Dictionary with entries for all statistics (see Notes under :func:`extract`) estimated for all synaptic properties, each of which contains a numpy.ndarray of size <#source-mtypes x #target-mtypes x #properties> for all pairs of m-types and synapse properties, as returned by :func:`extract`
        m_types (list): Two-element list of lists of source (pre-synaptic) and target (post-synaptic) m-types, as returned by :func:`extract`
        m_type_class (list): Two-element list of lists of synapse classes (i.e., EXC, INH) belonging to each source and target m-type (assuming that each m-type corresponds to exactly one synapse class), as returned by :func:`extract`
        m_type_layer (list): Two-element list of lists of layers belonging to each source and target m-type (assuming that each m-type corresponds to exactly one cortical layer), as returned by :func:`extract`
        syn_props (list-like): List of synaptic property names stored in ``conn_prop_data``, as returned by :func:`extract`
        distr_types (dict): Optional dictionary specifying the distribution type (dict value) for each property (dict key); if omitted, a "normal" distribution is assumed (and will raise a warning); see Notes for available distribution types
        data_types (dict): Optional dictionary specifying the output data type (dict value; e.g., "int", "float", ...) for each property (dict key) when drawing values from the fitted model
        data_bounds (dict): Optional dictionary specifying the output data bounds (dict value; list-like with two elements for lower/upper bounds) for each property (dict key) when drawing values from the fitted model
        shared_within (dict): Optional dictionary specifying if the same values are shared among synapses belonging to the same connections (boolean dict value) for each property (dict key) when drawing values from the fitted model; can be used to manually overwrite the data-derived value

    Returns:
        connectome_manipulator.model_building.model_types.ConnPropsModel: Fitted connection/synapse properties model

    Note:
        The property name "n_syn_per_conn" (defined by ``model_types.N_SYN_PER_CONN_NAME``) can be used in ``distr_types``, ``data_types``, and ``data_bounds`` dicts to specify distribution types, data types, and bounds for the #synapses/connection property.

    Note:
        The following distribution types are supported:

        * "constant": Constant value (define by "mean")
        * "normal": Gaussian normal distribution (define by "mean", "std")
        * "truncnorm": Truncated normal distribution (define by "norm_loc", "norm_scale", "min", "max")
        * "gamma": Gamma distribution (define by "mean", "std")
        * "poisson": Poisson distribution (define by "mean")
        * "ztpoisson": Zero-truncated poisson distribution (define by "mean")
        * "discrete": Discrete distribution (define by "val", "p")
        * "zero": Empty distribution always returning zero; can be used to model unused parameters
    """
    # Interpolate missing values in lookup tables
    syns_per_conn_model = {k: v.copy() for (k, v) in syns_per_conn_data.items()}
    conn_prop_model = {k: v.copy() for (k, v) in conn_prop_data.items()}
    missing_list = np.array(np.where(np.logical_not(np.isfinite(syns_per_conn_model["mean"])))).T
    level_counts = {}  # Count interpolation levels for reporting
    for sidx, tidx in missing_list:
        # Select level of granularity
        for level in range(5):
            if level == 0:  # Use source m-type/target layer/synapse class value, if existent
                src_sel = np.array([sidx])
                tgt_sel = np.where(
                    np.logical_and(
                        np.array(m_type_layer[1]) == m_type_layer[1][tidx],
                        np.array(m_type_class[1]) == m_type_class[1][tidx],
                    )
                )[0]
            elif level == 1:  # Use source m-type/target synapse class value, if existent
                src_sel = np.array([sidx])
                tgt_sel = np.where(np.array(m_type_class[1]) == m_type_class[1][tidx])[0]
            elif level == 2:  # Use per layer/synapse class value, if existent
                src_sel = np.where(
                    np.logical_and(
                        np.array(m_type_layer[0]) == m_type_layer[0][sidx],
                        np.array(m_type_class[0]) == m_type_class[0][sidx],
                    )
                )[0]
                tgt_sel = np.where(
                    np.logical_and(
                        np.array(m_type_layer[1]) == m_type_layer[1][tidx],
                        np.array(m_type_class[1]) == m_type_class[1][tidx],
                    )
                )[0]
            elif level == 3:  # Use per synapse class value, if existent
                src_sel = np.where(np.array(m_type_class[0]) == m_type_class[0][sidx])[0]
                tgt_sel = np.where(np.array(m_type_class[1]) == m_type_class[1][tidx])[0]
            else:  # Otherwise: Use overall value
                src_sel = np.array(list(range(len(m_types[0]))))
                tgt_sel = np.array(list(range(len(m_types[1]))))
            if np.any(np.isfinite(syns_per_conn_data["mean"][src_sel, :][:, tgt_sel])):
                level_counts[f"Level{level}"] = level_counts.get(f"Level{level}", 0) + 1
                break

        def merge_uniq(vals, cnts):
            """Helper function to merge unique values/counts"""
            vals = list(
                filter(lambda x: len(x) > 0 if hasattr(x, "__iter__") else np.isfinite(x), vals)
            )  # [Remove NaNs/empty lists, so not to mess up data type]
            cnts = list(
                filter(lambda x: len(x) > 0 if hasattr(x, "__iter__") else np.isfinite(x), cnts)
            )  # [Remove NaNs/empty lists, so not to mess up data type]
            if len(vals) > 0:
                vc_dict = {
                    v: 0 for v in np.unique(np.hstack(vals))
                }  # Init value/count dict [increasing order!]
                for v, c in zip(np.hstack(vals), np.hstack(cnts)):
                    vc_dict[v] += c  # Add to existing count
                vals = np.array(list(vc_dict.keys()))
                cnts = np.array([vc_dict[v] for v in vals])
            return vals, cnts

        # Interpolate missing values
        syns_per_conn_model["mean"][sidx, tidx] = np.nanmean(
            syns_per_conn_data["mean"][src_sel, :][:, tgt_sel]
        )
        syns_per_conn_model["std"][sidx, tidx] = np.nanmean(
            syns_per_conn_data["std"][src_sel, :][:, tgt_sel]
        )
        syns_per_conn_model["min"][sidx, tidx] = np.nanmin(
            syns_per_conn_data["min"][src_sel, :][:, tgt_sel]
        )
        syns_per_conn_model["max"][sidx, tidx] = np.nanmax(
            syns_per_conn_data["max"][src_sel, :][:, tgt_sel]
        )
        syns_per_conn_model["norm_loc"][sidx, tidx] = np.nanmean(
            syns_per_conn_data["norm_loc"][src_sel, :][:, tgt_sel]
        )
        syns_per_conn_model["norm_scale"][sidx, tidx] = np.nanmean(
            syns_per_conn_data["norm_scale"][src_sel, :][:, tgt_sel]
        )

        uvals = syns_per_conn_data["val"][src_sel, :][:, tgt_sel].flatten()
        ucnts = syns_per_conn_data["cnt"][src_sel, :][:, tgt_sel].flatten()
        v, c = merge_uniq(uvals, ucnts)
        syns_per_conn_model["val"][sidx, tidx] = v
        syns_per_conn_model["cnt"][sidx, tidx] = c
        syns_per_conn_model["p"][sidx, tidx] = c / np.sum(c)

        conn_prop_model["mean"][sidx, tidx, :] = [
            np.nanmean(conn_prop_data["mean"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["std"][sidx, tidx, :] = [
            np.nanmean(conn_prop_data["std"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["min"][sidx, tidx, :] = [
            np.nanmin(conn_prop_data["min"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["max"][sidx, tidx, :] = [
            np.nanmax(conn_prop_data["max"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["norm_loc"][sidx, tidx, :] = [
            np.nanmean(conn_prop_data["norm_loc"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["norm_scale"][sidx, tidx, :] = [
            np.nanmean(conn_prop_data["norm_scale"][src_sel, :, p][:, tgt_sel])
            for p in range(len(syn_props))
        ]
        conn_prop_model["shared_within"][sidx, tidx, :] = [
            np.round(np.nanmean(conn_prop_data["shared_within"][src_sel, :, p][:, tgt_sel])).astype(
                bool
            )
            for p in range(len(syn_props))
        ]  # Majority vote in case of inconsistent sharing behavior

        for pidx in range(len(syn_props)):
            uvals = [conn_prop_model["val"][s][t][pidx] for s in src_sel for t in tgt_sel]
            ucnts = [conn_prop_model["cnt"][s][t][pidx] for s in src_sel for t in tgt_sel]
            v, c = merge_uniq(uvals, ucnts)
            conn_prop_model["val"][sidx, tidx, pidx] = v
            conn_prop_model["cnt"][sidx, tidx, pidx] = c
            conn_prop_model["p"][sidx, tidx, pidx] = c / np.sum(c)

    log.info(
        f"Interpolated {missing_list.shape[0]} missing values. Interpolation level counts: {{k: level_counts[k] for k in sorted(level_counts.keys())}}"
    )

    # Create model properties dictionary
    prop_model_dict = {}
    for pidx, prop in enumerate(syn_props + [model_types.N_SYN_PER_CONN_NAME]):
        prop_model_dict[prop] = {}
        if prop not in distr_types:
            log.warning(f'No distribution type for "{prop}" specified - Using "normal"!')
        distr_type = distr_types.get(prop, "normal")
        log.log_assert(
            distr_type in model_types.ConnPropsModel.distribution_attributes,
            f'ERROR: Distribution type "{distr_type}" not supported!',
        )
        dtype = data_types.get(prop)
        bounds = data_bounds.get(prop)
        shared = shared_within.get(prop)
        for sidx, src in enumerate(m_types[0]):
            prop_model_dict[prop][src] = {}
            for tidx, tgt in enumerate(m_types[1]):
                attr_dict = {"type": distr_type}
                distr_attr = model_types.ConnPropsModel.distribution_attributes[distr_type]
                if prop == model_types.N_SYN_PER_CONN_NAME:
                    log.log_assert(
                        np.all([attr in syns_per_conn_model for attr in distr_attr]),
                        f'ERROR: Not all required attribute(s) {distr_attr} for distribution "{distr_type}" found!',
                    )
                    attr_dict.update(
                        {attr: syns_per_conn_model[attr][sidx, tidx] for attr in distr_attr}
                    )
                else:
                    distr_attr = distr_attr + ["shared_within"]
                    log.log_assert(
                        np.all([attr in conn_prop_model for attr in distr_attr]),
                        f'ERROR: Not all required attribute(s) {distr_attr} for distribution "{distr_type}" found!',
                    )
                    attr_dict.update(
                        {attr: conn_prop_model[attr][sidx, tidx, pidx] for attr in distr_attr}
                    )
                if dtype is not None:
                    attr_dict.update({"dtype": dtype})
                if bounds is not None and hasattr(bounds, "__iter__") and len(bounds) == 2:
                    if bounds[0] is not None:
                        attr_dict.update({"lower_bound": bounds[0]})
                    if bounds[1] is not None:
                        attr_dict.update({"upper_bound": bounds[1]})
                if shared is not None and isinstance(shared, bool):
                    # Overwrite data-derived value
                    attr_dict.update({"shared_within": shared})
                elif distr_type == "zero":
                    # Overwrite data-derived value for unused properties
                    attr_dict.update({"shared_within": False})
                prop_model_dict[prop][src][tgt] = attr_dict

    # Create model
    model = model_types.ConnPropsModel(
        src_types=m_types[0], tgt_types=m_types[1], prop_stats=prop_model_dict
    )
    log.debug("Model description:\n%s", model)

    return model


def plot(
    out_dir,
    syns_per_conn_data,
    conn_prop_data,
    m_types,
    syn_props,
    model,
    plot_sample_size=1000,
    **_,
):  # pragma: no cover
    """Visualizes extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        syns_per_conn_data (dict): Dictionary with entries for all statistics (see Notes under :func:`extract`) estimated for #synapses/connection, each of which contains a numpy.ndarray of size <#source-mtypes x #target-mtypes> for all pairs of m-types, as returned by :func:`extract`
        conn_prop_data (dict): Dictionary with entries for all statistics (see Notes under :func:`extract`) estimated for all synaptic properties, each of which contains a numpy.ndarray of size <#source-mtypes x #target-mtypes x #properties> for all pairs of m-types and synapse properties, as returned by :func:`extract`
        m_types (list): Two-element list of lists of source (pre-synaptic) and target (post-synaptic) m-types, as returned by :func:`extract`
        syn_props (list-like): List of synaptic property names stored in ``conn_prop_data``, as returned by :func:`extract`
        model (connectome_manipulator.model_building.model_types.ConnPropsModel): Fitted connection/synapse properties model, as returned by :func:`build`
        plot_sample_size (int): Number of samples to draw when plotting model distributions
    """
    model_params = model.get_param_dict()
    prop_names = model.get_prop_names()

    # Plot data vs. model: property maps
    title_str = ["Data", "Model"]
    for stat_sel in ["mean", "std", "shared_within"]:
        for pidx, p in enumerate(prop_names):
            if pidx < len(prop_names) - 1:
                data_stat_sel = conn_prop_data[stat_sel][:, :, pidx]
            else:  # Last element is n_syn_per_conn
                if stat_sel not in syns_per_conn_data:
                    continue  # Skip if statistics does not exist here
                data_stat_sel = syns_per_conn_data[stat_sel]
            plt.figure(figsize=(8, 3), dpi=300)
            model_stat_sel = np.full((len(m_types[0]), len(m_types[1])), np.nan)
            for sidx, s in enumerate(m_types[0]):
                for tidx, t in enumerate(m_types[1]):
                    model_stat_sel[sidx, tidx] = _get_model_stat(
                        stat_sel, model_params["prop_stats"][p][s][t]
                    )
            for didx, data in enumerate([data_stat_sel, model_stat_sel]):
                plt.subplot(1, 2, didx + 1)
                plt.imshow(data, interpolation="nearest", cmap="jet")
                plt.xticks(range(len(m_types[1])), m_types[1], rotation=90, fontsize=3)
                plt.yticks(range(len(m_types[0])), m_types[0], rotation=0, fontsize=3)
                plt.colorbar()
                plt.title(title_str[didx])
            plt.suptitle(f"{p} ({stat_sel})", fontweight="bold")
            plt.tight_layout()

            out_fn = os.path.abspath(
                os.path.join(out_dir, f"data_vs_model_map_{stat_sel}__{p}.png")
            )
            log.info(f"Saving {out_fn}...")
            plt.savefig(out_fn)

    # Plot data vs. model: Distribution histogram examples (generative model) + AUC
    conn_counts = [
        [
            (
                np.sum(syns_per_conn_data["hist"][sidx, tidx][0])
                if (
                    hasattr(syns_per_conn_data["hist"][sidx, tidx], "__iter__")
                    and len(syns_per_conn_data["hist"][sidx, tidx]) > 0
                )
                else 0
            )
            for tidx in range(len(m_types[1]))
        ]
        for sidx in range(len(m_types[0]))
    ]
    max_pathways = np.where(
        np.array(conn_counts) == np.max(conn_counts)
    )  # Select pathway(s) with maximum number of connections (i.e., most robust statistics)
    sidx, tidx = [
        max_pathways[0][0],
        max_pathways[1][0],
    ]  # Select first of these pathways for plotting
    src, tgt = [m_types[0][sidx], m_types[1][tidx]]
    for pidx, p in enumerate(prop_names):
        plt.figure(figsize=(5, 3), dpi=300)
        if pidx < len(syn_props):
            data_hist = conn_prop_data["hist"][sidx, tidx, pidx]
        else:
            data_hist = syns_per_conn_data["hist"][sidx, tidx]
        plt.bar(
            data_hist[1][:-1],
            data_hist[0] / np.sum(data_hist[0]),
            align="edge",
            width=np.min(np.diff(data_hist[1])),
            label=f"Data (N={np.max(conn_counts)})",
        )
        model_data = np.hstack(
            [
                model.draw(prop_name=p, src_type=src, tgt_type=tgt, size=1)
                for n in range(plot_sample_size)
            ]
        )  # Draw <plot_sample_size> single values from property distribution
        hist_bins = data_hist[1]  # Use same model distribution binning as for data
        bin_size = np.min(np.diff(hist_bins))
        if min(model_data) < hist_bins[0]:  # Extend binning to lower values to cover whole range
            hist_bins = np.hstack(
                [
                    np.flip(np.arange(hist_bins[0], min(model_data) - bin_size, -bin_size)),
                    hist_bins[1:],
                ]
            )
        if max(model_data) > hist_bins[-1]:  # Extend binning to higher values to cover whole range
            hist_bins = np.hstack(
                [hist_bins[:-1], np.arange(hist_bins[-1], max(model_data) + bin_size, bin_size)]
            )
        model_hist = np.histogram(model_data, bins=hist_bins)
        plt.step(
            model_hist[1],
            np.hstack([model_hist[0][0], model_hist[0]]) / np.sum(model_hist[0]),
            where="pre",
            color="tab:orange",
            label=f"Model (N={plot_sample_size})",
        )
        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.title(f"{src} to {tgt}", fontweight="bold")
        plt.xlabel(p)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        out_fn = os.path.abspath(os.path.join(out_dir, f"data_vs_model_hist__{p}.png"))
        log.info(f"Saving {out_fn}...")
        plt.savefig(out_fn)

        # AUC/ROC plot
        (
            AUC,
            ERR,
            _FPR,
            _TPR,
            _distr_norm_A,
            _distr_norm_B,
            _pos_norm_A,
            _pos_norm_B,
            xp,
            yp,
            x,
            y,
        ) = _compute_AUC(data_hist[0], model_hist[0], data_hist[1], model_hist[1])

        plt.figure()
        plt.plot([0.0, 1.0], [0.0, 1.0], "--k")
        plt.plot(xp, yp, ".-")
        plt.plot(x, y, ".--")
        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.title(f"{src} to {tgt}: {p} (AUC={AUC:.2f}, ERR={ERR:.2f})", fontweight="bold")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout()

        out_fn = os.path.abspath(os.path.join(out_dir, f"data_vs_model_AUC__{p}.png"))
        log.info(f"Saving {out_fn}...")
        plt.savefig(out_fn)


# Helper functions
def _norm_fitting(
    hist_values,
    hist_counts,
    max_nfev=30,
    rel_fit_err_th=0.5,
    def_mn=np.nan,
    def_sd=np.nan,
    def_sc=np.nan,
):
    """Helper function to extract fitted norm mean/SD/scaling attributes from histogram (using a large error threshold by default in order to get at least a rough estimate)."""

    def norm_fct(x, mn, sd, sc):
        return sc * norm(loc=mn, scale=sd).pdf(x)

    p0 = [np.mean(hist_values), 1.0, 1.0]
    bounds = [[min(hist_values), 0, 0], [max(hist_values), np.inf, np.inf]]
    # Note: "mean" assumed to be within hist value bounds!!

    invalid_fit = False
    try:
        (mn_opt, sd_opt, sc_opt), pcov, *_ = curve_fit(
            norm_fct, hist_values, hist_counts, p0=p0, bounds=bounds, max_nfev=max_nfev
        )
    except (
        ValueError,
        RuntimeError,
    ):  # Raised if input data invalid or optimization fails
        invalid_fit = True

    if not invalid_fit:
        rel_err = np.sqrt(np.diag(pcov)) / np.array(
            [mn_opt, sd_opt, sc_opt]
        )  # Rel. standard error of the coefficients
        # log.debug(f"Rel. error of norm model fit: {rel_err}")
        if not all(np.isfinite(rel_err)) or max(rel_err) > rel_fit_err_th:
            # log.error(
            #     f"Rel. error of norm model fit exceeds error threshold of {rel_fit_err_th} (or could not be determined)!"
            # )
            invalid_fit = True

    if invalid_fit:  # Set default values
        mn_opt = def_mn
        sd_opt = def_sd
        sc_opt = def_sc

    return mn_opt, sd_opt, sc_opt


def _check_shared_property(edges_sel, prop_name, conn_sel, syn_conn_idx):
    """Check if shared property values within connections."""
    if len(conn_sel) > 1 and len(np.unique(edges_sel[prop_name])) == 1:
        # In case of a constant overall distribution, assume no sharing
        is_shared = False
    else:
        is_shared = True
        for c in conn_sel:
            if len(np.unique(edges_sel.loc[syn_conn_idx == c, prop_name])) > 1:
                # Found different property values within same connection
                is_shared = False
                break
    return is_shared


def _get_property_values(edges_sel, prop_name, conn_sel, syn_conn_idx, is_shared):
    """Collect property values over connections (if shared values) or synapses (if non-shared values)."""
    prop_values = []
    for c in conn_sel:
        if is_shared:
            # Shared within connection, so take only first value
            prop_values.append(edges_sel.loc[syn_conn_idx == c, prop_name].iloc[0])
        else:
            # Different values within connection, so take all values
            prop_values.append(edges_sel.loc[syn_conn_idx == c, prop_name].to_numpy())
    prop_values = np.hstack(prop_values)
    return prop_values


def _compute_AUC(distr_A, distr_B, bins_A, bins_B, dth=0.05, dx=0.01):
    """Computes area under the ROC curve for comparing two distributions."""
    pos_A = np.array(
        [np.mean(bins_A[i : i + 2]) for i in range(len(bins_A) - 1)]
    )  # Bin center positions
    pos_B = np.array(
        [np.mean(bins_B[i : i + 2]) for i in range(len(bins_B) - 1)]
    )  # Bin center positions

    min_range = np.minimum(pos_A[0], pos_B[0])
    max_range = np.maximum(pos_A[-1], pos_B[-1])
    pos_norm_A = (pos_A - min_range) / (max_range - min_range)  # Normalized positions
    pos_norm_B = (pos_B - min_range) / (max_range - min_range)  # Normalized positions

    bin_size_A = np.mean(np.diff(bins_A))
    bin_size_B = np.mean(np.diff(bins_B))
    distr_norm_A = distr_A / (np.sum(distr_A) * bin_size_A)
    distr_norm_B = distr_B / (np.sum(distr_B) * bin_size_B)

    ths = np.arange(0, 1.0 + dth, dth)  # Thresholds
    TPR = np.array([np.sum(distr_norm_A[pos_norm_A >= th] * bin_size_A) for th in ths])
    FPR = np.array([np.sum(distr_norm_B[pos_norm_B >= th] * bin_size_B) for th in ths])

    sort_idx = np.argsort(FPR)
    xp = FPR[sort_idx]
    yp = TPR[sort_idx]

    x = np.arange(0.0, 1.0 + dx, dx)
    y = np.interp(x, xp, yp)
    AUC = np.trapz(y, x, dx) - 0.5
    ERR = np.trapz(np.abs(y - x), x, dx)  # Error: Area of abs. differences

    return AUC, ERR, FPR, TPR, distr_norm_A, distr_norm_B, pos_norm_A, pos_norm_B, xp, yp, x, y


def _get_model_stat(stat, m_params):
    """Get distribution statistic (if existing) or derive from other model paramters, if possible."""
    val = np.nan
    if stat in m_params:  # Return existing stat. parameter
        val = m_params[stat]
    else:
        if m_params["type"] == "constant":
            if stat == "std":
                val = 0.0
            elif stat in {"min", "max"}:
                val = m_params["mean"]
        elif m_params["type"] == "discrete":
            # Derive missing statistics from discrete values/probabilities
            if stat == "mean":
                val = np.sum(np.array(m_params["p"] * np.array(m_params["val"])))
            elif stat == "std":
                m = np.sum(np.array(m_params["p"] * np.array(m_params["val"])))
                val = np.sqrt(
                    np.sum(np.array(m_params["p"]) * (np.array(m_params["val"]) - m) ** 2)
                )
            elif stat == "min":
                val = np.min(np.array(m_params["val"])[np.array(m_params["p"]) > 0.0])
            elif stat == "max":
                val = np.max(np.array(m_params["val"])[np.array(m_params["p"]) > 0.0])
        elif m_params["type"] == "truncnorm":
            # Estimate missing statistics from generated truncnorm distribution
            distr = model_types.ConnPropsModel.draw_from_distribution(m_params, size=200)
            if stat == "mean":
                val = np.mean(distr)
            elif stat == "std":
                val = np.std(distr)
        elif m_params["type"] == "poisson":
            if stat == "std":
                # Derive std from poisson mean
                val = np.sqrt(m_params["mean"])
            elif stat == "min":
                val = 0
        elif m_params["type"] == "ztpoisson":
            if stat == "std":
                # Derive std from zero-truncated poisson mean
                mn = m_params["mean"]
                lam = model_types.ConnPropsModel.compute_ztpoisson_lambda(mn)
                val = np.sqrt(mn * (1 + lam - mn))
            elif stat == "min":
                val = 1
        elif m_params["type"] == "zero":
            # Set missing statistics to zero
            val = 0.0
    return val
