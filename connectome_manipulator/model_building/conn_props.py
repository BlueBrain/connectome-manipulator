"""Module for building connection/synapse properties model, consisting of three basic functions:

- extract(...): Extracts statistics for connection/synaptic properties between samples of neurons for each pair of m-types
- build(...): Fit model distribution to data, incl. missing values interpolated at different levels of granularity
- plot(...): Visualizes extracted data vs. actual model output
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_edges_population, get_node_ids

MAX_UNIQUE_COUNT = 100  # To be used in discrete distributions

# Ideas for improvement:
#   *Detect actual distributions of synaptic properties (incl. data type!)
#   *Capture cross-correlations between synaptic properties


def extract(
    circuit,
    min_sample_size_per_group=None,
    max_sample_size_per_group=None,
    hist_bins=50,
    sel_props=None,
    sel_src=None,
    sel_dest=None,
    **_,
):
    """Extract statistics for synaptic properties between samples of neurons for each pair of m-types.

    (sel_props: None to select default properties; if no properties are selected (empty list),
                only #synapses/connection will be estimated)
    """
    # Select edge population
    edges = get_edges_population(circuit)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)
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
        f"Estimating statistics for {len(syn_props)} properties (plus #synapses/connection) between {len(m_types[0])}x{len(m_types[1])} m-types (min_sample_size_per_group={min_sample_size_per_group}, max_sample_size_per_group={max_sample_size_per_group})"
    )

    # Statistics for #syn/conn
    syns_per_conn_data = {
        "mean": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "std": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "min": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "max": np.full((len(m_types[0]), len(m_types[1])), np.nan),
        "hist": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
        "val": np.full(
            (len(m_types[0]), len(m_types[1])), np.nan, dtype=object
        ),  # Unique values (for discrete distribution)
        "cnt": np.full(
            (len(m_types[0]), len(m_types[1])), np.nan, dtype=object
        ),  # Unique value counts (for discrete distribution)
        "p": np.full((len(m_types[0]), len(m_types[1])), np.nan, dtype=object),
    }  # Unique value probabilities (for discrete distribution)

    # Statistics for synapse/connection properties
    conn_prop_data = {
        "mean": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan
        ),  # Property value means across connections
        "std": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan
        ),  # Property value stds across connections
        "std-within": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan
        ),  # Property value stds across synapses within connections
        "min": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan
        ),  # Property value overall min
        "max": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan
        ),  # Property value overall max
        "hist": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object
        ),  # Histogram of distribution
        "val": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object
        ),  # Unique values (for discrete distribution)
        "cnt": np.full(
            (len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object
        ),  # Unique value counts (for discrete distribution)
        "p": np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan, dtype=object),
    }  # Unique value probabilities (for discrete distribution)

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
            syns_per_conn_data["hist"][sidx, tidx] = np.histogram(
                num_syn_per_conn[conn_sel], bins=hist_bins
            )
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

            means_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            stds_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            mins_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            maxs_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            for cidx, c in enumerate(conn_sel):
                means_within[cidx, :] = np.mean(
                    edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0, dtype=np.float64
                )  # [float64 required, otherwise rounding problems!!!]
                stds_within[cidx, :] = np.std(
                    edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0, dtype=np.float64
                )
                mins_within[cidx, :] = np.min(
                    edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0
                ).astype(np.float64)
                maxs_within[cidx, :] = np.max(
                    edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0
                ).astype(np.float64)

            conn_prop_data["mean"][sidx, tidx, :] = np.mean(means_within, 0)
            conn_prop_data["std"][sidx, tidx, :] = np.std(means_within, 0)
            conn_prop_data["std-within"][sidx, tidx, :] = np.mean(stds_within, 0)
            conn_prop_data["min"][sidx, tidx, :] = np.min(mins_within, 0)
            conn_prop_data["max"][sidx, tidx, :] = np.max(maxs_within, 0)
            for pidx in range(len(syn_props)):
                conn_prop_data["hist"][sidx, tidx, pidx] = np.histogram(
                    means_within[:, pidx], bins=hist_bins
                )
                v, c = np.unique(means_within[:, pidx], return_counts=True)
                if len(v) == 1 or (
                    len(v) < len(means_within[:, pidx]) and len(v) <= MAX_UNIQUE_COUNT
                ):  # Store discrete values only if not too many [otherwise, it is probably not a discrete distribution]
                    conn_prop_data["val"][sidx, tidx, pidx] = v
                    conn_prop_data["cnt"][sidx, tidx, pidx] = c
                    conn_prop_data["p"][sidx, tidx, pidx] = c / np.sum(c)
                else:
                    conn_prop_data["val"][sidx, tidx, pidx] = []
                    conn_prop_data["cnt"][sidx, tidx, pidx] = []
                    conn_prop_data["p"][sidx, tidx, pidx] = []
                    # log.warning(f'Discrete {syn_props[pidx]} values not stored for {m_types[0][sidx]}-{m_types[1][tidx]} ({len(v)} of {len(means_within[:, pidx])} unique values)!')

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
    **_,
):
    """Build model from data (lookup table with missing values interpolated at different levels of granularity)."""
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
        conn_prop_model["std-within"][sidx, tidx, :] = [
            np.nanmean(conn_prop_data["std-within"][src_sel, :, p][:, tgt_sel])
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

        for pidx in range(len(syn_props)):
            uvals = [conn_prop_model["val"][s][t][pidx] for s in src_sel for t in tgt_sel]
            ucnts = [conn_prop_model["cnt"][s][t][pidx] for s in src_sel for t in tgt_sel]
            v, c = merge_uniq(uvals, ucnts)
            conn_prop_model["val"][sidx, tidx, pidx] = v
            conn_prop_model["cnt"][sidx, tidx, pidx] = c
            conn_prop_model["p"][sidx, tidx, pidx] = c / np.sum(c)

    log.info(
        f"Interpolated {missing_list.shape[0]} missing values. Interpolation level counts: { {k: level_counts[k] for k in sorted(level_counts.keys())} }"
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
                    distr_attr = distr_attr + ["std-within"]
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
                prop_model_dict[prop][src][tgt] = attr_dict

    # Create model
    model = model_types.ConnPropsModel(
        src_types=m_types[0], tgt_types=m_types[1], prop_stats=prop_model_dict
    )
    log.debug("Model description:\n%s", model)

    return model


def plot(
    out_dir, syns_per_conn_data, conn_prop_data, m_types, syn_props, model, **_
):  # pragma: no cover
    """Visualize data vs. model."""
    model_params = model.get_param_dict()
    prop_names = model.get_prop_names()

    # Plot data vs. model: property maps
    def get_model_stat(stat, m_params):
        """Get distribution statistic (if existing) or derive from other model paramters, if possible."""
        val = np.nan
        if stat in m_params:  # Return existing stat. parameter
            val = m_params[stat]
        else:
            if (
                stat == "mean" and "val" in m_params and "p" in m_params
            ):  # Derive mean from discrete values/probabilities
                val = np.sum(np.array(m_params["p"] * np.array(m_params["val"])))
            elif (
                stat == "std" and "val" in m_params and "p" in m_params
            ):  # Derive std from discrete values/probabilities
                m = np.sum(np.array(m_params["p"] * np.array(m_params["val"])))
                val = np.sqrt(
                    np.sum(np.array(m_params["p"]) * (np.array(m_params["val"]) - m) ** 2)
                )
            elif (
                stat == "min" and "val" in m_params and "p" in m_params
            ):  # Derive min from discrete values/probabilities
                val = np.min(np.array(m_params["val"])[np.array(m_params["p"]) > 0.0])
            elif (
                stat == "max" and "val" in m_params and "p" in m_params
            ):  # Derive max from discrete values/probabilities
                val = np.max(np.array(m_params["val"])[np.array(m_params["p"]) > 0.0])
        return val

    title_str = ["Data", "Model"]
    for stat_sel in ["mean", "std"]:
        for pidx, p in enumerate(prop_names):
            plt.figure(figsize=(8, 3), dpi=300)
            if pidx < conn_prop_data[stat_sel].shape[2]:
                data_stat_sel = conn_prop_data[stat_sel][:, :, pidx]
            else:
                data_stat_sel = syns_per_conn_data[stat_sel]
            model_stat_sel = np.full((len(m_types[0]), len(m_types[1])), np.nan)
            for sidx, s in enumerate(m_types[0]):
                for tidx, t in enumerate(m_types[1]):
                    model_stat_sel[sidx, tidx] = get_model_stat(
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

    # Plot data vs. model: Distribution histogram examples (generative model)
    N = 1000  # Number of samples
    conn_counts = [
        [
            np.sum(syns_per_conn_data["hist"][sidx, tidx][0])
            if (
                hasattr(syns_per_conn_data["hist"][sidx, tidx], "__iter__")
                and len(syns_per_conn_data["hist"][sidx, tidx]) > 0
            )
            else 0
            for sidx in range(len(m_types[0]))
        ]
        for tidx in range(len(m_types[1]))
    ]
    max_pathways = np.where(
        np.array(conn_counts) == np.max(conn_counts)
    )  # Select pathway(s) with maximum number of connections (i.e., most robust statistics)
    sidx, tidx = [
        max_pathways[i][0] for i in range(len(max_pathways))
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
            [model.draw(prop_name=p, src_type=src, tgt_type=tgt) for n in range(N)]
        )
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
            label=f"Model (N={N})",
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
