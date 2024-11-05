# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for building synaptic delay models"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from sklearn.linear_model import LinearRegression

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_node_ids, get_edges_population, get_cv_data


def extract(
    circuit,
    bin_size_um,
    max_range_um=None,
    sel_src=None,
    sel_dest=None,
    sample_size=None,
    edges_popul_name=None,
    CV_dict=None,
    **_,
):
    """Extracts distance-dependent synaptic delays between samples of neurons.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        bin_size_um (float): Distance bin size in um
        max_range_um (float): Maximum distance range in um to consider
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        sample_size (int): Size of random subsample of data to extract data from
        edges_popul_name (str): Name of SONATA egdes population to extract data from
        CV_dict (dict): Optional cross-validation dictionary, containing "n_folds" (int), "fold_idx" (int), "training_set" (bool) keys; will be automatically provided by the framework if "CV_folds" are specified

    Returns:
        dict: Dictionary containing the extracted data elements
    """
    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)

    if sample_size is None or sample_size <= 0:
        sample_size = np.inf  # Select all nodes
    sample_size_src = min(sample_size, len(node_ids_src))
    sample_size_dest = min(sample_size, len(node_ids_dest))
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

    # Extract distance/delay values
    edges_table = edges.pathway_edges(
        source=node_ids_src_sel,
        target=node_ids_dest_sel,
        properties=[
            "@source_node",
            "delay",
            "afferent_center_x",
            "afferent_center_y",
            "afferent_center_z",
        ],
    )

    log.debug(
        f"Extracting delays from {edges_table.shape[0]} synapses (sel_src={sel_src}, sel_dest={sel_dest}, sample_size={sample_size} neurons, CV_dict={CV_dict})"
    )

    src_pos = src_nodes.positions(
        edges_table["@source_node"].to_numpy()
    ).to_numpy()  # Soma position of pre-synaptic neuron
    tgt_pos = edges_table[
        ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
    ].to_numpy()  # Synapse position on post-synaptic dendrite
    src_tgt_dist = np.sqrt(np.sum((tgt_pos - src_pos) ** 2, 1))
    src_tgt_delay = edges_table["delay"].to_numpy()

    # Extract distance-dependent delays
    if max_range_um is None:
        max_range_um = np.max(src_tgt_dist)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um
    dist_delays_mean = np.full(num_bins, np.nan)
    dist_delays_std = np.full(num_bins, np.nan)
    dist_count = np.zeros(num_bins).astype(int)

    log.debug("Extracting distance-dependent synaptic delays...")
    pbar = progressbar.ProgressBar()
    for idx in pbar(range(num_bins)):
        d_sel = np.logical_and(
            src_tgt_dist >= dist_bins[idx],
            (
                (src_tgt_dist < dist_bins[idx + 1])
                if idx < num_bins - 1
                else (src_tgt_dist <= dist_bins[idx + 1])
            ),
        )  # Including last edge
        dist_count[idx] = np.sum(d_sel)
        if dist_count[idx] > 0:
            dist_delays_mean[idx] = np.mean(src_tgt_delay[d_sel])
            dist_delays_std[idx] = np.std(src_tgt_delay[d_sel])

    return {
        "dist_bins": dist_bins,
        "dist_delays_mean": dist_delays_mean,
        "dist_delays_std": dist_delays_std,
        "dist_count": dist_count,
        "dist_delay_min": np.nanmin(src_tgt_delay),
    }


def build(dist_bins, dist_delays_mean, dist_delays_std, dist_delay_min, bin_size_um, **_):
    """Fits a linear distance-dependent synaptic delay model of type ``LinDelayModel`` to the data.

    Args:
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract`
        dist_delays_mean (numpy.ndarray): Delay mean for all bins, as returned by :func:`extract`
        dist_delays_std (numpy.ndarray): Delay std for all bins, as returned by :func:`extract`
        dist_delay_min (float): Overall delay minimum, as returned by :func:`extract`
        bin_size_um (float): Distance bin size in um

    Returns:
        connectome_manipulator.model_building.model_types.LinDelayModel: Fitted linear distance-dependent delay model
    """
    log.log_assert(np.all((np.diff(dist_bins) - bin_size_um) < 1e-12), "ERROR: Bin size mismatch!")
    bin_offset = 0.5 * bin_size_um

    # Mean delay model (linear)
    X = np.array(dist_bins[:-1][np.isfinite(dist_delays_mean)] + bin_offset, ndmin=2).T
    y = dist_delays_mean[np.isfinite(dist_delays_mean)]
    dist_delays_mean_fit = LinearRegression().fit(X, y)
    delay_mean_coeff_a = dist_delays_mean_fit.intercept_
    delay_mean_coeff_b = dist_delays_mean_fit.coef_[0]

    # Std delay model (const)
    delay_std = np.nanmean(dist_delays_std)

    # Min delay model (const)
    delay_min = dist_delay_min

    # Create model
    model = model_types.LinDelayModel(
        delay_mean_coeff_a=float(delay_mean_coeff_a),
        delay_mean_coeff_b=float(delay_mean_coeff_b),
        delay_std=float(delay_std),
        delay_min=float(delay_min),
    )
    log.debug("Model description:\n%s", model)

    return model


def plot(
    out_dir, dist_bins, dist_delays_mean, dist_delays_std, dist_count, model, **_
):  # pragma: no cover
    """Visualizes extracted data vs. actual model output.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        dist_bins (numpy.ndarray): Distance bin edges, as returned by :func:`extract`
        dist_delays_mean (numpy.ndarray): Delay mean for all bins, as returned by :func:`extract`
        dist_delays_std (numpy.ndarray): Delay std for all bins, as returned by :func:`extract`
        dist_count (numpy.ndarray): Number of data elemets in each bin, as returned by :func:`extract`
        model (connectome_manipulator.model_building.model_types.LinDelayModel): Fitted linear distance-dependent delay model, as returned by :func:`build`
    """
    bin_width = np.diff(dist_bins[:2])[0]

    model_params = model.get_param_dict()
    mean_model_str = f'f(x) = {model_params["delay_mean_coeff_b"]:.3f} * x + {model_params["delay_mean_coeff_a"]:.3f}'
    std_model_str = f'f(x) = {model_params["delay_std"]:.3f}'
    min_model_str = f'f(x) = {model_params["delay_min"]:.3f}'

    # Draw figure
    model_kwargs = dict(zip(("src_type", "tgt_type"), model.default_types)) | {
        "distance": dist_bins
    }
    plt.figure(figsize=(8, 4), dpi=300)
    plt.bar(
        dist_bins[:-1] + 0.5 * bin_width,
        dist_delays_mean,
        width=0.95 * bin_width,
        facecolor="tab:blue",
        label=f"Data mean: N = {np.sum(dist_count)} synapses",
    )
    plt.bar(
        dist_bins[:-1] + 0.5 * bin_width,
        dist_delays_std,
        width=0.5 * bin_width,
        facecolor="tab:red",
        label=f"Data std: N = {np.sum(dist_count)} synapses",
    )
    plt.plot(
        dist_bins,
        model.get_mean(**model_kwargs),
        "--",
        color="tab:brown",
        label="Model mean: " + mean_model_str,
    )
    plt.plot(
        dist_bins,
        model.get_std(**model_kwargs),
        "--",
        color="tab:olive",
        label="Model std: " + std_model_str,
    )
    plt.plot(
        dist_bins,
        model.get_min(**model_kwargs),
        "--",
        color="tab:gray",
        label="Model min: " + min_model_str,
    )
    plt.xlim((dist_bins[0], dist_bins[-1]))
    plt.xlabel("Distance [um]")
    plt.ylabel("Delay [ms]")
    plt.title("Distance-dependent synaptic delays", fontweight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))

    # Add second axis with bin counts
    count_color = "tab:orange"
    ax_count = plt.gca().twinx()
    ax_count.set_yscale("log")
    ax_count.step(dist_bins, np.concatenate((dist_count[:1], dist_count)), color=count_color)
    ax_count.set_ylabel("Count", color=count_color)
    ax_count.tick_params(axis="y", which="both", colors=count_color)
    ax_count.spines["right"].set_color(count_color)

    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Visualize model output (generative model)
    dist_centers = [np.mean(dist_bins[:2]), np.mean(dist_bins), np.mean(dist_bins[-2:])]
    N = 1000  # Number of samples
    plt.figure(figsize=(8, 4), dpi=300)
    for didx, d in enumerate(dist_centers):
        plt.subplot(1, len(dist_centers), didx + 1)
        plt.hist(model.apply(distance=np.full(N, d)), bins=50)
        plt.ylim(plt.ylim())  # Freeze limit
        plt.title(f"{d:.0f} um")
        plt.xlabel("Delay [ms]")
        plt.ylabel("Count")
    plt.suptitle("Delay distributions")
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, "model_output.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)
