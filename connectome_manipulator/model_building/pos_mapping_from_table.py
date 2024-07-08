# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for position mapping model creation, consisting of three basic functions:

- extract(...): Loads positions of a given nodes population based on some alternative
                coordinate system from a pre-computed position table (pandas DataFrame)
- build(...): Build flat space position mapping (LUT) model of type "PosMapModel" from the data
- plot(...): Visualizes data vs. model output
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
from scipy.spatial import distance_matrix

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_node_ids


def extract(
    circuit,
    pos_file,
    coord_names,
    coord_scale=None,
    nodes_pop_name=None,
    nodes_spec=None,
    zero_based_indexing=False,
    gid_column=None,
    **_,
):
    """Loads pre-computed position mapping of a given nodes population."""
    # Get neuron GIDs
    if nodes_pop_name is None:
        log.log_assert(
            len(circuit.nodes.population_names) == 1,
            f"ERROR: Nodes population could not be determined (found {circuit.nodes.population_names})!",
        )
        nodes_pop_name = circuit.nodes.population_names[0]
        log.debug(f'Loading nodes population "{nodes_pop_name}"')
    nodes = circuit.nodes[nodes_pop_name]

    nrn_ids = get_node_ids(nodes, nodes_spec)
    nrn_pos = nodes.positions(nrn_ids).to_numpy()  # Just for visualization
    nrn_lay = nodes.get(nrn_ids, properties="layer")  # Just for visualization
    if zero_based_indexing:
        idx_offset = 0  # Zero-based indexing in pos_file (SNAP/SONATA GIDs)
    else:
        idx_offset = 1  # One-based indexing in pos_file (BluePy GIDs)

    # Load position table
    log.log_assert(os.path.exists(pos_file), f'ERROR: Position file "{pos_file}" not found!')
    file_format = os.path.splitext(pos_file)[1]
    if file_format == ".feather":
        nrn_table = pd.read_feather(pos_file)
    else:
        log.log_assert(False, f'ERROR: "{file_format}" format not supported!')

    # pylint: disable=E0606
    log.debug(f'Loaded position table for {nrn_table.shape[0]} neurons from "{pos_file}"')

    # Assign mapped positions
    if gid_column is None:  # Use index column
        tab_gids = nrn_table.index.to_numpy()
    else:  # Use specified gid_column
        log.log_assert(
            gid_column in nrn_table.columns, f'ERROR: GID column "{gid_column}" not found!'
        )
        tab_gids = nrn_table[gid_column].to_numpy()

    log.log_assert(np.all(np.isin(nrn_ids + idx_offset, tab_gids)), "ERROR: Neuron IDs mismatch!")
    log.log_assert(
        np.all(np.isin(coord_names, nrn_table.columns)),
        "ERROR: Coordinate name(s) not found in position table!",
    )
    map_pos = nrn_table.loc[np.isin(tab_gids, nrn_ids + idx_offset), coord_names].to_numpy()

    # Apply scale
    if coord_scale is not None:
        log.log_assert(len(coord_names) == len(coord_scale), "ERROR: Coordinate scale mismatch!")
        map_pos = map_pos * coord_scale

    log.info(
        f'Loaded {", ".join(coord_names)} coordinates for {map_pos.shape[0]} neurons from position table (coord_scale={coord_scale})'
    )

    return {"nrn_ids": nrn_ids, "map_pos": map_pos, "nrn_pos": nrn_pos, "nrn_lay": nrn_lay}


def build(nrn_ids, coord_names, map_pos, model_coord_names=None, **_):
    """Build position mapping model."""
    if model_coord_names is None:
        model_coord_names = coord_names  # Same model coord names as input coord names
    else:
        log.log_assert(
            len(model_coord_names) == len(coord_names), "ERROR: Model coordinate names mismatch!"
        )

    map_pos_table = pd.DataFrame(map_pos, index=nrn_ids, columns=model_coord_names)
    log.log_assert(np.all(np.isfinite(map_pos_table)), "ERROR: Invalid mapped positions found!")

    # Create model
    model = model_types.PosMapModel(pos_table=map_pos_table)
    log.debug("Model description:\n%s", model)

    return model


def plot(out_dir, nrn_ids, nrn_pos, nrn_lay, model, **_):  # pragma: no cover
    """Visualize data vs. model."""
    nrn_pos_model = model.apply(gids=nrn_ids)

    # Cell positions in 3D original vs. mapped space
    model_coord_names = model.get_coord_names()
    num_layers = len(np.unique(nrn_lay))
    lay_colors = plt.get_cmap("jet")(np.linspace(0, 1, num_layers))
    views_3d = [[90, 0], [0, 0]]
    pos_list = [nrn_pos, nrn_pos_model]
    lbl_list = ["Original space (data)", "Mapped space (model)"]
    coord_list = [["x [$\\mu$m]", "y [$\\mu$m]", "z [$\\mu$m]"], model_coord_names]
    fig = plt.figure(figsize=(10, 3 * len(views_3d)), dpi=300)
    plt.gcf().patch.set_facecolor("w")
    for vidx, v in enumerate(views_3d):
        for pidx, (pos, lbl, coo) in enumerate(zip(pos_list, lbl_list, coord_list)):
            if len(coo) == 2:
                if vidx == 0:  # Only plot once, since only single 2D view
                    ax = fig.add_subplot(
                        len(views_3d), len(pos_list), vidx * len(pos_list) + pidx + 1
                    )
                    for lidx in range(num_layers):
                        pos_sel = pos[nrn_lay == lidx + 1, :]
                        plt.plot(
                            pos_sel[:, 0],
                            pos_sel[:, 1],
                            ".",
                            color=lay_colors[lidx, :],
                            markersize=1.0,
                            alpha=0.5,
                            label=f"L{lidx + 1}",
                        )
                    ax.set_xlabel(coo[0])
                    ax.set_ylabel(coo[1])
            elif len(coo) == 3:
                ax = fig.add_subplot(
                    len(views_3d), len(pos_list), vidx * len(pos_list) + pidx + 1, projection="3d"
                )
                for lidx in range(num_layers):
                    pos_sel = pos[nrn_lay == lidx + 1, :]
                    plt.plot(
                        pos_sel[:, 0],
                        pos_sel[:, 1],
                        pos_sel[:, 2],
                        ".",
                        color=lay_colors[lidx, :],
                        markersize=1.0,
                        alpha=0.5,
                        label=f"L{lidx + 1}",
                    )
                ax.view_init(*v)
                ax.set_xlabel(coo[0])
                ax.set_ylabel(coo[1])
                ax.set_zlabel(coo[2])
            else:
                log.debug(f'Only 2D/3D plotting supported! Skipping "{lbl}"...')
                continue
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
            if vidx == 0:
                plt.title(lbl + f"\n[N={len(nrn_ids)}cells]")
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model_positions.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Cell distances in atlas vs. flat space
    max_plot = 10000
    if len(nrn_ids) > max_plot:
        log.debug("Using subsampling for distance plots!")
        nrn_sel = np.random.choice(len(nrn_ids), max_plot)
        nrn_ids = nrn_ids[nrn_sel]
        nrn_pos = nrn_pos[nrn_sel, :]
        nrn_pos_model = nrn_pos_model[nrn_sel, :]

    dist_mat_data = distance_matrix(nrn_pos, nrn_pos)
    dist_mat_model = distance_matrix(nrn_pos_model, nrn_pos_model)

    triu_idx = np.triu_indices(len(nrn_ids), 1)
    dist_val_data = dist_mat_data[triu_idx]
    dist_val_model = dist_mat_model[triu_idx]

    dist_max = max(dist_val_data, dist_val_model)
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(dist_val_data, dist_val_model, "b.", alpha=0.1, markersize=1.0, markeredgecolor="none")
    plt.plot([0, dist_max], [0, dist_max], "k--")
    plt.xlim((0, dist_max))
    plt.ylim((0, dist_max))
    plt.grid(True)
    plt.xlabel("Distance in original space (data) [$\\mu$m]")
    plt.ylabel("Distance in mapped space (model)")
    plt.title(f"Cell distances in atlas vs. flat space [N={len(nrn_ids)}cells]")
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model_distances.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)

    # Nearest neighbors in atlas vs. flat space
    NN_mat_data = np.argsort(dist_mat_data, axis=1)
    NN_mat_model = np.argsort(dist_mat_model, axis=1)

    num_NN_list = list(range(1, 30, 1))
    NN_match = np.full(len(num_NN_list), np.nan)

    log.debug("Computing nearest neighbors in atlas vs. flat space...")
    pbar = progressbar.ProgressBar()
    for nidx in pbar(range(len(num_NN_list))):
        num_NN = num_NN_list[nidx]
        NN_match[nidx] = np.mean(
            [
                len(np.intersect1d(NN_mat_data[i, 1 : 1 + num_NN], NN_mat_model[i, 1 : 1 + num_NN]))
                / num_NN
                for i in range(len(nrn_ids))
            ]
        )

    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(num_NN_list, NN_match, ".-")
    plt.grid(True)
    plt.ylim((0, 1))
    plt.xlabel("#Nearest neighbors")
    plt.ylabel("Mean match")
    plt.title(f"Nearest neighbors in atlas vs. flat space [N={len(nrn_ids)}cells]")

    out_fn = os.path.abspath(os.path.join(out_dir, "data_vs_model_neighbors.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)
