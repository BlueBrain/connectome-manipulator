# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module for building (deterministic) connection probability models based on adjacency matrices"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from connectome_manipulator import log
from connectome_manipulator.connectome_comparison import adjacency
from connectome_manipulator.model_building import model_types


def extract(circuit, sel_src=None, sel_dest=None, edges_popul_name=None, CV_dict=None, **_):
    """Extracts adjacency matrix between selected src/dest neurons.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        edges_popul_name (str): Name of SONATA egdes population to extract data from
        CV_dict (dict): Cross-validation dictionary - Not supported

    Returns:
        dict: Dictionary containing the extracted adjacency matrix and source/target node ids
    """
    log.info(f"Running adjacency data extraction (sel_src={sel_src}, sel_dest={sel_dest})...")
    log.log_assert(CV_dict is None, "ERROR: Cross-validation not supported!")

    adj_dict = adjacency.compute(circuit, sel_src, sel_dest, edges_popul_name)

    return {
        "adj_mat": adj_dict["adj"]["data"].tocsc(),
        "src_node_ids": adj_dict["common"]["src_gids"],
        "tgt_node_ids": adj_dict["common"]["tgt_gids"],
    }


def build(adj_mat, src_node_ids, tgt_node_ids, inverted=False, **_):
    """Builds a (deterministic) connection probability model of type ``ConnProbAdjModel`` from an adjacency matrix (i.e., returning probabilities 0.0 or 1.0 only).

    Args:
        adj_mat (scipy.sparse.csc_matrix): Sparse adjacency matrix with boolean entries (i.e., True...connection, False...no connection)
        src_node_ids (list-like): List of source (pre-synaptic) neuron IDs
        tgt_node_ids (list-like): List of target (post-synaptic) neuron IDs
        inverted (bool): Flag for interpreting the boolean matrix entries in an inverted way (i.e., True...no connection, False...connection)

    Returns:
        connectome_manipulator.model_building.model_types.ConnProbAdjModel: Resulting adjacency model
    """
    log.info(f"Running {'inverted ' if inverted else ''}adjacency model building...")

    log.log_assert(
        adj_mat.dtype == "bool", "ERROR: Adjacency matrix with boolean data type required!"
    )
    log.log_assert(
        adj_mat.format.lower() == "csc", "ERROR: Adjacency matrix in CSC format required!"
    )
    log.log_assert(adj_mat.shape[0] == len(src_node_ids), "ERROR: Source nodes mismatch!")
    log.log_assert(adj_mat.shape[1] == len(tgt_node_ids), "ERROR: Target nodes mismatch!")

    # Prepare data frames
    src_nodes_table = pd.DataFrame(src_node_ids, columns=["src_node_ids"])
    tgt_nodes_table = pd.DataFrame(tgt_node_ids, columns=["tgt_node_ids"])
    rows, cols = adj_mat.nonzero()
    adj_table = pd.DataFrame({"row_ind": rows, "col_ind": cols})

    # Create model
    model = model_types.ConnProbAdjModel(
        src_nodes_table=src_nodes_table,
        tgt_nodes_table=tgt_nodes_table,
        adj_table=adj_table,
        inverted=inverted,
    )
    log.debug("Model description:\n%s", model)

    return model


def plot(out_dir, model, **_):
    """Visualizes the adjacency model.

    Args:
        out_dir (str): Path to output directory where the results figures will be stored
        model (connectome_manipulator.model_building.model_types.ConnProbAdjModel): Adjacency model, as returned by :func:`build`
    """
    log.info("Running adjacency model visualization...")

    # Apply model, i.e., getting connection probabilities between all pairs of neurons
    src_node_ids = model.get_src_nids()
    tgt_node_ids = model.get_tgt_nids()
    p_model = model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids)
    log.log_assert(
        np.array_equal(np.unique(p_model), [0.0, 1.0]), "ERROR: Model output not boolean!"
    )
    res_dict = {"data": csc_matrix(p_model).astype(bool)}

    # Visualize model output
    if model.is_inverted():
        model_str = "Inverted adjacency model"
    else:
        model_str = "Adjacency model"
    plt.figure(figsize=(4, 4))
    adjacency.plot(
        res_dict,
        None,
        fig_title=f"{model_str}\n({len(src_node_ids)}x{len(tgt_node_ids)} neurons, {np.sum(p_model > 0)} connections)",
        vmin=0.0,
        vmax=1.0,
    )
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, "model_output.png"))
    log.info(f"Saving {out_fn}...")
    plt.savefig(out_fn)
