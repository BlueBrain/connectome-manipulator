"""TODO: improve description"""
# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import numpy as np
from scipy.spatial import distance_matrix

from connectome_manipulator import log
from connectome_manipulator.access_functions import get_node_ids


def apply(edges_table, nodes, aux_dict, sel_grp1, sel_grp2, R, amount_pct=100.0):
    """Rewiring (interchange) of axons between pairs of neurons belonging to disjoint groups within given distance range R."""
    log.log_assert(R > 0.0, "R must be larger than 0um!")
    log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")

    pair_gids = aux_dict.get(
        "pair_gids", None
    )  # Load GID mapping from earlier split iteration, if existing
    if pair_gids is None:
        log.info("Sampling pairs of neurons for axon rewiring...")

        gids1 = get_node_ids(nodes[0], sel_grp1)
        gids2 = get_node_ids(nodes[0], sel_grp2)
        log.log_assert(
            len(np.intersect1d(gids1, gids2)) == 0, "Overlapping groups of neurons not supported!"
        )

        sclass1 = nodes[0].get(gids1, properties="synapse_class").unique()
        sclass2 = nodes[0].get(gids2, properties="synapse_class").unique()
        log.log_assert(
            (len(sclass1) == len(sclass2) == 1) and (sclass1[0] == sclass2[0]),
            "Synapse class mismatch!",
        )

        # Compute distance matrix
        nrn_pos1 = nodes[0].positions(gids1)
        nrn_pos2 = nodes[0].positions(gids2)
        dist_mat = distance_matrix(nrn_pos1.to_numpy(), nrn_pos2.to_numpy())

        log.debug(
            f"Upper limit of possible pairs: {np.min(dist_mat.shape)} (|grp1|={len(gids1)}, |grp2|={len(gids2)})"
        )

        # Thresholded distance matrix
        dist_mat_R = np.ones_like(dist_mat).astype(bool)
        dist_mat_R[dist_mat > R] = False
        del dist_mat

        # Remove cells that cannot be rewired (no potential neighbors in vicinity)
        sel_idx1 = np.sum(dist_mat_R, 1) > 0
        sel_idx2 = np.sum(dist_mat_R, 0) > 0

        dist_mat_R = dist_mat_R[sel_idx1, :]
        dist_mat_R = dist_mat_R[:, sel_idx2]

        log.debug(f"Radius-dependent upper limit: {np.min(dist_mat_R.shape)} (R={R}um)")

        # Assure that first dimension is always the lower one (= the one used to run pair selection)
        if dist_mat_R.shape[0] > dist_mat_R.shape[1]:
            dist_mat_R = dist_mat_R.T
            gids = [gids2[sel_idx2], gids1[sel_idx1]]
        else:
            gids = [gids1[sel_idx1], gids2[sel_idx2]]
        del gids1, gids2

        # Sample pairs n1-n2 of neurons to interchange axons
        dist_mat_R_choices = np.copy(dist_mat_R)  # To keep track of possible choices in each step
        target_count = np.round(dist_mat_R.shape[0] * amount_pct / 100).astype(int)
        pair_samples = []
        for n1 in np.random.permutation(dist_mat_R.shape[0]):
            if (
                len(pair_samples) == target_count
            ):  # Target count reached...FINISHING [Note: due to random sampling, it is possible that target count won't be reached exactly]
                break
            n2_all = np.nonzero(dist_mat_R_choices[n1, :])[0]
            if len(n2_all) == 0:  # No choices available for n1...SKIPPING
                continue
            n2 = np.random.choice(n2_all)  # Randomly select one of the choices...
            pair_samples.append([n1, n2])  # ...and add to list of selected pairs
            dist_mat_R_choices[:, n2] = False  # n2 already taken, don't reuse!!

        pair_samples = np.array(pair_samples)
        if pair_samples.size > 0:
            pair_gids = np.array([gids[0][pair_samples[:, 0]], gids[1][pair_samples[:, 1]]]).T
        else:
            pair_gids = np.array([])
        log.log_assert(
            len(np.unique(pair_gids)) == pair_gids.size, "Duplicates in gid pairs found!"
        )
        aux_dict.update(
            {"pair_gids": pair_gids}
        )  # Save GID mapping, to be reused in subsequent split iterations

        log.info(f"Actually selected GID pairs: {pair_gids.shape[0]} (amount={amount_pct}%)")

    # Rewire axons (interchange source ids in edges table)
    for id1, id2 in pair_gids:
        src_idx1 = edges_table["@source_node"] == id1
        src_idx2 = edges_table["@source_node"] == id2
        edges_table.loc[src_idx1, "@source_node"] = id2
        edges_table.loc[src_idx2, "@source_node"] = id1

    return edges_table
