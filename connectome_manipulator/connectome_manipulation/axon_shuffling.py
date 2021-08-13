'''TODO: improve description'''
# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging

import numpy as np
from scipy.spatial import distance_matrix

from connectome_manipulator import log


def apply(edges_table, nodes, aux_dict, sel_grp, R, amount_pct=100.0):
    """Shuffling of axons between pairs of neurons within given group and distance range R."""
    log.log_assert(R > 0.0, 'R must be larger than 0um!')
    log.log_assert(0.0 <= amount_pct <= 100.0, 'amount_pct out of range!')

    pair_gids = aux_dict.get('pair_gids', None) # Load GID mapping from earlier split iteration, if existing
    if pair_gids is None:
        logging.info('INFO: Sampling pairs of neurons for axon shuffling...')

        gids = nodes[0].ids(sel_grp)

        sclass = nodes[0].get(gids, properties='synapse_class').unique()
        log.log_assert(len(sclass) == 1, 'Multiple synapse classes found!')

        # Compute distance matrix
        nrn_pos = nodes[0].positions(gids).to_numpy()
        dist_mat = distance_matrix(nrn_pos, nrn_pos)

        logging.info(f'Upper limit of possible pairs: {np.floor(dist_mat.shape[0] / 2).astype(int)} (|grp|={len(gids)})')

        # Thresholded distance matrix
        dist_mat_R = np.ones_like(dist_mat).astype(bool)
        dist_mat_R[dist_mat > R] = False
        np.fill_diagonal(dist_mat_R, False)
        del dist_mat

        # Remove cells that cannot be rewired (no potential neighbors in vicinity)
        sel_idx = np.sum(dist_mat_R, 0) > 0

        dist_mat_R = dist_mat_R[sel_idx, :]
        dist_mat_R = dist_mat_R[:, sel_idx]

        logging.info(f'Radius-dependent upper limit: {np.floor(dist_mat_R.shape[0] / 2).astype(int)} (R={R}um)')

        # Sample pairs n1-n2 of neurons to interchange axons
        dist_mat_R_choices = np.copy(dist_mat_R) # To keep track of possible choices in each step
        target_count = np.round(0.5 * dist_mat_R.shape[0] * amount_pct / 100).astype(int)
        pair_samples = []
        for n1 in np.random.permutation(dist_mat_R.shape[0]):
            if len(pair_samples) == target_count: # Target count reached...FINISHING [Note: due to random sampling, it is possible that target count won't be reached exactly]
                break
            n2_all = np.nonzero(dist_mat_R_choices[n1, :])[0]
            if len(n2_all) == 0: # No choices available for n1...SKIPPING
                continue
            n2 = np.random.choice(n2_all) # Randomly select one of the choices...
            pair_samples.append([n1, n2]) # ...and add to list of selected pairs
            dist_mat_R_choices[:, n2] = False # n2 already taken, don't reuse!!
            dist_mat_R_choices[n2, :] = False # (apply symmetrically)
            dist_mat_R_choices[:, n1] = False # n1 already taken, don't reuse!!
            # dist_mat_R_choices[n1, :] = False # (apply symmetrically) => NOT REQUIRED: n1 indices used only once!
        pair_samples = np.array(pair_samples)
        if pair_samples.size > 0:
            pair_gids = gids[pair_samples]
        else:
            pair_gids = np.array([])
        log.log_assert(len(np.unique(pair_gids)) == pair_gids.size, 'Duplicates in gid pairs found!')
        aux_dict.update({'pair_gids': pair_gids}) # Save GID mapping, to be reused in subsequent split iterations

        logging.info(f'Actually selected GID pairs: {pair_gids.shape[0]} (amount={amount_pct}%)')

    # Rewire axons (interchange source ids in edges table)
    for id1, id2 in pair_gids:
        src_idx1 = edges_table['@source_node'] == id1
        src_idx2 = edges_table['@source_node'] == id2
        edges_table.loc[src_idx1, '@source_node'] = id2
        edges_table.loc[src_idx2, '@source_node'] = id1

    return edges_table
