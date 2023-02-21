"""Manipulation name: conn_extract

Description: Extraction of a cell target, i.e., keeping only connections within that target.

Note 1: Nodes table is kept unchanged!
Note 2: If no cell target is given, an empty connectome is returned!
"""

import os

import json
import numpy as np

from connectome_manipulator import log


def apply(edges_table, nodes, _aux_dict, target_name=None, node_sets_file=None):
    """Extraction of a cell target as given by target_name

    Extraction keeping only connections within that target (empty connectome if no target_name provided).
    Optionally, a node sets file containing 'node_id' entries can be provided in case
    the cell target is not part of the circuit's intrinsic node sets.
    """
    if target_name is None:
        log.info("No target name provided, returning empty connectome!")
        return edges_table.loc[[]].copy()

    # Load cell targets
    assert nodes[0] is nodes[1], "ERROR: Only one source/target node population supported!"
    # pylint: disable=W0212
    if target_name in nodes[0]._node_sets.content:
        target_gids = nodes[0].ids(target_name)
    else:
        log.log_assert(
            node_sets_file is not None, f'Target "{target_name}" unknown, node sets file required!'
        )

        # Load node sets file
        log.log_assert(
            os.path.exists(node_sets_file), f'Node sets file "{node_sets_file}" not found!'
        )
        with open(node_sets_file, "r") as f:
            node_sets = json.load(f)

        log.log_assert(
            target_name in node_sets.keys(),
            f'Target "{target_name}" not found in "{node_sets_file}"!',
        )
        log.log_assert(
            "node_id" in node_sets[target_name],
            f'Node IDs for target "{target_name}" not found in "{node_sets_file}"!',
        )
        target_gids = node_sets[target_name]["node_id"]
    log.info(f'Found {len(target_gids)} GIDs for target "{target_name}" to extract')

    # Extract connectome
    syn_sel_idx = np.logical_and(
        np.isin(edges_table["@source_node"], target_gids),
        np.isin(edges_table["@target_node"], target_gids),
    )
    edges_table_manip = edges_table[syn_sel_idx].copy()

    # TESTING/DEBUGGING #
    log.log_assert(
        not np.any(edges_table_manip.sum(1) == 0),
        f"Empty edges table entries found!\ntarget_gids={target_gids}\np.sum(syn_sel_idx)={np.sum(syn_sel_idx)}\nedges_table_manip.size={edges_table_manip.size}",
    )
    # ################# #

    return edges_table_manip
