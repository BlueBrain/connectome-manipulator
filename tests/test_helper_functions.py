import os

import numpy as np
from bluepysnap import Circuit
from numpy.testing import assert_approx_equal, assert_array_equal

from utils import TEST_DATA_DIR
import connectome_manipulator.connectome_manipulation.helper_functions as test_module


def test_get_gsyn_sum_per_conn():
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    node_ids = [*range(10)]
    edges_table = edges.afferent_edges(node_ids, properties=sorted(edges.property_names))
    gsyn_table = test_module.get_gsyn_sum_per_conn(edges_table, node_ids, node_ids)

    edge_groups = edges_table.groupby(["@source_node", "@target_node"])
    for i, sgid in enumerate(node_ids):
        for j, tgid in enumerate(node_ids):
            if (sgid, tgid) in edge_groups.groups:
                assert_approx_equal(
                    gsyn_table[i, j], edge_groups.get_group((sgid, tgid)).conductance.sum()
                )


def test_rescale_gsyn_per_conn():
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]

    # Basically test rescaling the conductance of every connection by a factor of 2
    ratio = 2
    node_ids = [*range(10)]
    edges_table_orig = edges.afferent_edges(node_ids, properties=sorted(edges.property_names))

    edges_table = edges_table_orig.copy()
    gsyn_table = test_module.get_gsyn_sum_per_conn(edges_table, node_ids, node_ids)
    gsyn_table_manip = np.array(gsyn_table) * ratio

    test_module.rescale_gsyn_per_conn(edges_table, node_ids, node_ids, gsyn_table, gsyn_table_manip)
    assert_array_equal(edges_table.conductance, edges_table_orig.conductance / ratio)

    # Now the same but only for a subset
    node_ids = [0, 1]

    edges_table = edges_table_orig.copy()
    gsyn_table = test_module.get_gsyn_sum_per_conn(edges_table, node_ids, node_ids)
    gsyn_table_manip = np.array(gsyn_table) * ratio

    test_module.rescale_gsyn_per_conn(edges_table, node_ids, node_ids, gsyn_table, gsyn_table_manip)

    # Check that only the connections defined by node_ids are changed
    mask = np.logical_and(
        np.in1d(edges_table["@source_node"], node_ids),
        np.in1d(edges_table["@target_node"], node_ids),
    )

    assert_array_equal(edges_table.conductance[mask], edges_table_orig.conductance[mask] / ratio)

    assert_array_equal(edges_table.conductance[~mask], edges_table_orig.conductance[~mask])
