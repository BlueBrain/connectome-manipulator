import os

import numpy as np
from numpy.testing import assert_array_equal

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
import connectome_manipulator.access_functions as test_module


def test_get_node_ids():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = circuit.nodes[circuit.nodes.population_names[0]]

    # Check direct node set access
    for node_set in list(circuit.node_sets.content.keys()):
        ids = test_module.get_node_ids(nodes, node_set)
        assert_array_equal(ids, nodes.ids(node_set))

    # Check node access based on properties (layer)
    layers = list(np.unique(nodes.get(properties="layer")))
    for lay in layers:
        ids = test_module.get_node_ids(nodes, {"layer": lay})
        assert_array_equal(ids, nodes.ids({"layer": lay}))

    # Check combination of node set and properties (layer)
    for node_set in list(circuit.node_sets.content.keys()):
        for lay in layers:
            ids = test_module.get_node_ids(nodes, {"node_set": node_set, "layer": lay})
            ref_ids = np.intersect1d(nodes.ids(node_set), nodes.ids({"layer": lay}))
            assert_array_equal(ids, ref_ids)


def get_edges_population():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    popul_names = circuit.edges.population_names

    # Check selecting single (default) population
    edges = test_module.get_edges_population()
    assert edges is circuit.edges[popul_names[0]]

    # Check selecting population by name (in case of single population)
    edges = test_module.get_edges_population(popul_names[0])
    assert edges is circuit.edges[popul_names[0]]

    # Check selecting population by name (in case of multiple populations)
    # TODO
