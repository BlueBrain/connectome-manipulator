import os

import numpy as np
from numpy.testing import assert_allclose

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
import connectome_manipulator.connectome_manipulation.syn_removal as test_module


def test_apply():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    node_ids = [*range(10)]
    nodes = [edges.source, edges.target]
    edges_table = edges.afferent_edges(node_ids, properties=edges.property_names)

    # Check that corrent number of synapses is removed
    total_syns = len(edges_table)
    for pct in range(0, 101, 25):
        res = test_module.apply(edges_table, nodes, None, amount_pct=pct)
        assert len(res) == int(total_syns * (100 - pct) / 100)

    # Check that only given ids are considered
    src_id = 0
    res = test_module.apply(
        edges_table, nodes, None, sel_src={*node_ids} - {src_id}, amount_pct=100
    )

    assert np.all(res["@source_node"] == src_id)

    tgt_id = 9
    res = test_module.apply(
        edges_table, nodes, None, sel_dest={*node_ids} - {tgt_id}, amount_pct=100
    )

    assert np.all(res["@target_node"] == tgt_id)

    # Check when both sel_src and sel_dest are given
    res = test_module.apply(
        edges_table,
        nodes,
        None,
        sel_src={*node_ids} - {src_id},
        sel_dest={*node_ids} - {tgt_id},
        amount_pct=100,
    )

    assert np.all(np.logical_or(res["@source_node"] == src_id, res["@target_node"] == tgt_id))

    # Check that one synapse per connection is left when keep-cons=True
    total_conns = len(np.unique(edges_table[["@source_node", "@target_node"]], axis=0))
    res = test_module.apply(
        edges_table, nodes, None, amount_pct=100, keep_conns=True, rescale_gsyn=True
    )
    res_conn = np.unique(res[["@source_node", "@target_node"]], axis=0)
    assert len(res_conn) == total_conns
    assert len(res) == total_conns

    # Check that conductance per connection is maintained with rescale_gsyn=True
    assert_allclose(
        res.groupby(["@source_node", "@target_node"])["conductance"].sum(),
        edges_table.groupby(["@source_node", "@target_node"])["conductance"].sum(),
    )
