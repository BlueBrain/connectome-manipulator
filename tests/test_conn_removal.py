import os

from mock import Mock
import numpy as np
import pandas as pd

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
import connectome_manipulator.connectome_manipulation.conn_removal as test_module


def test_apply():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = circuit.edges[circuit.edges.population_names[0]]
    node_ids = [*range(10)]
    nodes = [edges.source, edges.target]
    edges_table = edges.afferent_edges(node_ids, properties=edges.property_names)

    # Test that given percentage of connections is removed
    total_conns = len(np.unique(edges_table[['@source_node', '@target_node']], axis=0))
    for pct in range(0, 101, 25):
        res = test_module.apply(edges_table, nodes, None, amount_pct=pct)
        res_conns = len(np.unique(res[['@source_node', '@target_node']], axis=0))
        assert res_conns == int(total_conns * (100 - pct) / 100)

    # Check that only given ids are considered
    src_id = 0
    res = test_module.apply(edges_table, nodes, None,
                            sel_src={*node_ids} - {src_id},
                            amount_pct=100)

    assert np.all(res['@source_node'] == src_id)

    tgt_id = 9
    res = test_module.apply(edges_table, nodes, None,
                            sel_dest={*node_ids} - {tgt_id},
                            amount_pct=100)

    assert np.all(res['@target_node'] == tgt_id)

    # Check when both sel_src and sel_dest are given
    res = test_module.apply(edges_table, nodes, None,
                            sel_src={*node_ids} - {src_id},
                            sel_dest={*node_ids} - {tgt_id},
                            amount_pct=100)

    assert np.all(np.logical_or(res['@source_node'] == src_id,
                                res['@target_node'] == tgt_id))

    # Check that connection size filtering works for minimum connection size
    res = test_module.apply(edges_table, nodes, None,
                            min_syn_per_conn=2,
                            amount_pct=100)
    _, n_syn_conn = np.unique(res[['@source_node', '@target_node']],
                              axis=0, return_counts=True)
    assert np.all(n_syn_conn < 2)

    # Check that connection size filtering works for maximum connection size
    res = test_module.apply(edges_table, nodes, None,
                            max_syn_per_conn=2,
                            amount_pct=100)
    _, n_syn_conn = np.unique(res[['@source_node', '@target_node']],
                              axis=0, return_counts=True)
    assert np.all(n_syn_conn > 2)

    # Check that connection size filtering works when both are enabled
    res = test_module.apply(edges_table, nodes, None,
                            min_syn_per_conn=2,
                            max_syn_per_conn=2,
                            amount_pct=100)
    _, n_syn_conn = np.unique(res[['@source_node', '@target_node']],
                              axis=0, return_counts=True)
    assert np.all(n_syn_conn != 2)

    # Check with empty selection, i.e. nothing to be removed
    res = test_module.apply(edges_table, nodes, None,
                            min_syn_per_conn=np.inf,
                            amount_pct=100)
    _, n_syn_conn = np.unique(res[['@source_node', '@target_node']],
                              axis=0, return_counts=True)
    assert res.equals(edges_table)
