import json
import numpy as np
import os

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
import connectome_manipulator.connectome_manipulation.conn_extraction as test_module


def test_apply():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    node_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(node_ids, properties=edges.property_names)

    # Test that given intrinsic cell target is extracted (no extra node sets file given)
    for tgt_name in ['LayerA', 'RegionB']:
        res = test_module.apply(edges_table, nodes, None, target_name=tgt_name, node_sets_file=None)

        src_ids = nodes[0].ids(tgt_name)
        tgt_ids = nodes[1].ids(tgt_name)
        assert np.all(np.isin(res['@source_node'], src_ids)) and np.all(np.isin(res['@target_node'], tgt_ids))
        assert np.sum(np.logical_and(np.isin(edges_table['@source_node'], src_ids), np.isin(edges_table['@target_node'], tgt_ids))) == res.shape[0]

    # Test that given external cell target is extracted (given by extra node sets file)
    node_sets_file = os.path.join(TEST_DATA_DIR, 'node_sets_extra.json')
    with open(node_sets_file, 'r') as f:
        node_sets = json.load(f)
    for tgt_name in ['NSet1', 'NSet1']:
        res = test_module.apply(edges_table, nodes, None, target_name=tgt_name, node_sets_file=node_sets_file)

        src_ids = node_sets[tgt_name]['node_id']
        tgt_ids = node_sets[tgt_name]['node_id']
        assert np.all(np.isin(res['@source_node'], src_ids)) and np.all(np.isin(res['@target_node'], tgt_ids))
        assert np.sum(np.logical_and(np.isin(edges_table['@source_node'], src_ids), np.isin(edges_table['@target_node'], tgt_ids))) == res.shape[0]

    # Test special case when no cell target specified (should return empty connectome)
    res = test_module.apply(edges_table, nodes, None, target_name=None, node_sets_file=None)
    assert res.empty
    assert np.all(res.columns == edges_table.columns)
