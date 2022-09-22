import os
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bluepysnap import Circuit
from utils import TEST_DATA_DIR

import connectome_manipulator.model_building.conn_props as test_module


def test_extract():
    c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    src_mtypes = np.unique(nodes[0].get(src_ids, properties='mtype'))
    tgt_mtypes = np.unique(nodes[1].get(tgt_ids, properties='mtype'))
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    tab_mtypes_src = nodes[0].get(edges_table['@source_node'].to_numpy(), properties='mtype').to_numpy()
    tab_mtypes_tgt = nodes[1].get(edges_table['@target_node'].to_numpy(), properties='mtype').to_numpy()

    # Extract selected synapse/connection properties per pathway (= reference data)
    sel_prop = 'conductance'
    nsyn_data = {k: np.full((len(src_mtypes), len(tgt_mtypes)), np.nan) for k in ['mean', 'std', 'min', 'max']}
    prop_data = {k: np.full((len(src_mtypes), len(tgt_mtypes), 1), np.nan) for k in ['mean', 'std', 'std-within', 'min', 'max']}
    for i1, mt1 in enumerate(src_mtypes):
        for i2, mt2 in enumerate(tgt_mtypes):
            tab_sel = edges_table.loc[np.logical_and(tab_mtypes_src == mt1, tab_mtypes_tgt == mt2)]
            if tab_sel.size == 0:
                continue
            conns, nsyn_conn = np.unique(tab_sel[['@source_node', '@target_node']], axis=0, return_counts=True)
            nsyn_data['mean'][i1, i2] = np.mean(nsyn_conn)
            nsyn_data['std'][i1, i2] = np.std(nsyn_conn)
            nsyn_data['min'][i1, i2] = np.min(nsyn_conn)
            nsyn_data['max'][i1, i2] = np.max(nsyn_conn)

            prop_conn_means = []
            prop_conn_stds = []
            for conn in conns:
                sel_data = tab_sel[sel_prop].loc[np.logical_and(tab_sel['@source_node'] == conn[0], tab_sel['@target_node'] == conn[1])]
                prop_conn_means.append(np.mean(sel_data))
                prop_conn_stds.append(np.std(sel_data))
            prop_data['mean'][i1, i2] = np.mean(prop_conn_means)
            prop_data['std'][i1, i2] = np.std(prop_conn_means)
            prop_data['std-within'][i1, i2] = np.mean(prop_conn_stds)
            prop_data['min'][i1, i2] = np.min(tab_sel[sel_prop])
            prop_data['max'][i1, i2] = np.max(tab_sel[sel_prop])

    # Check extraction (w/o histograms)
    res = test_module.extract(c, min_sample_size_per_group=None, max_sample_size_per_group=None, hist_bins=50, sel_props=[sel_prop], sel_src=None, sel_dest=None)
    assert_array_equal(res['m_types'][0], src_mtypes)
    assert_array_equal(res['m_types'][1], tgt_mtypes)
    assert_array_equal(res['syn_props'], [sel_prop])
    for k in nsyn_data.keys():
        assert_array_equal(res['syns_per_conn_data'][k], nsyn_data[k])
    for k in prop_data.keys():
        assert_array_almost_equal(res['conn_prop_data'][k], prop_data[k])

    # Check extraction with Inf min sample size => Everything should be NaN
    res = test_module.extract(c, min_sample_size_per_group=np.inf, max_sample_size_per_group=None, hist_bins=50, sel_props=[sel_prop], sel_src=None, sel_dest=None)
    assert_array_equal(res['m_types'][0], src_mtypes)
    assert_array_equal(res['m_types'][1], tgt_mtypes)
    assert_array_equal(res['syn_props'], [sel_prop])
    for k in nsyn_data.keys():
        assert_array_equal(res['syns_per_conn_data'][k], np.full_like(nsyn_data[k], np.nan))
    for k in prop_data.keys():
        assert_array_almost_equal(res['conn_prop_data'][k], np.full_like(prop_data[k], np.nan))


def test_build():
    
#  'm_types': [['L4_MC', 'L4_PC', 'L5_PC'], ['L4_MC', 'L4_PC', 'L5_PC']],
#  'm_type_class': [['INH', 'EXC', 'EXC'], ['INH', 'EXC', 'EXC']],
#  'm_type_layer': [['LA', 'LB', 'LA'], ['LA', 'LB', 'LA']],
#  'syn_props': ['conductance'],
#     build(syns_per_conn_data, conn_prop_data, m_types, m_type_class, m_type_layer, syn_props, distr_types={}, data_types={}, data_bounds={}, **_):

    pass
