import os
import numpy as np
import pytest
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
#     c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
#     m_types = [list(c.nodes.property_values('mtype'))] * 2 # src/tgt mtypes
#     m_type_class = [c.nodes.get({'mtype': m})['synapse_class'] for m in m_types[0]]

    m_types = [['L4_MC', 'L4_PC', 'L5_PC']] * 2
    m_type_class = [['INH', 'EXC', 'EXC']] * 2
    m_type_layer = [[4, 4, 5]] * 2

    props = ['conductance', 'u_syn']
    src_mtypes = m_types[0]
    tgt_mtypes = m_types[1]
    np.random.seed(0)
    nsyn_data = {k: np.random.randint(low=1, high=10, size=(len(src_mtypes), len(tgt_mtypes))) for k in ['mean', 'std', 'min', 'max']}
    prop_data = {k: np.random.rand(len(src_mtypes), len(tgt_mtypes), len(props)) for k in ['mean', 'std', 'std-within', 'min', 'max']}

    # Check model building (default settings)
    res = test_module.build(nsyn_data, prop_data, m_types, m_type_class, m_type_layer, props, distr_types={}, data_types={}, data_bounds={})
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr['type'] == 'normal'
                assert model_distr['mean'] == prop_data['mean'][sidx, tidx, pidx]
                assert model_distr['std'] == prop_data['std'][sidx, tidx, pidx]
                assert model_distr['std-within'] == prop_data['std-within'][sidx, tidx, pidx]
            model_distr = res.get_distr_props(prop_name='n_syn_per_conn', src_type=s_mt, tgt_type=t_mt)
            assert model_distr['mean'] == nsyn_data['mean'][sidx, tidx]
            assert model_distr['std'] == nsyn_data['std'][sidx, tidx]

    with pytest.raises(KeyError):
        res.get_distr_props(prop_name='WRONG_NAME', src_type=s_mt, tgt_type=t_mt)
    with pytest.raises(KeyError):
        res.get_distr_props(prop_name='n_syn_per_conn', src_type='WRONG_TYPE', tgt_type=t_mt)
    with pytest.raises(KeyError):
        res.get_distr_props(prop_name='n_syn_per_conn', src_type=s_mt, tgt_type='WRONG_TYPE')

    # Check distribution types
    res = test_module.build(nsyn_data, prop_data, m_types, m_type_class, m_type_layer, props, distr_types={pp: 'truncnorm' for pp in props}, data_types={}, data_bounds={})
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr['type'] == 'truncnorm'
                assert model_distr['mean'] == prop_data['mean'][sidx, tidx, pidx]
                assert model_distr['std'] == prop_data['std'][sidx, tidx, pidx]
                assert model_distr['min'] == prop_data['min'][sidx, tidx, pidx]
                assert model_distr['max'] == prop_data['max'][sidx, tidx, pidx]
                assert model_distr['std-within'] == prop_data['std-within'][sidx, tidx, pidx]

    with pytest.raises(AssertionError, match='ERROR: Distribution type "WRONG_TYPE" not supported!'):
        test_module.build(nsyn_data, prop_data, m_types, m_type_class, m_type_layer, props, distr_types={pp: 'WRONG_TYPE' for pp in props}, data_types={}, data_bounds={})

    # Check data types
    res = test_module.build(nsyn_data, prop_data, m_types, m_type_class, m_type_layer, props, distr_types={}, data_types={pp: 'int' for pp in props[1:]}, data_bounds={})
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr['dtype'] == 'int'
                assert isinstance(res.draw(prop_name=pp, src_type=s_mt, tgt_type=t_mt)[0], np.int64)

    # Check data bounds
    # TODO

    # Check synapse generation (w/o randomization)
    # syn = res.apply()
    # assert syn.shape[0] == 
    # assert np.all([pp in syn.columns for pp in props])

    # Check interpolation
    # TODO
