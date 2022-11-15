import os

import numpy as np
import pandas as pd
import pytest
from bluepysnap import Circuit
from mock import Mock, patch
from numpy.testing import assert_array_equal
from scipy.spatial import distance_matrix

from utils import TEST_DATA_DIR, setup_tempdir
import connectome_manipulator.model_building.conn_prob as test_module
from connectome_manipulator.model_building import model_types


def get_random_pos_matrix(n_row=None):
    """Get n_row positions between -50, 50"""
    n_row = n_row or np.random.randint(25, 50)
    return np.random.random((n_row, 3)) * 100 + 50


def test_extract():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))

    functions = [
        'extract_1st_order',
        'extract_2nd_order',
        'extract_3rd_order',
        'extract_4th_order',
        'extract_5th_order',
    ]

    for i, f in enumerate(functions):
        with patch(f'connectome_manipulator.model_building.conn_prob.{f}') as patched:
            test_module.extract(circuit, i + 1, pos_map_file='')
            patched.assert_called()

    order = 'fake'
    with pytest.raises(AssertionError, match=f'Order-{order} data extraction not supported!'):
        test_module.extract(circuit, order)


def test_build():
    functions = [
        'build_1st_order',
        'build_2nd_order',
        'build_3rd_order',
        'build_4th_order',
        'build_5th_order',
    ]

    for i, f in enumerate(functions):
        with patch(f'connectome_manipulator.model_building.conn_prob.{f}') as patched:
            test_module.build(i + 1)
            patched.assert_called()

    order = 'fake'
    with pytest.raises(AssertionError, match=f'Order-{order} model building not supported!'):
        test_module.build(order)


def test_plot():
    functions = [
        'plot_1st_order',
        'plot_2nd_order',
        'plot_3rd_order',
        'plot_4th_order',
        'plot_5th_order',
    ]

    for i, f in enumerate(functions):
        with patch(f'connectome_manipulator.model_building.conn_prob.{f}') as patched:
            test_module.plot(i + 1)
            patched.assert_called()

    order = 'fake'
    with pytest.raises(AssertionError,
                       match=f'Order-{order} data/model visualization not supported!'):
        test_module.plot(order)


def test_load_pos_mapping_model():

    test_module.load_pos_mapping_model(None)

    with setup_tempdir(__name__) as tempdir:
        filepath = os.path.join(tempdir, 'fake.json')

        with pytest.raises(AssertionError, match='Position mapping model file not found!'):
            test_module.load_pos_mapping_model(filepath)

        # Create dummy position mapping model
        pos_model = model_types.PosMapModel(pos_table=pd.DataFrame(np.random.rand(10, 3), columns=['x', 'y', 'z']))
        pos_model.save_model(os.path.split(filepath)[0], os.path.splitext(os.path.split(filepath)[1])[0])

        test_module.load_pos_mapping_model(filepath)


def test_get_neuron_positions():
    functions = [
        lambda x: x + 0.1,
        lambda x: x * 2,
        lambda x: x ** 3,
    ]

    res = test_module.get_neuron_positions(functions, range(3))
    assert_array_equal(res, [0.1, 2, 8])

    res = test_module.get_neuron_positions(functions, [np.arange(3)] * 3)
    assert_array_equal(res, [[0.1, 1.1, 2.1], [0, 2, 4], [0, 1, 8]])

    res = test_module.get_neuron_positions(lambda x: x + 1, range(3))
    assert_array_equal(res, range(1, 4))


# NOT USED ANY MORE
# def test_compute_dist_matrix():
#     src_pos = get_random_pos_matrix()
#     tgt_pos = get_random_pos_matrix()

#     # add one pos from src_pos to make sure there's a nan
#     tgt_pos = np.vstack((tgt_pos, src_pos[-1]))

#     expected = np.array([[np.sqrt(np.sum((t - s) ** 2)) for t in tgt_pos] for s in src_pos])
#     expected[expected == 0] = np.nan

#     res = test_module.compute_dist_matrix(src_pos, tgt_pos)
#     assert_array_equal(res, expected)


# NOT USED ANY MORE
# def test_compute_bip_matrix():
#     src_pos = get_random_pos_matrix()
#     tgt_pos = get_random_pos_matrix()

#     expected = np.array([[t - s for t in tgt_pos[:, 2]] for s in src_pos[:, 2]])

#     # basically what np.sign does:
#     expected[expected < 0] = -1
#     expected[expected > 0] = 1

#     res = test_module.compute_bip_matrix(src_pos, tgt_pos)
#     assert_array_equal(res, expected)


# NOT USED ANY MORE
# def test_compute_offset_matrices():
#     src_pos = get_random_pos_matrix()
#     tgt_pos = get_random_pos_matrix()

#     # transpose the axis (x, y, z) as the first dimension
#     expected = [[t - s for t in tgt_pos] for s in src_pos]
#     expected = np.transpose(expected, (2, 0, 1))

#     res = test_module.compute_offset_matrices(src_pos, tgt_pos)
#     assert_array_equal(res, expected)


# NOT USED ANY MORE
# def test_compute_position_matrices():
#     src_pos = get_random_pos_matrix()
#     tgt_pos = get_random_pos_matrix()

#     expected = [src_pos for _ in range(len(tgt_pos))]
#     expected = np.transpose(expected, (2, 1, 0))

#     res = test_module.compute_position_matrices(src_pos, tgt_pos)
#     assert_array_equal(res, expected)


def test_extract_dependant_p_conn():
    c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = c.edges[c.edges.population_names[0]]
    src_ids = edges.source.ids()
    tgt_ids = edges.target.ids()

    # Case 1: No dependencies
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(src_ids, tgt_ids, edges, [], [])
    assert res_count_all == len(src_ids) * len(tgt_ids), 'ERROR: Possible connection count mismatch!'
    assert res_count_conn == len(list(edges.iter_connections(src_ids, tgt_ids))), 'ERROR: Connection count mismatch!'
    assert res_p_conn == res_count_conn / res_count_all, 'ERROR: Connection probability mismatch!'

    # Case 2: 1-D connection probabilities (2 bins only)
    dep_mat = np.array([np.arange(len(src_ids))]).T * np.arange(len(tgt_ids))
    dep_bins = np.linspace(np.min(dep_mat), np.max(dep_mat), 3)
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(src_ids, tgt_ids, edges, [dep_mat], [dep_bins])
    assert len(res_p_conn) == len(res_count_conn) == len(res_count_all) == len(dep_bins) - 1, 'ERROR: Size mismatch!'
    adj_mat = np.array([[len(list(edges.iter_connections(s, t))) > 0 for t in tgt_ids] for s in src_ids])
    assert np.sum(res_count_all) == len(src_ids) * len(tgt_ids) and np.array_equal(res_count_all, np.histogram(dep_mat, dep_bins)[0]), 'ERROR: Possible connection count mismatch!'
    assert np.sum(res_count_conn) == np.sum(adj_mat) and \
           np.array_equal(res_count_conn, [np.sum(np.logical_and(adj_mat, dep_mat < dep_bins[1])),
                                           np.sum(np.logical_and(adj_mat, dep_mat >= dep_bins[1]))]), 'ERROR: Connection count mismatch!'
    assert np.array_equal(res_p_conn, res_count_conn / res_count_all), 'ERROR: Connection probability mismatch!'

    # Case 3: N-D connection probabilities (variable bins)
    N = 3
    dep_mats = [np.array([np.arange(len(src_ids))]).T * np.arange(len(tgt_ids))] * N
    dep_bins = [np.linspace(np.min(dep_mat), np.max(dep_mat), 3 * n + 5) for n in range(N)]
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(src_ids, tgt_ids, edges, dep_mats, dep_bins)
    assert res_p_conn.shape == res_count_conn.shape == res_count_all.shape == tuple([len(b) - 1 for b in dep_bins]), 'ERROR: Size mismatch!'
    assert np.sum(res_count_all) == len(src_ids) * len(tgt_ids), 'ERROR: Possible connection count mismatch!'
    assert np.sum(res_count_conn) == len(list(edges.iter_connections(src_ids, tgt_ids))), 'ERROR: Connection count mismatch!'
    assert np.array_equal(np.isnan(res_p_conn), np.isnan(res_count_conn / res_count_all)) and \
           np.array_equal(res_p_conn[np.isfinite(res_p_conn)], res_count_conn[res_count_all > 0] / res_count_all[res_count_all > 0]), 'ERROR: Connection probability mismatch!'

#     # NOTE by chrpok on 08/06/2022:
#     #  -edges must contain specific edges population
#     #  -input ids must be of type np.array
#
#     # NOTE by herttuai on 08/10/2021:
#     # These calls are exactly like those in test_conn_prob.py:211
#     # So I expect test_conn_prob.extract_1st_order to fail, too.
#     # I am not sure how this function is supposed to work.
#
#     with pytest.raises(ValueError):
#         # NOTE by herttuai on 08/10/2021:
#         # This throws a ValueError like: "invalid literal for int() with base 10: 'nodeA'".
#         # The failing line is:
#         #     test_conn_prob.py:175
#         #
#         # Has the iter_connections changed meanwhile?
#         res = test_module.extract_dependent_p_conn(src_ids, tgt_ids, edges, [], [])
#
#     with pytest.raises(TypeError):
#         # NOTE by herttuai on 08/10/2021:
#         # This raises TypeError like: "only integer scalar arrays can be converted to a scalar index".
#         # The failing line is:
#         #     test_conn_prob.py:192
#         res = test_module.extract_dependent_p_conn(np.array([0]), np.array([0]), edges, [], [])


def test_get_value_ranges():

    np.random.seed(0)

    # Check special case: 0-dim
    ndim = 0
    rng = np.nan
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == ndim

    # Check special case: 1-dim
    ndim = 1

    ## (a) Symmetric range
    rng = 100.0 * np.random.rand()
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == 2
    assert res == [-rng, rng]

    ## (a) Arbitrary range
    rng = [-100.0 * np.random.rand(), 100.0 * np.random.rand()]
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == 2
    assert res == [rng[0], rng[1]]

    # Check same range for all dims
    ndim = 10
    rng = 100.0 * np.random.rand()

    ## (a) Pos./neg. range
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == ndim
    assert np.all([r == [-rng, rng] for r in res])

    ## (b) Pos. range only
    res = test_module.get_value_ranges(rng, ndim, True)
    assert len(res) == ndim
    assert np.all([r == [0, rng] for r in res])
    
    # Check different ranges for differnt dims
    rng = [100.0 * np.random.rand() for d in range(ndim)]

    ## (a) Pos./neg. range
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == ndim
    assert np.all([res[i] == [-rng[i], rng[i]] for i in range(len(res))])

    ## (b) Pos. range only
    res = test_module.get_value_ranges(rng, ndim, True)
    assert len(res) == ndim
    assert np.all([res[i] == [0, rng[i]] for i in range(len(res))])

    ## (c) Mixed range
    pos_range = np.random.choice(2, ndim).astype(bool)
    res = test_module.get_value_ranges(rng, ndim, pos_range)
    assert len(res) == ndim
    assert np.all([res[i] == [0, rng[i]] for i in range(len(res)) if pos_range[i]])
    assert np.all([res[i] == [-rng[i], rng[i]] for i in range(len(res)) if not pos_range[i]])

    ## (d) Wrong numbers
    with pytest.raises(AssertionError, match=f'ERROR: max_range must have {ndim} elements!'):
        res = test_module.get_value_ranges(rng[:-1], ndim, pos_range)
    with pytest.raises(AssertionError, match=f'ERROR: pos_range must have {ndim} elements!'):
        res = test_module.get_value_ranges(rng, ndim, pos_range[:-1])

    # Check arbitrary ranges
    ndim = 10

    ## (a) Correct ranges (pos./neg.)
    rng = [[-100.0 * np.random.rand(), 100.0 * np.random.rand()] for i in range(ndim)]
    res = test_module.get_value_ranges(rng, ndim)
    assert len(res) == ndim
    assert np.all([res[i] == [rng[i][0], rng[i][1]] for i in range(len(res))])

    ## (b) Correct ranges (pos. only)
    rng = [[0.0, 100.0 * np.random.rand()] for i in range(ndim)]
    res = test_module.get_value_ranges(rng, ndim, True)
    assert len(res) == ndim
    assert np.all([res[i] == [rng[i][0], rng[i][1]] for i in range(len(res))])

    ## (c) Wrong pos./neg.
    rng = [[-100.0 * np.random.rand(), 100.0 * np.random.rand()] for i in range(ndim)]
    with pytest.raises(AssertionError, match=f'ERROR: Range of coord 0 must include 0!'):
        res = test_module.get_value_ranges(rng, ndim, True)

    ## (d) Wrong ranges
    rng = [[100.0 * np.random.rand(), -100.0 * np.random.rand()] for i in range(ndim)]
    with pytest.raises(AssertionError, match=f'ERROR: Range of coord 0 invalid!'):
        res = test_module.get_value_ranges(rng, ndim)


def test_extract_1st_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    nodes = [circuit.nodes['nodeA']] * 2 # Src/tgt populations
    edges = circuit.edges['nodeA__nodeA__chemical']
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    for n in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)
        nconn = len(list(edges.iter_connections(source=src_sel, target=tgt_sel)))

        for min_nbins in [1, 10, 100]:
            res = test_module.extract_1st_order(nodes, edges, src_sel, tgt_sel, min_count_per_bin=min_nbins)
            if nsrc * ntgt >= min_nbins:
                assert np.isclose(res['p_conn'], nconn / (nsrc * ntgt))
            else:
                assert np.isnan(res['p_conn']) # Not enought data points
            assert res['src_cell_count'] == nsrc
            assert res['tgt_cell_count'] == ntgt


def test_build_1st_order():
    np.random.seed(0)
    for p_conn in np.random.rand(10):
        model = test_module.build_1st_order(p_conn)
        assert np.isclose(model.p_conn, p_conn)


def test_extract_2nd_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    nodes = [circuit.nodes['nodeA']] * 2 # Src/tgt populations
    edges = circuit.edges['nodeA__nodeA__chemical']
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 100
    for rep in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        dist = distance_matrix(src_pos, tgt_pos) # Distance matrix
        dist[dist == 0.0] = np.nan
        nconn = np.array([[len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel] for s in src_sel]) # Number of connection matrix
        
        num_bins = np.ceil(np.nanmax(dist) / bin_size_um).astype(int) # Distance binning
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um
        dist_bins[-1] += 1e-3 # So that max. value is always included in last bin
        conn_cnt = np.full(num_bins, -1) # Conn. count
        all_cnt = np.full(num_bins, -1) # All pair count
        for bidx in range(num_bins):
            dsel = np.logical_and(dist >= dist_bins[bidx], dist < dist_bins[bidx + 1])
            conn_cnt[bidx] = np.sum(nconn[dsel])
            all_cnt[bidx] = np.sum(dsel)
        p = conn_cnt / all_cnt # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_2nd_order(nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins)
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res['p_conn_dist'], p_sel, equal_nan=True)
            assert np.array_equal(res['count_conn'], conn_cnt)
            assert np.array_equal(res['count_all'], all_cnt)
            assert np.array_equal(res['dist_bins'], dist_bins)
            assert res['src_cell_count'] == nsrc
            assert res['tgt_cell_count'] == ntgt


def test_build_2nd_order():

    dist_bins = np.arange(0, 1001, 10)
    d = np.array([np.mean(dist_bins[i : i + 2]) for i in range(len(dist_bins) - 1)])

    # Check simple exponential model building
    np.random.seed(0)
    for rep in range(10):
        exp_coefs = [1e-1 * np.random.rand(), 1e-2 * np.random.rand()]
        exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
        exp_data = exp_model(d, *exp_coefs)

        model = test_module.build_2nd_order(exp_data, dist_bins, np.zeros_like(exp_data), model_specs={'type': 'SimpleExponential'})
        model_coefs = [model.get_param_dict()[k] for k in ['scale', 'exponent']]
        assert np.allclose(exp_coefs, model_coefs)

    # Check complex exponential model building [EXPERIMENTAL: Just one working test example, since model fitting not so robust]
    exp_coefs = [1e-1, 1e-4, 2.0, 1e-1, 1e-4]
    exp_model = lambda x, a, b, c, d, e: a * np.exp(-b * np.array(x)**c) + d * np.exp(-e * np.array(x))
    exp_data = exp_model(d, *exp_coefs)

    model = test_module.build_2nd_order(exp_data, dist_bins, np.zeros_like(exp_data), model_specs={'type': 'ComplexExponential'})
    model_coefs = [model.get_param_dict()[k] for k in ['prox_scale', 'prox_exp', 'prox_exp_pow', 'dist_scale', 'dist_exp']]
    assert np.allclose(exp_coefs, model_coefs)


def test_extract_3rd_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    nodes = [circuit.nodes['nodeA']] * 2 # Src/tgt populations
    edges = circuit.edges['nodeA__nodeA__chemical']
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 100
    for rep in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        dist = distance_matrix(src_pos, tgt_pos) # Distance matrix
        dist[dist == 0.0] = np.nan
        bip = np.array([[np.sign(tgt_pos['z'].loc[t] - src_pos['z'].loc[s]) for t in tgt_sel] for s in src_sel])
        nconn = np.array([[len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel] for s in src_sel]) # Number of connection matrix

        num_bins = np.ceil(np.nanmax(dist) / bin_size_um).astype(int) # Distance binning
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um
        dist_bins[-1] += 1e-3 # So that max. value is always included in last bin
        conn_cnt = np.full((num_bins, 2), -1) # Conn. count
        all_cnt = np.full((num_bins, 2), -1) # All pair count
        for bidx in range(num_bins):
            for bipidx, bipval in enumerate([-1, 1]):
                dsel = np.logical_and(np.logical_and(dist >= dist_bins[bidx], dist < dist_bins[bidx + 1]), bip == bipval)
                conn_cnt[bidx, bipidx] = np.sum(nconn[dsel])
                all_cnt[bidx, bipidx] = np.sum(dsel)
        p = conn_cnt / all_cnt # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_3rd_order(nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins)
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res['p_conn_dist_bip'], p_sel, equal_nan=True)
            assert np.array_equal(res['count_conn'], conn_cnt)
            assert np.array_equal(res['count_all'], all_cnt)
            assert np.array_equal(res['dist_bins'], dist_bins)
            assert res['src_cell_count'] == nsrc
            assert res['tgt_cell_count'] == ntgt


def test_build_3rd_order():

    dist_bins = np.arange(0, 1001, 10)
    d = np.array([np.mean(dist_bins[i : i + 2]) for i in range(len(dist_bins) - 1)])

    # Check simple exponential model building
    np.random.seed(0)
    for rep in range(10):
        exp_coefs = [1e-1 * np.random.rand(), 1e-2 * np.random.rand(), 1e-1 * np.random.rand(), 1e-2 * np.random.rand()] # 'scale_N', 'exponent_N', 'scale_P', 'exponent_P'
        exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
        exp_data = np.array([exp_model(d, *exp_coefs[:2]), exp_model(d, *exp_coefs[2:])]).T

        model = test_module.build_3rd_order(exp_data, dist_bins, np.zeros_like(exp_data), model_specs={'type': 'SimpleExponential'})
        model_coefs = [model.get_param_dict()[k] for k in ['scale_N', 'exponent_N', 'scale_P', 'exponent_P']]
        assert np.allclose(exp_coefs, model_coefs)

    # Check complex exponential model building [EXPERIMENTAL: Just one working test example, since model fitting not so robust]
    exp_coefs = [1e-1, 1e-4, 2.0, 1e-1, 1e-4, 2e-1, 2e-4, 1.75, 2e-1, 2e-4] # 'prox_scale_N', 'prox_exp_N', 'prox_exp_pow_N', 'dist_scale_N', 'dist_exp_N', 'prox_scale_P', 'prox_exp_P', 'prox_exp_pow_P', 'dist_scale_P', 'dist_exp_P'
    exp_model = lambda x, a, b, c, d, e: a * np.exp(-b * np.array(x)**c) + d * np.exp(-e * np.array(x))
    exp_data = np.array([exp_model(d, *exp_coefs[:5]), exp_model(d, *exp_coefs[5:])]).T

    model = test_module.build_3rd_order(exp_data, dist_bins, np.zeros_like(exp_data), model_specs={'type': 'ComplexExponential'})
    model_coefs = [model.get_param_dict()[k] for k in ['prox_scale_N', 'prox_exp_N', 'prox_exp_pow_N', 'dist_scale_N', 'dist_exp_N', 'prox_scale_P', 'prox_exp_P', 'prox_exp_pow_P', 'dist_scale_P', 'dist_exp_P']]
    assert np.allclose(exp_coefs, model_coefs)
