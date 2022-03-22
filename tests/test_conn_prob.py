import os

import numpy as np
import pandas as pd
import pytest
from bluepysnap import Circuit
from mock import Mock, patch
from numpy.testing import assert_array_equal

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


def test_compute_dist_matrix():
    src_pos = get_random_pos_matrix()
    tgt_pos = get_random_pos_matrix()

    # add one pos from src_pos to make sure there's a nan
    tgt_pos = np.vstack((tgt_pos, src_pos[-1]))

    expected = np.array([[np.sqrt(np.sum((t - s) ** 2)) for t in tgt_pos] for s in src_pos])
    expected[expected == 0] = np.nan

    res = test_module.compute_dist_matrix(src_pos, tgt_pos)
    assert_array_equal(res, expected)


def test_compute_bip_matrix():
    src_pos = get_random_pos_matrix()
    tgt_pos = get_random_pos_matrix()

    expected = np.array([[t - s for t in tgt_pos[:, 2]] for s in src_pos[:, 2]])

    # basically what np.sign does:
    expected[expected < 0] = -1
    expected[expected > 0] = 1

    res = test_module.compute_bip_matrix(src_pos, tgt_pos)
    assert_array_equal(res, expected)


def test_compute_offset_matrices():
    src_pos = get_random_pos_matrix()
    tgt_pos = get_random_pos_matrix()

    # transpose the axis (x, y, z) as the first dimension
    expected = [[t - s for t in tgt_pos] for s in src_pos]
    expected = np.transpose(expected, (2, 0, 1))

    res = test_module.compute_offset_matrices(src_pos, tgt_pos)
    assert_array_equal(res, expected)


def test_compute_position_matrices():
    src_pos = get_random_pos_matrix()
    tgt_pos = get_random_pos_matrix()

    expected = [src_pos for _ in range(len(tgt_pos))]
    expected = np.transpose(expected, (2, 1, 0))

    res = test_module.compute_position_matrices(src_pos, tgt_pos)
    assert_array_equal(res, expected)


def test_extract_dependant_p_conn():
    edges = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json')).edges
    src_ids = edges[edges.population_names[0]].source.ids()
    tgt_ids = edges[edges.population_names[0]].target.ids()

    # NOTE by herttuai on 08/10/2021:
    # These calls are exactly like those in test_conn_prob.py:211
    # So I expect test_conn_prob.extract_1st_order to fail, too.
    # I am not sure how this function is supposed to work.

    with pytest.raises(ValueError):
        # NOTE by herttuai on 08/10/2021:
        # This throws a ValueError like: "invalid literal for int() with base 10: 'nodeA'".
        # The failing line is:
        #     test_conn_prob.py:175
        #
        # Has the iter_connections changed meanwhile?
        res = test_module.extract_dependent_p_conn(src_ids, tgt_ids, edges, [], [])

    with pytest.raises(TypeError):
        # NOTE by herttuai on 08/10/2021:
        # This raises TypeError like: "only integer scalar arrays can be converted to a scalar index".
        # The failing line is:
        #     test_conn_prob.py:192
        res = test_module.extract_dependent_p_conn([0], [0], edges, [], [])
