# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import re

import numpy as np
import pandas as pd
import pytest
from bluepysnap import Circuit
from mock import Mock, patch
from numpy.testing import assert_array_equal
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix
from voxcell import VoxelData

from utils import TEST_DATA_DIR, setup_tempdir
import connectome_manipulator.model_building.conn_prob as test_module
from connectome_manipulator.model_building import model_types


def get_random_pos_matrix(n_row=None):
    """Get n_row positions between -50, 50"""
    n_row = n_row or np.random.randint(25, 50)
    return np.random.random((n_row, 3)) * 100 + 50


def test_extract():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))

    functions = [
        "extract_1st_order",
        "extract_2nd_order",
        "extract_3rd_order",
        "extract_4th_order",
        "extract_5th_order",
    ]

    for i, f in enumerate(functions):
        with patch(f"connectome_manipulator.model_building.conn_prob.{f}") as patched:
            test_module.extract(circuit, i + 1, pos_map_file="")
            patched.assert_called()

    order = "fake"
    with pytest.raises(
        AssertionError, match=re.escape(f"Order-{order} data extraction not supported!")
    ):
        test_module.extract(circuit, order)


def test_build():
    functions = [
        "build_1st_order",
        "build_2nd_order",
        "build_3rd_order",
        "build_4th_order",
        "build_5th_order",
    ]

    for i, f in enumerate(functions):
        with patch(f"connectome_manipulator.model_building.conn_prob.{f}") as patched:
            test_module.build(i + 1)
            patched.assert_called()

    order = "fake"
    with pytest.raises(
        AssertionError, match=re.escape(f"Order-{order} model building not supported!")
    ):
        test_module.build(order)


def test_plot():
    functions = [
        "plot_1st_order",
        "plot_2nd_order",
        "plot_3rd_order",
        "plot_4th_order",
        "plot_5th_order",
    ]

    for i, f in enumerate(functions):
        with patch(f"connectome_manipulator.model_building.conn_prob.{f}") as patched:
            test_module.plot(i + 1)
            patched.assert_called()

    order = "fake"
    with pytest.raises(
        AssertionError, match=re.escape(f"Order-{order} data/model visualization not supported!")
    ):
        test_module.plot(order)


def test_load_pos_mapping_model():
    test_module.load_pos_mapping_model(None)

    with setup_tempdir(__name__) as tempdir:
        filepath = os.path.join(tempdir, "fake.json")

        with pytest.raises(
            AssertionError, match=re.escape("Position mapping model file not found!")
        ):
            test_module.load_pos_mapping_model(filepath)

        # Create dummy position mapping model
        pos_model = model_types.PosMapModel(pos_table=pd.DataFrame(np.random.rand(10, 3)))
        pos_model.save_model(
            os.path.split(filepath)[0], os.path.splitext(os.path.split(filepath)[1])[0]
        )

        test_module.load_pos_mapping_model(filepath)


def test_get_neuron_positions_by_id():
    functions = [
        lambda x: x + 0.1,
        lambda x: x * 2,
        lambda x: x**3,
    ]

    res = test_module.get_neuron_positions_by_id(functions, range(3))
    assert_array_equal(res, [0.1, 2, 8])

    res = test_module.get_neuron_positions_by_id(functions, [np.arange(3)] * 3)
    assert_array_equal(res, [[0.1, 1.1, 2.1], [0, 2, 4], [0, 1, 8]])

    res = test_module.get_neuron_positions_by_id(lambda x: x + 1, range(3))
    assert_array_equal(res, range(1, 4))


def test_get_neuron_positions():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    # Permute node IDs, so that src/tgt are different
    np.random.seed(0)
    src_ids = np.random.permutation(nodes[0].ids())
    tgt_ids = np.random.permutation(nodes[1].ids())

    # Case 1: No access function, no voxel map => Should correspond to node positions
    res = test_module.get_neuron_positions(nodes, [src_ids, tgt_ids], pos_acc=None, vox_map=None)
    assert_array_equal(nodes[0].positions(src_ids), res[0])
    assert_array_equal(nodes[1].positions(tgt_ids), res[1])

    # Case 2: Position access through PosMapModel (no voxel map) => Should correspond to mapped positions
    src_pos_table = pd.DataFrame(np.random.rand(len(src_ids), 3))
    tgt_pos_table = pd.DataFrame(np.random.rand(len(tgt_ids), 3))
    src_pos_model = model_types.PosMapModel(pos_table=src_pos_table)
    tgt_pos_model = model_types.PosMapModel(pos_table=tgt_pos_table)
    src_pos_acc = lambda gids: src_pos_model.apply(gids=gids)
    tgt_pos_acc = lambda gids: tgt_pos_model.apply(gids=gids)

    res = test_module.get_neuron_positions(
        nodes, [src_ids, tgt_ids], pos_acc=[src_pos_acc, tgt_pos_acc], vox_map=None
    )
    assert_array_equal(src_pos_table.loc[src_ids], res[0])
    assert_array_equal(tgt_pos_table.loc[tgt_ids], res[1])

    # Case 3: Position access through voxel map (no access function) => Should correspond to voxel positions
    src_vox_map = VoxelData.load_nrrd(os.path.join(TEST_DATA_DIR, "xy_map_lin.nrrd"))
    tgt_vox_map = VoxelData.load_nrrd(os.path.join(TEST_DATA_DIR, "xy_map_ones.nrrd"))

    res = test_module.get_neuron_positions(
        nodes, [src_ids, tgt_ids], pos_acc=None, vox_map=[src_vox_map, tgt_vox_map]
    )
    assert_array_equal(src_vox_map.lookup(nodes[0].positions(src_ids).to_numpy()), res[0])
    assert_array_equal(tgt_vox_map.lookup(nodes[1].positions(tgt_ids).to_numpy()), res[1])

    # Case 4: Both PosMapModel and voxel map provided => Must fail, not supported
    with pytest.raises(
        AssertionError,
        match=re.escape("Voxel map not supported when providing position access functions!"),
    ):
        res = test_module.get_neuron_positions(
            nodes,
            [src_ids, tgt_ids],
            pos_acc=[src_pos_acc, tgt_pos_acc],
            vox_map=[src_vox_map, tgt_vox_map],
        )


def test_get_pos_mapping_fcts():
    # Case 1: No mapping provided => Must return None's
    res = test_module.get_pos_mapping_fcts(pos_map_file=None)
    assert res[0] is None  # pos_acc
    assert res[1] is None  # vox_map

    # Case 2: Position mapping model .json file(s) => Must return pos_acc, but no vox_map
    with setup_tempdir(__name__) as tempdir:
        # Create dummy position mapping models
        pos_tables = []
        nids = 10
        np.random.seed(0)
        for i in range(2):
            _pos_tab = pd.DataFrame(np.random.rand(10, 3), index=i * nids + np.arange(nids))
            pos_model = model_types.PosMapModel(pos_table=_pos_tab)
            pos_model.save_model(tempdir, f"dummy_pos_map{i}")
            pos_tables.append(_pos_tab)

        # (a) Single src/tgt file
        res = test_module.get_pos_mapping_fcts(
            pos_map_file=os.path.join(tempdir, "dummy_pos_map0.json")
        )
        assert res[1] is None  # vox_map
        assert_array_equal(pos_tables[0].to_numpy(), res[0][0](pos_tables[0].index))  # src
        assert_array_equal(pos_tables[0].to_numpy(), res[0][1](pos_tables[0].index))  # tgt

        # (b) Separate src/tgt files
        res = test_module.get_pos_mapping_fcts(
            pos_map_file=[os.path.join(tempdir, f"dummy_pos_map{i}.json") for i in range(2)]
        )
        assert res[1] is None  # vox_map
        assert_array_equal(pos_tables[0].to_numpy(), res[0][0](pos_tables[0].index))  # src
        assert_array_equal(pos_tables[1].to_numpy(), res[0][1](pos_tables[1].index))  # tgt

    # Case 3: Voxel map .nrrd file(s) => Must return vox_map, but no pos_acc

    # (a) Single src/tgt file
    vox_file = os.path.join(TEST_DATA_DIR, "xy_map_lin.nrrd")
    vox_map = VoxelData.load_nrrd(vox_file)
    res = test_module.get_pos_mapping_fcts(pos_map_file=vox_file)
    assert res[0] is None  # pos_acc
    assert_array_equal(vox_map.raw, res[1][0].raw)  # src
    assert_array_equal(vox_map.bbox, res[1][0].bbox)
    assert_array_equal(vox_map.raw, res[1][1].raw)  # tgt
    assert_array_equal(vox_map.bbox, res[1][1].bbox)

    # (b) Separate src/tgt files
    src_vox_file = os.path.join(TEST_DATA_DIR, "xy_map_lin.nrrd")
    tgt_vox_file = os.path.join(TEST_DATA_DIR, "xy_map_ones.nrrd")
    src_vox_map = VoxelData.load_nrrd(src_vox_file)
    tgt_vox_map = VoxelData.load_nrrd(tgt_vox_file)
    res = test_module.get_pos_mapping_fcts(pos_map_file=[src_vox_file, tgt_vox_file])
    assert res[0] is None  # pos_acc
    assert_array_equal(src_vox_map.raw, res[1][0].raw)  # src
    assert_array_equal(src_vox_map.bbox, res[1][0].bbox)
    assert_array_equal(tgt_vox_map.raw, res[1][1].raw)  # tgt
    assert_array_equal(tgt_vox_map.bbox, res[1][1].bbox)

    # Case 4: Invalid file(s) => Must fail

    # (a) Single src/tgt file with wrong extension
    with pytest.raises(
        AssertionError, match=re.escape("Position mapping file error (must be .json or .nrrd)!")
    ):
        res = test_module.get_pos_mapping_fcts(pos_map_file="dummy.xyz")

    # (b) Separate src/tgt files with wrong extension
    with pytest.raises(
        AssertionError, match=re.escape("Position mapping file error (must be .json or .nrrd)!")
    ):
        res = test_module.get_pos_mapping_fcts(pos_map_file=["dummy1.xyz", "dummy2.xyz"])

    # (c) Separate src/tgt files with valid but different extensions
    with pytest.raises(
        AssertionError,
        match=re.escape("Same file type for source/target position mappings required!"),
    ):
        res = test_module.get_pos_mapping_fcts(pos_map_file=["dummy1.json", "dummy2.nrrd"])

    # (d) Single file provided as list
    with pytest.raises(
        AssertionError, match=re.escape("Two position mapping files (source/target) expected!")
    ):
        res = test_module.get_pos_mapping_fcts(pos_map_file=["dummy1.json"])

    # (e) More than two separte files provided
    with pytest.raises(
        AssertionError, match=re.escape("Two position mapping files (source/target) expected!")
    ):
        res = test_module.get_pos_mapping_fcts(
            pos_map_file=["dummy1.json", "dummy2.json", "dummy3.json"]
        )


def test_extract_dependant_p_conn():
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    src_ids = edges.source.ids()
    tgt_ids = edges.target.ids()

    # Case 1: No dependencies
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(
        src_ids, tgt_ids, edges, [], []
    )
    assert res_count_all == len(src_ids) * len(
        tgt_ids
    ), "ERROR: Possible connection count mismatch!"
    assert res_count_conn == len(
        list(edges.iter_connections(src_ids, tgt_ids))
    ), "ERROR: Connection count mismatch!"
    assert res_p_conn == res_count_conn / res_count_all, "ERROR: Connection probability mismatch!"

    # Case 2: 1-D connection probabilities (2 bins only)
    dep_mat = np.array([np.arange(len(src_ids))]).T * np.arange(len(tgt_ids))
    dep_bins = np.linspace(np.min(dep_mat), np.max(dep_mat), 3)
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(
        src_ids, tgt_ids, edges, [dep_mat], [dep_bins]
    )
    assert (
        len(res_p_conn) == len(res_count_conn) == len(res_count_all) == len(dep_bins) - 1
    ), "ERROR: Size mismatch!"
    adj_mat = np.array(
        [[len(list(edges.iter_connections(s, t))) > 0 for t in tgt_ids] for s in src_ids]
    )
    assert np.sum(res_count_all) == len(src_ids) * len(tgt_ids) and np.array_equal(
        res_count_all, np.histogram(dep_mat, dep_bins)[0]
    ), "ERROR: Possible connection count mismatch!"
    assert np.sum(res_count_conn) == np.sum(adj_mat) and np.array_equal(
        res_count_conn,
        [
            np.sum(np.logical_and(adj_mat, dep_mat < dep_bins[1])),
            np.sum(np.logical_and(adj_mat, dep_mat >= dep_bins[1])),
        ],
    ), "ERROR: Connection count mismatch!"
    assert np.array_equal(
        res_p_conn, res_count_conn / res_count_all
    ), "ERROR: Connection probability mismatch!"

    # Case 3: N-D connection probabilities (variable bins)
    N = 3
    dep_mats = [np.array([np.arange(len(src_ids))]).T * np.arange(len(tgt_ids))] * N
    dep_bins = [np.linspace(np.min(dep_mat), np.max(dep_mat), 3 * n + 5) for n in range(N)]
    res_p_conn, res_count_conn, res_count_all = test_module.extract_dependent_p_conn(
        src_ids, tgt_ids, edges, dep_mats, dep_bins
    )
    assert (
        res_p_conn.shape
        == res_count_conn.shape
        == res_count_all.shape
        == tuple([len(b) - 1 for b in dep_bins])
    ), "ERROR: Size mismatch!"
    assert np.sum(res_count_all) == len(src_ids) * len(
        tgt_ids
    ), "ERROR: Possible connection count mismatch!"
    assert np.sum(res_count_conn) == len(
        list(edges.iter_connections(src_ids, tgt_ids))
    ), "ERROR: Connection count mismatch!"
    assert np.array_equal(
        np.isnan(res_p_conn), np.isnan(res_count_conn / res_count_all)
    ) and np.array_equal(
        res_p_conn[np.isfinite(res_p_conn)],
        res_count_conn[res_count_all > 0] / res_count_all[res_count_all > 0],
    ), "ERROR: Connection probability mismatch!"


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
    with pytest.raises(AssertionError, match=f"ERROR: max_range must have {ndim} elements!"):
        res = test_module.get_value_ranges(rng[:-1], ndim, pos_range)
    with pytest.raises(AssertionError, match=f"ERROR: pos_range must have {ndim} elements!"):
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
    with pytest.raises(AssertionError, match=f"ERROR: Range of coord 0 must include 0!"):
        res = test_module.get_value_ranges(rng, ndim, True)

    ## (d) Wrong ranges
    rng = [[100.0 * np.random.rand(), -100.0 * np.random.rand()] for i in range(ndim)]
    with pytest.raises(AssertionError, match=f"ERROR: Range of coord 0 invalid!"):
        res = test_module.get_value_ranges(rng, ndim)


def test_extract_1st_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    np.random.seed(0)
    for n in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)
        nconn = len(list(edges.iter_connections(source=src_sel, target=tgt_sel)))

        for min_nbins in [1, 10, 100]:
            res = test_module.extract_1st_order(
                nodes, edges, src_sel, tgt_sel, min_count_per_bin=min_nbins
            )
            if nsrc * ntgt >= min_nbins:
                assert np.isclose(res["p_conn"], nconn / (nsrc * ntgt))
            else:
                assert np.isnan(res["p_conn"])  # Not enought data points
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_1st_order():
    np.random.seed(0)
    for p_conn in np.random.rand(10):
        model = test_module.build_1st_order(p_conn)
        assert np.isclose(model.p_conn, p_conn)


def test_extract_2nd_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 100
    np.random.seed(0)
    for rep in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        dist = distance_matrix(src_pos, tgt_pos)  # Distance matrix
        dist[dist == 0.0] = np.nan
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        num_bins = np.ceil(np.nanmax(dist) / bin_size_um).astype(int)  # Distance binning
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um
        dist_bins[-1] += 1e-3  # So that max. value is always included in last bin
        conn_cnt = np.full(num_bins, -1)  # Conn. count
        all_cnt = np.full(num_bins, -1)  # All pair count
        for bidx in range(num_bins):
            dsel = np.logical_and(dist >= dist_bins[bidx], dist < dist_bins[bidx + 1])
            conn_cnt[bidx] = np.sum(nconn[dsel])
            all_cnt[bidx] = np.sum(dsel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_2nd_order(
                nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_dist"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.array_equal(res["dist_bins"], dist_bins)
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_2nd_order():
    dist_bins = np.arange(0, 1001, 10)
    d = np.array([np.mean(dist_bins[i : i + 2]) for i in range(len(dist_bins) - 1)])

    # Check simple exponential model building
    np.random.seed(0)
    for rep in range(10):
        exp_coefs = [1e-1 * np.random.rand(), 1e-2 * np.random.rand()]
        exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
        exp_data = exp_model(d, *exp_coefs)

        model = test_module.build_2nd_order(
            exp_data, dist_bins, np.zeros_like(exp_data), model_specs={"type": "SimpleExponential"}
        )
        model_coefs = [model.get_param_dict()[k] for k in ["scale", "exponent"]]
        assert np.allclose(exp_coefs, model_coefs)

    # Check complex exponential model building [EXPERIMENTAL: Just one working test example, since model fitting not so robust]
    exp_coefs = [1e-1, 1e-4, 2.0, 1e-1, 1e-4]
    exp_model = lambda x, a, b, c, d, e: a * np.exp(-b * np.array(x) ** c) + d * np.exp(
        -e * np.array(x)
    )
    exp_data = exp_model(d, *exp_coefs)

    model = test_module.build_2nd_order(
        exp_data, dist_bins, np.zeros_like(exp_data), model_specs={"type": "ComplexExponential"}
    )
    model_coefs = [
        model.get_param_dict()[k]
        for k in ["prox_scale", "prox_exp", "prox_exp_pow", "dist_scale", "dist_exp"]
    ]
    assert np.allclose(exp_coefs, model_coefs)


def test_extract_3rd_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 100
    np.random.seed(0)
    for rep in range(10):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        dist = distance_matrix(src_pos, tgt_pos)  # Distance matrix
        dist[dist == 0.0] = np.nan
        bip = np.array(
            [[np.sign(tgt_pos["z"].loc[t] - src_pos["z"].loc[s]) for t in tgt_sel] for s in src_sel]
        )
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        num_bins = np.ceil(np.nanmax(dist) / bin_size_um).astype(int)  # Distance binning
        dist_bins = np.arange(0, num_bins + 1) * bin_size_um
        dist_bins[-1] += 1e-3  # So that max. value is always included in last bin
        conn_cnt = np.full((num_bins, 2), -1)  # Conn. count
        all_cnt = np.full((num_bins, 2), -1)  # All pair count
        for bidx in range(num_bins):
            for bipidx, bipval in enumerate([-1, 1]):
                dsel = np.logical_and(
                    np.logical_and(dist >= dist_bins[bidx], dist < dist_bins[bidx + 1]),
                    bip == bipval,
                )
                conn_cnt[bidx, bipidx] = np.sum(nconn[dsel])
                all_cnt[bidx, bipidx] = np.sum(dsel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_3rd_order(
                nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_dist_bip"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.array_equal(res["dist_bins"], dist_bins)
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_3rd_order():
    dist_bins = np.arange(0, 1001, 10)
    d = np.array([np.mean(dist_bins[i : i + 2]) for i in range(len(dist_bins) - 1)])

    # Check simple exponential model building
    np.random.seed(0)
    for rep in range(10):
        exp_coefs = [
            1e-1 * np.random.rand(),
            1e-2 * np.random.rand(),
            1e-1 * np.random.rand(),
            1e-2 * np.random.rand(),
        ]  # 'scale_N', 'exponent_N', 'scale_P', 'exponent_P'
        exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
        exp_data = np.array([exp_model(d, *exp_coefs[:2]), exp_model(d, *exp_coefs[2:])]).T

        model = test_module.build_3rd_order(
            exp_data,
            dist_bins,
            np.zeros_like(exp_data),
            bip_coord_data=0,
            model_specs={"type": "SimpleExponential"},
        )
        model_coefs = [
            model.get_param_dict()[k] for k in ["scale_N", "exponent_N", "scale_P", "exponent_P"]
        ]
        assert np.allclose(exp_coefs, model_coefs)

    # Check complex exponential model building [EXPERIMENTAL: Just one working test example, since model fitting not so robust]
    exp_coefs = [
        1e-1,
        1e-4,
        2.0,
        1e-1,
        1e-4,
        2e-1,
        2e-4,
        1.75,
        2e-1,
        2e-4,
    ]  # 'prox_scale_N', 'prox_exp_N', 'prox_exp_pow_N', 'dist_scale_N', 'dist_exp_N', 'prox_scale_P', 'prox_exp_P', 'prox_exp_pow_P', 'dist_scale_P', 'dist_exp_P'
    exp_model = lambda x, a, b, c, d, e: a * np.exp(-b * np.array(x) ** c) + d * np.exp(
        -e * np.array(x)
    )
    exp_data = np.array([exp_model(d, *exp_coefs[:5]), exp_model(d, *exp_coefs[5:])]).T

    model = test_module.build_3rd_order(
        exp_data,
        dist_bins,
        np.zeros_like(exp_data),
        bip_coord_data=0,
        model_specs={"type": "ComplexExponential"},
    )
    model_coefs = [
        model.get_param_dict()[k]
        for k in [
            "prox_scale_N",
            "prox_exp_N",
            "prox_exp_pow_N",
            "dist_scale_N",
            "dist_exp_N",
            "prox_scale_P",
            "prox_exp_P",
            "prox_exp_pow_P",
            "dist_scale_P",
            "dist_exp_P",
        ]
    ]
    assert np.allclose(exp_coefs, model_coefs)


def test_extract_4th_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 200
    np.random.seed(0)
    for rep in range(5):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[1].positions(tgt_sel)
        offmat = np.array(
            [
                [
                    [tgt_pos[coord].loc[t] - src_pos[coord].loc[s] for coord in ["x", "y", "z"]]
                    for t in tgt_sel
                ]
                for s in src_sel
            ]
        )
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        ranges = [
            [np.nanmin(offmat[:, :, coord]), np.nanmax(offmat[:, :, coord])]
            for coord in range(offmat.shape[2])
        ]
        num_bins = [
            np.ceil((ranges[coord][1] - ranges[coord][0]) / bin_size_um).astype(int)
            for coord in range(len(ranges))
        ]
        off_bins = [
            np.arange(0, num_bins[coord] + 1) * bin_size_um + ranges[coord][0]
            for coord in range(len(ranges))
        ]

        conn_cnt = np.full(num_bins, -1)  # Conn. count
        all_cnt = np.full(num_bins, -1)  # All pair count
        for xidx in range(num_bins[0]):
            for yidx in range(num_bins[1]):
                for zidx in range(num_bins[2]):
                    xsel = np.logical_and(
                        offmat[:, :, 0] >= off_bins[0][xidx],
                        offmat[:, :, 0] < off_bins[0][xidx + 1],
                    )
                    ysel = np.logical_and(
                        offmat[:, :, 1] >= off_bins[1][yidx],
                        offmat[:, :, 1] < off_bins[1][yidx + 1],
                    )
                    zsel = np.logical_and(
                        offmat[:, :, 2] >= off_bins[2][zidx],
                        offmat[:, :, 2] < off_bins[2][zidx + 1],
                    )
                    sel = np.logical_and(np.logical_and(xsel, ysel), zsel)
                    conn_cnt[xidx, yidx, zidx] = np.sum(nconn[sel])
                    all_cnt[xidx, yidx, zidx] = np.sum(sel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_4th_order(
                nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_offset"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.all(
                [
                    np.allclose(res[k], off_bins[i])
                    for i, k in enumerate(["dx_bins", "dy_bins", "dz_bins"])
                ]
            )
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_4th_order():
    np.random.seed(0)
    bin_sizes = [100, 200, 400]
    dx_bins = np.arange(0, 1001, bin_sizes[0])
    dy_bins = np.arange(0, 1001, bin_sizes[1])
    dz_bins = np.arange(0, 1001, bin_sizes[2])
    dx = np.array([np.mean(dx_bins[i : i + 2]) for i in range(len(dx_bins) - 1)])
    dy = np.array([np.mean(dy_bins[i : i + 2]) for i in range(len(dy_bins) - 1)])
    dz = np.array([np.mean(dz_bins[i : i + 2]) for i in range(len(dz_bins) - 1)])
    p = 1e-2 * np.random.rand(len(dx_bins) - 1, len(dy_bins) - 1, len(dz_bins) - 1)

    # Check linear interpolation model building
    model = test_module.build_4th_order(
        p, dx_bins, dy_bins, dz_bins, np.zeros_like(p), model_specs={"type": "LinearInterpolation"}
    )
    p_model = np.array(
        [
            [[model.get_conn_prob(dx=_dx, dy=_dy, dz=_dz)[0] for _dz in dz] for _dy in dy]
            for _dx in dx
        ]
    )
    assert np.allclose(p, p_model)

    # Check linear interpolation model building with Gaussian filtering
    smoothing_sigma_um = 100.0
    sigmas = [smoothing_sigma_um / b for b in bin_sizes]
    p_filt = gaussian_filter(p, sigmas, mode="constant")
    model = test_module.build_4th_order(
        p,
        dx_bins,
        dy_bins,
        dz_bins,
        np.zeros_like(p),
        model_specs={"type": "LinearInterpolation"},
        smoothing_sigma_um=smoothing_sigma_um,
    )
    p_model = np.array(
        [
            [[model.get_conn_prob(dx=_dx, dy=_dy, dz=_dz)[0] for _dz in dz] for _dy in dy]
            for _dx in dx
        ]
    )
    assert np.allclose(p_filt, p_model)

    # Check random forest regressor model building [Not yet implemented]
    with pytest.raises(AssertionError, match='ERROR: Model type "RandomForestRegressor" unknown!'):
        model = test_module.build_4th_order(
            p,
            dx_bins,
            dy_bins,
            dz_bins,
            np.zeros_like(p),
            model_specs={"type": "RandomForestRegressor"},
        )


def test_extract_4th_order_reduced():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    bin_size_um = 200
    np.random.seed(0)
    for rep in range(5):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        offmat = np.array(
            [
                [
                    [tgt_pos[coord].loc[t] - src_pos[coord].loc[s] for coord in ["x", "y", "z"]]
                    for t in tgt_sel
                ]
                for s in src_sel
            ]
        )
        offmat = np.stack(
            [np.sqrt(offmat[:, :, 0] ** 2 + offmat[:, :, 1] ** 2), offmat[:, :, 2]], axis=2
        )
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        ranges = [
            [0.0, np.nanmax(offmat[:, :, 0])],
            [np.nanmin(offmat[:, :, 1]), np.nanmax(offmat[:, :, 1])],
        ]
        num_bins = [
            np.ceil((ranges[coord][1] - ranges[coord][0]) / bin_size_um).astype(int)
            for coord in range(len(ranges))
        ]
        off_bins = [
            np.arange(0, num_bins[coord] + 1) * bin_size_um + ranges[coord][0]
            for coord in range(len(ranges))
        ]

        conn_cnt = np.full(num_bins, -1)  # Conn. count
        all_cnt = np.full(num_bins, -1)  # All pair count
        for ridx in range(num_bins[0]):
            for zidx in range(num_bins[1]):
                rsel = np.logical_and(
                    offmat[:, :, 0] >= off_bins[0][ridx], offmat[:, :, 0] < off_bins[0][ridx + 1]
                )
                zsel = np.logical_and(
                    offmat[:, :, 1] >= off_bins[1][zidx], offmat[:, :, 1] < off_bins[1][zidx + 1]
                )
                sel = np.logical_and(rsel, zsel)
                conn_cnt[ridx, zidx] = np.sum(nconn[sel])
                all_cnt[ridx, zidx] = np.sum(sel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_4th_order_reduced(
                nodes, edges, src_sel, tgt_sel, bin_size_um=bin_size_um, min_count_per_bin=min_nbins
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_offset"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.all(
                [np.allclose(res[k], off_bins[i]) for i, k in enumerate(["dr_bins", "dz_bins"])]
            )
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_4th_order_reduced():
    np.random.seed(0)
    bin_sizes = [100, 200]
    dr_bins = np.arange(0, 1001, bin_sizes[0])
    dz_bins = np.arange(0, 1001, bin_sizes[1])
    dr = np.array([np.mean(dr_bins[i : i + 2]) for i in range(len(dr_bins) - 1)])
    dz = np.array([np.mean(dz_bins[i : i + 2]) for i in range(len(dz_bins) - 1)])
    p = 1e-2 * np.random.rand(len(dr_bins) - 1, len(dz_bins) - 1)

    # Check linear interpolation model building
    model = test_module.build_4th_order_reduced(
        p,
        dr_bins,
        dz_bins,
        np.zeros_like(p),
        axial_coord_data=0,
        model_specs={"type": "LinearInterpolation"},
    )
    p_model = np.array([[model.get_conn_prob(dr=_dr, dz=_dz)[0] for _dz in dz] for _dr in dr])
    assert np.allclose(p, p_model)

    # Check linear interpolation model building with Gaussian filtering
    smoothing_sigma_um = 100.0
    sigmas = [smoothing_sigma_um / b for b in bin_sizes]
    p_reflect = np.vstack(
        [p[::-1, :], p]
    )  # Mirror along radial axis at dr==0, to avoid edge effect
    p_reflect = gaussian_filter(p_reflect, sigmas, mode="constant")
    p_filt = p_reflect[p.shape[0] :, :]  # Cut original part of the data
    model = test_module.build_4th_order_reduced(
        p,
        dr_bins,
        dz_bins,
        np.zeros_like(p),
        axial_coord_data=0,
        model_specs={"type": "LinearInterpolation"},
        smoothing_sigma_um=smoothing_sigma_um,
    )
    p_model = np.array([[model.get_conn_prob(dr=_dr, dz=_dz)[0] for _dz in dz] for _dr in dr])
    assert np.allclose(p_filt, p_model)

    # Check random forest regressor model building [Not yet implemented]
    with pytest.raises(AssertionError, match='ERROR: Model type "RandomForestRegressor" unknown!'):
        model = test_module.build_4th_order_reduced(
            p,
            dr_bins,
            dz_bins,
            np.zeros_like(p),
            axial_coord_data=0,
            model_specs={"type": "RandomForestRegressor"},
        )


def test_extract_5th_order():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    off_bin_size_um = 500
    pos_bin_size_um = 1000
    np.random.seed(0)
    for rep in range(5):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        posmat = np.array(
            [np.repeat(src_pos[[coord]].to_numpy(), ntgt, axis=1).T for coord in ["x", "y", "z"]]
        ).T
        posmat2 = np.array(
            [np.repeat(tgt_pos[[coord]].to_numpy(), ntgt, axis=1).T for coord in ["x", "y", "z"]]
        ).T
        offmat = np.array(
            [
                [
                    [tgt_pos[coord].loc[t] - src_pos[coord].loc[s] for coord in ["x", "y", "z"]]
                    for t in tgt_sel
                ]
                for s in src_sel
            ]
        )
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        pos_ranges = [
            [
                np.minimum(np.nanmin(posmat[:, :, coord]), np.nanmin(posmat2[:, :, coord])),
                np.maximum(np.nanmax(posmat[:, :, coord]), np.nanmax(posmat2[:, :, coord])),
            ]
            for coord in range(posmat.shape[2])
        ]  # Set range based on pre- AND post-neurons
        off_ranges = [
            [np.nanmin(offmat[:, :, coord]), np.nanmax(offmat[:, :, coord])]
            for coord in range(offmat.shape[2])
        ]
        pos_num_bins = [
            np.ceil((pos_ranges[coord][1] - pos_ranges[coord][0]) / pos_bin_size_um).astype(int)
            for coord in range(len(pos_ranges))
        ]
        off_num_bins = [
            np.ceil((off_ranges[coord][1] - off_ranges[coord][0]) / off_bin_size_um).astype(int)
            for coord in range(len(off_ranges))
        ]
        pos_bins = [
            np.arange(0, pos_num_bins[coord] + 1) * pos_bin_size_um + pos_ranges[coord][0]
            for coord in range(len(pos_ranges))
        ]
        off_bins = [
            np.arange(0, off_num_bins[coord] + 1) * off_bin_size_um + off_ranges[coord][0]
            for coord in range(len(off_ranges))
        ]

        conn_cnt = np.full(pos_num_bins + off_num_bins, -1)  # Conn. count
        all_cnt = np.full(pos_num_bins + off_num_bins, -1)  # All pair count
        for xidx in range(pos_num_bins[0]):
            for yidx in range(pos_num_bins[1]):
                for zidx in range(pos_num_bins[2]):
                    for dxidx in range(off_num_bins[0]):
                        for dyidx in range(off_num_bins[1]):
                            for dzidx in range(off_num_bins[2]):
                                xsel = np.logical_and(
                                    posmat[:, :, 0] >= pos_bins[0][xidx],
                                    posmat[:, :, 0] < pos_bins[0][xidx + 1],
                                )
                                ysel = np.logical_and(
                                    posmat[:, :, 1] >= pos_bins[1][yidx],
                                    posmat[:, :, 1] < pos_bins[1][yidx + 1],
                                )
                                zsel = np.logical_and(
                                    posmat[:, :, 2] >= pos_bins[2][zidx],
                                    posmat[:, :, 2] < pos_bins[2][zidx + 1],
                                )
                                dxsel = np.logical_and(
                                    offmat[:, :, 0] >= off_bins[0][dxidx],
                                    offmat[:, :, 0] < off_bins[0][dxidx + 1],
                                )
                                dysel = np.logical_and(
                                    offmat[:, :, 1] >= off_bins[1][dyidx],
                                    offmat[:, :, 1] < off_bins[1][dyidx + 1],
                                )
                                dzsel = np.logical_and(
                                    offmat[:, :, 2] >= off_bins[2][dzidx],
                                    offmat[:, :, 2] < off_bins[2][dzidx + 1],
                                )
                                sel = np.logical_and(
                                    np.logical_and(
                                        np.logical_and(xsel, dxsel), np.logical_and(ysel, dysel)
                                    ),
                                    np.logical_and(zsel, dzsel),
                                )
                                conn_cnt[xidx, yidx, zidx, dxidx, dyidx, dzidx] = np.sum(nconn[sel])
                                all_cnt[xidx, yidx, zidx, dxidx, dyidx, dzidx] = np.sum(sel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_5th_order(
                nodes,
                edges,
                src_sel,
                tgt_sel,
                position_bin_size_um=pos_bin_size_um,
                offset_bin_size_um=off_bin_size_um,
                min_count_per_bin=min_nbins,
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_position"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.all(
                [
                    np.array_equal(res[k], pos_bins[i])
                    for i, k in enumerate(["x_bins", "y_bins", "z_bins"])
                ]
            )
            assert np.all(
                [
                    np.allclose(res[k], off_bins[i])
                    for i, k in enumerate(["dx_bins", "dy_bins", "dz_bins"])
                ]
            )
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_5th_order():
    np.random.seed(0)
    bin_sizes = [500, 500, 500, 200, 300, 400]
    x_bins = np.arange(0, 1001, bin_sizes[0])
    y_bins = np.arange(0, 1001, bin_sizes[1])
    z_bins = np.arange(0, 2001, bin_sizes[2])
    dx_bins = np.arange(0, 1001, bin_sizes[3])
    dy_bins = np.arange(0, 1001, bin_sizes[4])
    dz_bins = np.arange(0, 1001, bin_sizes[5])
    x = np.array([np.mean(x_bins[i : i + 2]) for i in range(len(x_bins) - 1)])
    y = np.array([np.mean(y_bins[i : i + 2]) for i in range(len(y_bins) - 1)])
    z = np.array([np.mean(z_bins[i : i + 2]) for i in range(len(z_bins) - 1)])
    dx = np.array([np.mean(dx_bins[i : i + 2]) for i in range(len(dx_bins) - 1)])
    dy = np.array([np.mean(dy_bins[i : i + 2]) for i in range(len(dy_bins) - 1)])
    dz = np.array([np.mean(dz_bins[i : i + 2]) for i in range(len(dz_bins) - 1)])
    p = 1e-2 * np.random.rand(
        len(x_bins) - 1,
        len(y_bins) - 1,
        len(z_bins) - 1,
        len(dx_bins) - 1,
        len(dy_bins) - 1,
        len(dz_bins) - 1,
    )

    # Check linear interpolation model building
    model = test_module.build_5th_order(
        p,
        x_bins,
        y_bins,
        z_bins,
        dx_bins,
        dy_bins,
        dz_bins,
        np.zeros_like(p),
        model_specs={"type": "LinearInterpolation"},
    )
    p_model = np.array(
        [
            [
                [
                    [
                        [
                            [
                                model.get_conn_prob(x=_x, y=_y, z=_z, dx=_dx, dy=_dy, dz=_dz)[0]
                                for _dz in dz
                            ]
                            for _dy in dy
                        ]
                        for _dx in dx
                    ]
                    for _z in z
                ]
                for _y in y
            ]
            for _x in x
        ]
    )
    assert np.allclose(p, p_model)

    # Check linear interpolation model building with Gaussian filtering
    smoothing_sigma_um = 100.0
    sigmas = [smoothing_sigma_um / b for b in bin_sizes]
    p_filt = gaussian_filter(p, sigmas, mode="constant")
    model = test_module.build_5th_order(
        p,
        x_bins,
        y_bins,
        z_bins,
        dx_bins,
        dy_bins,
        dz_bins,
        np.zeros_like(p),
        model_specs={"type": "LinearInterpolation"},
        smoothing_sigma_um=smoothing_sigma_um,
    )
    p_model = np.array(
        [
            [
                [
                    [
                        [
                            [
                                model.get_conn_prob(x=_x, y=_y, z=_z, dx=_dx, dy=_dy, dz=_dz)[0]
                                for _dz in dz
                            ]
                            for _dy in dy
                        ]
                        for _dx in dx
                    ]
                    for _z in z
                ]
                for _y in y
            ]
            for _x in x
        ]
    )
    assert np.allclose(p_filt, p_model)

    # Check random forest regressor model building [Not yet implemented]
    with pytest.raises(AssertionError, match='ERROR: Model type "RandomForestRegressor" unknown!'):
        model = test_module.build_5th_order(
            p,
            x_bins,
            y_bins,
            z_bins,
            dx_bins,
            dy_bins,
            dz_bins,
            np.zeros_like(p),
            model_specs={"type": "RandomForestRegressor"},
        )


def test_extract_5th_order_reduced():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()

    off_bin_size_um = 250
    pos_bin_size_um = 1000
    np.random.seed(0)
    for rep in range(5):
        src_sel = np.random.choice(src_ids, np.random.choice(len(src_ids)) + 1, replace=False)
        tgt_sel = np.random.choice(tgt_ids, np.random.choice(len(tgt_ids)) + 1, replace=False)
        nsrc = len(src_sel)
        ntgt = len(tgt_sel)

        src_pos = nodes[0].positions(src_sel)
        tgt_pos = nodes[0].positions(tgt_sel)
        posmat = np.repeat(src_pos[["z"]].to_numpy(), ntgt, axis=1)
        posmat2 = np.repeat(tgt_pos[["z"]].to_numpy(), ntgt, axis=1)
        offmat = np.array(
            [
                [
                    [tgt_pos[coord].loc[t] - src_pos[coord].loc[s] for coord in ["x", "y", "z"]]
                    for t in tgt_sel
                ]
                for s in src_sel
            ]
        )
        offmat = np.stack(
            [np.sqrt(offmat[:, :, 0] ** 2 + offmat[:, :, 1] ** 2), offmat[:, :, 2]], axis=2
        )
        nconn = np.array(
            [
                [len(list(edges.iter_connections(source=s, target=t))) for t in tgt_sel]
                for s in src_sel
            ]
        )  # Number of connection matrix

        pos_ranges = [
            np.minimum(np.nanmin(posmat), np.nanmin(posmat2)),
            np.maximum(np.nanmax(posmat), np.nanmax(posmat2)),
        ]  # Set range based on pre- AND post-neurons
        off_ranges = [
            [0.0, np.nanmax(offmat[:, :, 0])],
            [np.nanmin(offmat[:, :, 1]), np.nanmax(offmat[:, :, 1])],
        ]
        pos_num_bins = np.ceil((pos_ranges[1] - pos_ranges[0]) / pos_bin_size_um).astype(int)
        off_num_bins = [
            np.ceil((off_ranges[coord][1] - off_ranges[coord][0]) / off_bin_size_um).astype(int)
            for coord in range(len(off_ranges))
        ]
        pos_bins = np.arange(0, pos_num_bins + 1) * pos_bin_size_um + pos_ranges[0]
        off_bins = [
            np.arange(0, off_num_bins[coord] + 1) * off_bin_size_um + off_ranges[coord][0]
            for coord in range(len(off_ranges))
        ]

        conn_cnt = np.full([pos_num_bins] + off_num_bins, -1)  # Conn. count
        all_cnt = np.full([pos_num_bins] + off_num_bins, -1)  # All pair count
        for zidx in range(pos_num_bins):
            for dridx in range(off_num_bins[0]):
                for dzidx in range(off_num_bins[1]):
                    zsel = np.logical_and(posmat >= pos_bins[zidx], posmat < pos_bins[zidx + 1])
                    drsel = np.logical_and(
                        offmat[:, :, 0] >= off_bins[0][dridx],
                        offmat[:, :, 0] < off_bins[0][dridx + 1],
                    )
                    dzsel = np.logical_and(
                        offmat[:, :, 1] >= off_bins[1][dzidx],
                        offmat[:, :, 1] < off_bins[1][dzidx + 1],
                    )
                    sel = np.logical_and(zsel, np.logical_and(drsel, dzsel))
                    conn_cnt[zidx, dridx, dzidx] = np.sum(nconn[sel])
                    all_cnt[zidx, dridx, dzidx] = np.sum(sel)
        p = conn_cnt / all_cnt  # Conn. prob.

        for min_nbins in [1, 3, 5]:
            res = test_module.extract_5th_order_reduced(
                nodes,
                edges,
                src_sel,
                tgt_sel,
                position_bin_size_um=pos_bin_size_um,
                offset_bin_size_um=off_bin_size_um,
                min_count_per_bin=min_nbins,
            )
            p_sel = p.copy()
            p_sel[all_cnt < min_nbins] = np.nan
            assert np.array_equal(res["p_conn_position"], p_sel, equal_nan=True)
            assert np.array_equal(res["count_conn"], conn_cnt)
            assert np.array_equal(res["count_all"], all_cnt)
            assert np.array_equal(res["z_bins"], pos_bins)
            assert np.all(
                [np.allclose(res[k], off_bins[i]) for i, k in enumerate(["dr_bins", "dz_bins"])]
            )
            assert res["src_cell_count"] == nsrc
            assert res["tgt_cell_count"] == ntgt


def test_build_5th_order_reduced():
    np.random.seed(0)
    bin_sizes = [500, 100, 200]
    z_bins = np.arange(0, 2001, bin_sizes[0])
    dr_bins = np.arange(0, 1001, bin_sizes[1])
    dz_bins = np.arange(0, 1001, bin_sizes[2])
    z = np.array([np.mean(z_bins[i : i + 2]) for i in range(len(z_bins) - 1)])
    dr = np.array([np.mean(dr_bins[i : i + 2]) for i in range(len(dr_bins) - 1)])
    dz = np.array([np.mean(dz_bins[i : i + 2]) for i in range(len(dz_bins) - 1)])
    p = 1e-2 * np.random.rand(len(z_bins) - 1, len(dr_bins) - 1, len(dz_bins) - 1)

    # Check linear interpolation model building
    model = test_module.build_5th_order_reduced(
        p,
        z_bins,
        dr_bins,
        dz_bins,
        np.zeros_like(p),
        axial_coord_data=0,
        model_specs={"type": "LinearInterpolation"},
    )
    p_model = np.array(
        [[[model.get_conn_prob(z=_z, dr=_dr, dz=_dz)[0] for _dz in dz] for _dr in dr] for _z in z]
    )
    assert np.allclose(p, p_model)

    # Check linear interpolation model building with Gaussian filtering
    smoothing_sigma_um = 100.0
    sigmas = [smoothing_sigma_um / b for b in bin_sizes]
    p_reflect = np.concatenate(
        [p[:, ::-1, :], p], axis=1
    )  # Mirror along radial axis at dr==0, to avoid edge effect
    p_reflect = gaussian_filter(p_reflect, sigmas, mode="constant")
    p_filt = p_reflect[:, p.shape[1] :, :]  # Cut original part of the data
    model = test_module.build_5th_order_reduced(
        p,
        z_bins,
        dr_bins,
        dz_bins,
        np.zeros_like(p),
        axial_coord_data=0,
        model_specs={"type": "LinearInterpolation"},
        smoothing_sigma_um=smoothing_sigma_um,
    )
    p_model = np.array(
        [[[model.get_conn_prob(z=_z, dr=_dr, dz=_dz)[0] for _dz in dz] for _dr in dr] for _z in z]
    )
    assert np.allclose(p_filt, p_model)

    # Check random forest regressor model building [Not yet implemented]
    with pytest.raises(AssertionError, match='ERROR: Model type "RandomForestRegressor" unknown!'):
        model = test_module.build_5th_order_reduced(
            p,
            z_bins,
            dr_bins,
            dz_bins,
            np.zeros_like(p),
            axial_coord_data=0,
            model_specs={"type": "RandomForestRegressor"},
        )
