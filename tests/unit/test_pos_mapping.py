# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from bluepysnap import Circuit
from utils import setup_tempdir, TEST_DATA_DIR
from scipy.interpolate import griddata
from voxcell.nexus.voxelbrain import Atlas

import connectome_manipulator.model_building.pos_mapping as test_module


def test_extract():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nrn_pos = circuit.nodes["nodeA"].positions()
    nrn_ids = circuit.nodes["nodeA"].ids()
    flatmap_path = TEST_DATA_DIR

    # Check scaling
    xy_file = "xy_map_ones"
    z_file = "depth_map_ones"
    np.random.seed(0)
    for scale in np.random.rand(10, 3):
        res = test_module.extract(
            circuit, flatmap_path, xy_file, z_file, xy_scale=scale[:2], z_scale=scale[2]
        )
        assert_array_equal(nrn_ids, res["nrn_ids"])
        assert_array_equal(nrn_pos, res["nrn_pos"])
        assert np.all([np.allclose(res["flat_pos"][:, i], scale[i]) for i in range(len(scale))])

    # Check default scaling
    xy_file = "xy_map_ones"
    z_file = "depth_map_ones"
    xymap = Atlas.open(TEST_DATA_DIR).load_data(xy_file)

    res = test_module.extract(circuit, flatmap_path, xy_file, z_file, xy_scale=None, z_scale=None)
    np.testing.assert_allclose(res["flat_pos"][:, 0], xymap.voxel_dimensions[0])
    np.testing.assert_allclose(res["flat_pos"][:, 1], xymap.voxel_dimensions[1])
    np.testing.assert_allclose(res["flat_pos"][:, 2], 1.0)

    # Check interpolation (along z-axis only)
    xy_file = "xy_map_ones"
    z_file = "depth_map_lin"
    dmap = Atlas.open(TEST_DATA_DIR).load_data(z_file)

    ## (a) Nearest-neighbor interpolation
    zidx = dmap.positions_to_indices(nrn_pos.to_numpy(), keep_fraction=False)
    zval = zidx[:, 2].astype(float)  # Z value is same as index in this flatmap
    res = test_module.extract(
        circuit, flatmap_path, xy_file, z_file, xy_scale=[1.0, 1.0], z_scale=1.0, NN_only=True
    )
    assert_array_equal(nrn_ids, res["nrn_ids"])
    assert_array_equal(nrn_pos, res["nrn_pos"])
    assert np.all([np.all(res["flat_pos"][:, i] == 1.0) for i in range(2)])
    assert_array_equal(res["flat_pos"][:, 2], zval)

    ## (b) Linear interpolation using griddata
    zidx = dmap.positions_to_indices(nrn_pos.to_numpy(), keep_fraction=True)
    zval_lin = griddata(np.floor(zidx) + 0.5, np.floor(zidx), zidx)[
        :, 2
    ]  # Data points are at bin centers
    assert np.any(np.isfinite(zval_lin)), "ERROR: Linear interpolation on test data not possible!"
    zval_lin2 = np.interp(
        zidx[:, 2], np.sort(np.floor(zidx[:, 2])) + 0.5, np.sort(np.floor(zidx[:, 2]))
    )  # Interpolate linearly using numpy
    assert_allclose(
        zval_lin[np.isfinite(zval_lin)], zval_lin2[np.isfinite(zval_lin)]
    )  # Check that linearly interpolated along z-axis indeed
    zval[np.isfinite(zval_lin)] = zval_lin[
        np.isfinite(zval_lin)
    ]  # Use lin. interpolation if possible, otherwise keep NN

    res = test_module.extract(
        circuit, flatmap_path, xy_file, z_file, xy_scale=[1.0, 1.0], z_scale=1.0, NN_only=False
    )
    assert_array_equal(nrn_ids, res["nrn_ids"])
    assert_array_equal(nrn_pos, res["nrn_pos"])
    assert np.all([np.all(res["flat_pos"][:, i] == 1.0) for i in range(2)])
    assert_allclose(res["flat_pos"][:, 2], zval)


def test_build():
    np.random.seed(0)
    nrn_ids = np.arange(10)
    flat_pos = 1e3 * np.random.rand(len(nrn_ids), 3)

    model = test_module.build(nrn_ids, flat_pos)
    assert_array_equal(model.apply(gids=nrn_ids), flat_pos)
