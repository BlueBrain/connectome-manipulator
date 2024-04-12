# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bluepysnap import Circuit
from utils import TEST_DATA_DIR

import connectome_manipulator.model_building.delay as test_module


def test_extract():
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    bin_size_um = 500
    max_range_um = 2500

    # Extract delays in given bins
    src_pos = nodes[0].positions(edges_table["@source_node"].to_numpy()).to_numpy()
    tgt_pos = edges_table[
        ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
    ].to_numpy()
    dist = np.sqrt(np.sum((src_pos - tgt_pos) ** 2, 1))
    bins = np.arange(0, max_range_um + 1, bin_size_um)
    delays = [
        edges_table[np.logical_and(dist >= bins[idx], dist < bins[idx + 1])]["delay"].to_numpy()
        for idx in range(len(bins) - 1)
    ]

    res = test_module.extract(
        c, bin_size_um, max_range_um, sel_src=None, sel_dest=None, sample_size=None
    )

    assert np.sum(res["dist_count"]) == edges_table.shape[0]
    assert_array_equal(res["dist_bins"], bins)
    assert_array_equal(res["dist_count"], [len(d) for d in delays])
    assert_array_equal(res["dist_delays_mean"], [np.mean(d) for d in delays])
    assert_array_equal(res["dist_delays_std"], [np.std(d) for d in delays])
    assert res["dist_delay_min"] == np.min(np.hstack(delays))


def test_build():
    bin_size_um = 500
    max_range_um = 2500
    bins = np.arange(0, max_range_um + 1, bin_size_um)

    # Check delay model
    np.random.rand(0)
    dist = bins[:-1] + 0.5 * bin_size_um  # Bin centers
    d_coef = [0.1, 0.003]
    d_mean = d_coef[0] + d_coef[1] * dist
    d_std = np.random.rand(len(dist))
    d_min = np.random.rand()

    model = test_module.build(bins, d_mean, d_std, d_min, bin_size_um)
    model_dict = model.get_param_dict()

    assert_array_almost_equal(
        (model_dict["delay_mean_coeff_a"], model_dict["delay_mean_coeff_b"]), d_coef
    )
    assert np.isclose(model_dict["delay_std"], np.mean(d_std))
    assert model_dict["delay_min"] == d_min
