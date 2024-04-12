# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from bluepysnap import Circuit
import pytest
from utils import setup_tempdir, TEST_DATA_DIR

import connectome_manipulator.model_building.pos_mapping_from_table as test_module


def test_extract():
    np.random.seed(0)
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = c.nodes["nodeA"]
    node_ids = nodes.ids()
    with setup_tempdir(__name__) as tempdir:
        N = 100
        coord_names = ["xpos", "ypos", "zpos"]

        # Check basic extraction (zero-based indexing)
        pos_tab = pd.DataFrame(np.random.rand(N, len(coord_names)), columns=coord_names)
        pos_file = os.path.join(tempdir, "pos_tab_TEST.feather")
        pos_tab.to_feather(pos_file)

        res = test_module.extract(
            c,
            pos_file,
            coord_names,
            coord_scale=None,
            nodes_pop_name=None,
            nodes_spec=None,
            zero_based_indexing=True,
            gid_column=None,
        )
        assert_array_equal(res["nrn_ids"], node_ids)
        assert_array_equal(res["map_pos"], pos_tab[coord_names].loc[node_ids])
        assert_array_equal(res["nrn_pos"], nodes.positions(node_ids))
        assert_array_equal(res["nrn_lay"], nodes.get(node_ids, properties="layer"))

        # Check coordinate scaling (zero-based indexing)
        coord_scale = np.random.rand(len(coord_names))

        res = test_module.extract(
            c,
            pos_file,
            coord_names,
            coord_scale=coord_scale,
            nodes_pop_name=None,
            nodes_spec=None,
            zero_based_indexing=True,
            gid_column=None,
        )
        assert_array_equal(res["nrn_ids"], node_ids)
        assert_array_equal(res["map_pos"], pos_tab[coord_names].loc[node_ids] * coord_scale)
        assert_array_equal(res["nrn_pos"], nodes.positions(node_ids))
        assert_array_equal(res["nrn_lay"], nodes.get(node_ids, properties="layer"))

        # Check extraction using GID column (zero-based indexing)
        pos_tab = pd.DataFrame(
            np.hstack([np.random.rand(N, len(coord_names)), np.arange(N).reshape([N, 1])]),
            columns=coord_names + ["gid"],
        )
        pos_tab = pos_tab.astype({"gid": int})
        pos_file = os.path.join(tempdir, "pos_tab_TEST.feather")
        pos_tab.to_feather(pos_file)

        res = test_module.extract(
            c,
            pos_file,
            coord_names,
            coord_scale=None,
            nodes_pop_name=None,
            nodes_spec=None,
            zero_based_indexing=True,
            gid_column="gid",
        )
        assert_array_equal(res["nrn_ids"], node_ids)
        assert_array_equal(res["map_pos"], pos_tab[coord_names].loc[node_ids])
        assert_array_equal(res["nrn_pos"], nodes.positions(node_ids))
        assert_array_equal(res["nrn_lay"], nodes.get(node_ids, properties="layer"))

        # Check extraction using GID column (one-based indexing)
        pos_tab = pd.DataFrame(
            np.hstack([np.random.rand(N, len(coord_names)), np.arange(1, N + 1).reshape([N, 1])]),
            columns=coord_names + ["gid"],
        )
        pos_tab = pos_tab.astype({"gid": int})
        pos_file = os.path.join(tempdir, "pos_tab_TEST.feather")
        pos_tab.to_feather(pos_file)

        res = test_module.extract(
            c,
            pos_file,
            coord_names,
            coord_scale=None,
            nodes_pop_name=None,
            nodes_spec=None,
            zero_based_indexing=False,
            gid_column="gid",
        )
        assert_array_equal(res["nrn_ids"], node_ids)
        assert_array_equal(res["map_pos"], pos_tab[coord_names].loc[node_ids])
        assert_array_equal(res["nrn_pos"], nodes.positions(node_ids))
        assert_array_equal(res["nrn_lay"], nodes.get(node_ids, properties="layer"))


def test_build():
    N = 100
    coord_names = ["xpos", "ypos", "zpos"]
    nrn_ids = np.arange(N) + 5
    map_pos = np.random.rand(N, len(coord_names))

    # Check pos mapping model
    model = test_module.build(nrn_ids, coord_names, map_pos, model_coord_names=None)
    assert_array_equal(model.get_coord_names(), coord_names)
    assert_array_equal(model.get_gids(), nrn_ids)
    assert_array_equal(model.apply(gids=nrn_ids), map_pos)

    # Check model coordinate names
    model_coord_names = ["a", "b", "c"]
    model = test_module.build(nrn_ids, coord_names, map_pos, model_coord_names=model_coord_names)
    assert_array_equal(model.get_coord_names(), model_coord_names)
    assert_array_equal(model.get_gids(), nrn_ids)
    assert_array_equal(model.apply(gids=nrn_ids), map_pos)

    # Check error handling (NaN)
    map_pos = np.random.rand(N, len(coord_names))
    map_pos[np.random.choice(map_pos.shape[0]), np.random.choice(map_pos.shape[1])] = np.nan
    with pytest.raises(AssertionError, match="ERROR: Invalid mapped positions found!"):
        model = test_module.build(nrn_ids, coord_names, map_pos, model_coord_names=None)

    # Check error handling (Inf)
    map_pos = np.random.rand(N, len(coord_names))
    map_pos[np.random.choice(map_pos.shape[0]), np.random.choice(map_pos.shape[1])] = np.inf
    with pytest.raises(AssertionError, match="ERROR: Invalid mapped positions found!"):
        model = test_module.build(nrn_ids, coord_names, map_pos, model_coord_names=None)
