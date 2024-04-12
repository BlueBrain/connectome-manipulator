# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import re
import json
import pytest
from pathlib import Path
from cached_property import cached_property
from connectome_manipulator import utils as test_module
from bluepysnap.circuit import Circuit

from utils import TEST_DATA_DIR


@pytest.mark.parametrize(
    "base_dir, path, expected",
    [
        ("/a/b", "$c", "$c"),
        ("/a/b", "", ""),
        ("/a/b", "/c/d", "/c/d"),
        ("/a/b", "/a/b/c", "$BASE_DIR/c"),
        ("/a/b", "/a/b/c/d", "$BASE_DIR/c/d"),
        ("/a/b/c", "/a/b/c/d", "$BASE_DIR/d"),
        ("/a/b", "./c", "$BASE_DIR/c"),
        ("/a/b", "c/d", "$BASE_DIR/c/d"),
    ],
)
def tested_reduce_path(base_dir, path, expected):
    result = test_module._reduce_path(path, Path(base_dir))
    assert result == expected


def test_reduce_config_paths__raises_relative_config_dir():
    expected = re.escape("Circuit config's directory is not absolute: .")
    with pytest.raises(ValueError, match=expected):
        test_module.reduce_config_paths(None, ".")


def test_reduce_config_paths__not_resolved_config():
    config_path = Path(TEST_DATA_DIR, "circuit_sonata.json")
    config = json.loads(config_path.read_bytes())
    contains = "A reduced config with absolute paths and no manifest must be provided."
    with pytest.raises(ValueError, match=contains):
        test_module.reduce_config_paths(config, "/")


def test_reduce_config_paths():
    config_path = Path(TEST_DATA_DIR, "circuit_sonata.json")

    reduced_config = Circuit(config_path).config

    res = test_module.reduce_config_paths(reduced_config, TEST_DATA_DIR)

    assert res["version"] == 2
    assert res["node_sets_file"] == "$BASE_DIR/node_sets.json"
    assert res["networks"] == {
        "nodes": [
            {
                "nodes_file": "$BASE_DIR/nodes.h5",
                "populations": {
                    "nodeA": {
                        "type": "biophysical",
                        "morphologies_dir": "$BASE_DIR/swc",
                        "biophysical_neuron_models_dir": "$BASE_DIR",
                    },
                },
            },
        ],
        "edges": [
            {
                "edges_file": "$BASE_DIR/edges.h5",
                "populations": {"nodeA__nodeA__chemical": {"type": "chemical"}},
            },
        ],
    }
