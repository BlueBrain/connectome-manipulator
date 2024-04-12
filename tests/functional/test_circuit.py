# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

from dataclasses import dataclass
from pathlib import Path
import json

from click.testing import CliRunner
from connectome_manipulator.cli import app

from bluepysnap.circuit import Circuit
from libsonata import EdgeStorage, NodeStorage

import pytest

from connectome_manipulator import utils

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
WIRING_CONFIG_TPL = DATA_DIR / "wiring_config.json.tpl"


@dataclass(frozen=True)
class Helper:
    config_file: Path
    output_dir: Path

    @property
    def config(self):
        """Return manipulation config."""
        return utils.load_json(self.config_file)

    @property
    def input_circuit_config(self):
        """Return the input circuit config."""
        return Circuit(Path(self.config["circuit_path"], self.config["circuit_config"])).config

    def run(self, arglist):
        result = CliRunner().invoke(
            cli=app,
            args=arglist,
            catch_exceptions=False,
        )
        print(result.output)
        assert result.exit_code == 0, f"Application exited with non-zero code.\n{result.output}"

    @property
    def output_circuit_config(self):
        """Return the output circuit config."""
        return Circuit(self.output_dir / "circuit_config.json").config

    @property
    def output_edges_file(self):
        """Return path to the output edges file."""
        return self.output_circuit_config["networks"]["edges"][0]["edges_file"]


def _manipulation_config_path(output_dir, config_name):
    """Create and return path to manipulation config."""
    wiring_config = {}
    with open(WIRING_CONFIG_TPL, "r") as config_file:
        wiring_config = json.load(config_file)
    wiring_config["circuit_path"] = str(DATA_DIR)
    wiring_config["circuit_config"] = str(config_name)
    wiring_config["output_path"] = str(output_dir)
    wiring_config_path = output_dir / "wiring_config.json"
    with open(wiring_config_path, "w") as outf:
        json.dump(wiring_config, outf, indent=2)
    return wiring_config_path


@pytest.fixture(scope="module")
def local_connectome__empty_edges(tmpdir_factory):
    """Run the cli command for building the local connectome."""

    output_dir = Path(tmpdir_factory.mktemp("local-connectome-empty-edges"))

    config_file = _manipulation_config_path(output_dir, "circuit_100.json")

    obj = Helper(config_file, output_dir)

    obj.run(
        [
            "manipulate-connectome",
            str(obj.config_file),
            "--output-dir",
            str(obj.output_dir),
            "--overwrite-edges",
            "--convert-to-sonata",
        ],
    )
    return obj


def _check_number_of_edges(edges_file, expected_number):
    edge_storage = EdgeStorage(edges_file)
    pop_name = list(edge_storage.population_names)[0]
    edges = edge_storage.open_population(pop_name)
    assert len(edges) == expected_number, "Wrong number of connections found in output"


def test_number_of_edges__empty_edges(local_connectome__empty_edges):
    """Test the number of edges of the generated edge population."""
    _check_number_of_edges(
        local_connectome__empty_edges.output_edges_file,
        expected_number=134,
    )


def _check_input_output_configs(input_circuit_config, output_circuit_config):
    out_nodes = output_circuit_config["networks"]["nodes"][0]
    out_nodes_file = out_nodes["nodes_file"]

    inp_nodes = input_circuit_config["networks"]["nodes"][0]
    inp_nodes_file = inp_nodes["nodes_file"]

    assert out_nodes_file == inp_nodes_file
    assert Path(out_nodes_file).exists()

    out_edges = output_circuit_config["networks"]["edges"][0]
    out_edges_file = out_edges["edges_file"]

    assert Path(out_edges_file).exists()


def test_generated_circuit_config__empty_edges(local_connectome__empty_edges):
    """Test the validity of the output circuit config."""
    _check_input_output_configs(
        local_connectome__empty_edges.input_circuit_config,
        local_connectome__empty_edges.output_circuit_config,
    )


@pytest.fixture(scope="module")
def local_connectome__existing_edges(tmpdir_factory):
    """Run the cli command for building the local connectome."""

    output_dir = Path(tmpdir_factory.mktemp("local-connectome-existing-edges"))

    config_file = _manipulation_config_path(output_dir, "circuit_100_167.json")

    obj = Helper(config_file, output_dir)
    obj.run(
        [
            "manipulate-connectome",
            str(obj.config_file),
            "--output-dir",
            str(obj.output_dir),
            "--overwrite-edges",
            "--convert-to-sonata",
        ],
    )
    return obj


def test_generated_circuit_config__existing_edges(local_connectome__existing_edges):
    """Test the validity of the output circuit config."""
    _check_input_output_configs(
        local_connectome__existing_edges.input_circuit_config,
        local_connectome__existing_edges.output_circuit_config,
    )


def test_number_of_edges__existing_edges(local_connectome__existing_edges):
    """Test the number of edges of the generated edge population."""
    _check_number_of_edges(
        local_connectome__existing_edges.output_edges_file,
        expected_number=301,
    )
