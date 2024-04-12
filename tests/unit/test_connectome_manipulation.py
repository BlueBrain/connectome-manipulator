# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import glob
import json
import os

from mock import patch, Mock
from pathlib import Path
import pytest

from bluepysnap import Circuit
from bluepysnap.nodes import NodePopulation, Nodes
from bluepysnap.edges import EdgePopulation

from utils import setup_tempdir, TEST_DATA_DIR

import connectome_manipulator.connectome_manipulation.connectome_manipulation as test_module
import connectome_manipulator.connectome_manipulation.converters as converters
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


def test_load_circuit():
    config_path = os.path.join(TEST_DATA_DIR, "circuit_sonata.json")

    # Check error handling if no edge population exists
    with patch(
        f"connectome_manipulator.connectome_manipulation.connectome_manipulation.Circuit.edges"
    ) as patched:
        patched.return_value = Mock(population_names=[])
        # patched.return_value = Mock(edges=Mock(population_names=[]))

        # UPDATE 10/11/2022: Circuits w/o edge population can be loaded and won't rise an error any more.
        #                    Returned edges-related values will be None.
        # expected_error = 'No edge population found!'
        # with pytest.raises(AssertionError, match=expected_error):
        #     test_module.load_circuit('fake_config')
        _, *res = test_module.load_circuit(config_path)

        # Check that there are 2 instances of NodePopulation in first item
        assert len(res[0]) == 2
        assert all(isinstance(r, NodePopulation) for r in res[0])

        # Check that two paths are returned and they're correct
        assert len(res[1]) == 2
        assert all(r == os.path.join(TEST_DATA_DIR, "nodes.h5") for r in res[1])

        # Check that edges-related values are None
        assert res[3] is res[4] is None

    # Check error handling if more than one non-default populations exist and no population name is specified
    with patch(
        f"connectome_manipulator.connectome_manipulation.connectome_manipulation.Circuit"
    ) as patched:
        patched.return_value = Mock(edges=Mock(population_names=["name1", "name2"]))

        expected_error = 'Population "default" not found in edges file!'
        with pytest.raises(AssertionError, match=expected_error):
            test_module.load_circuit("fake_config")

    res = test_module.load_circuit(config_path)

    # Check valid config
    config = res[0]
    nodes = config["networks"]["nodes"][0]
    nodes_file = nodes["nodes_file"]
    assert nodes_file == str(Path(TEST_DATA_DIR, "nodes.h5"))

    edges = config["networks"]["edges"][0]
    edges_file = edges["edges_file"]
    assert edges_file == str(Path(TEST_DATA_DIR, "edges.h5"))

    # Check that there are 2 instances of NodePopulation in first item
    assert len(res[1]) == 2
    assert all(isinstance(r, NodePopulation) for r in res[1])

    # Check that two paths are returned and they're correct
    assert len(res[2]) == 2
    assert all(r == os.path.join(TEST_DATA_DIR, "nodes.h5") for r in res[2])

    # Check that last edge population is returned
    assert isinstance(res[3], EdgePopulation)

    # Check that a correct edge path is returned
    assert res[4] == os.path.join(TEST_DATA_DIR, "edges.h5")


@patch("connectome_manipulator.connectome_manipulation.manipulation.base.get_enumeration_map")
def test_manipulation_registration(nodes):
    manip_mods = glob.glob("connectome_manipulator/connectome_manipulation/manipulation/*.py")
    manip_mods = [os.path.splitext(os.path.basename(m))[0] for m in manip_mods]
    print(manip_mods)
    for mod_name in manip_mods:
        if mod_name not in ["base", "__init__"]:
            m = Manipulation.get(mod_name)(nodes, None)


@patch("connectome_manipulator.connectome_manipulation.manipulation.base.get_enumeration_map")
def test_null_manipulation(nodes):
    module_name = "null_manipulation"
    manip_config = {"manip": {"name": "test", "fcts": [{"source": module_name, "kwargs": {}}]}}

    m = Manipulation.get(module_name)

    writer = converters.EdgeWriter(None)
    m(nodes, writer).apply(None, **manip_config)
    assert len(writer.to_pandas()) == 0


@patch("connectome_manipulator.connectome_manipulation.manipulation.base.get_enumeration_map")
def test_null_manipulation_with_edges(nodes):
    module_name = "null_manipulation"
    manip_config = {"manip": {"name": "test", "fcts": [{"source": module_name, "kwargs": {}}]}}

    m = Manipulation.get(module_name)

    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    tgt_ids = edges.target.ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    writer = converters.EdgeWriter(None, edges_table.copy())
    m(nodes, writer).apply(None, **manip_config)
    assert edges_table.equals(writer.to_pandas())


def test_edges_to_parquet(tmp_path):
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    edges_table = edges.afferent_edges([0, 1], properties=sorted(edges.property_names))

    outfile = tmp_path / "test.parquet"
    with converters.EdgeWriter(outfile, existing_edges=edges_table) as writer:
        pass
    assert outfile.is_file()


def test_parquet_to_sonata():
    class FakeNode:
        name = ""

    infiles = ["dummy"]
    inpath = "some/path"
    nodes = [FakeNode(), FakeNode()]
    nodefiles = ["fake/path", "another/one"]

    class MockGlob:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def glob(path):
            return infiles

    class MockPopen:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        @staticmethod
        def communicate():
            return ["".encode()]

    class MockReadMetadataEmpty:
        def __init__(self, *args, **kwargs):
            pass

        num_rows = 0

    class MockReadMetadata:
        def __init__(self, *args, **kwargs):
            pass

        num_rows = 1

    with setup_tempdir(__name__) as tempdir:

        def mock_create_parquet_metadata(parquet_path, nodes):
            metadata_file = os.path.join(tempdir, "_metadata")
            with open(metadata_file, "w") as fd:
                fd.write("")
            return metadata_file

        outfile = os.path.join(tempdir, "test.sonata")

        with open(outfile, "w") as fd:
            fd.write("")

        with patch("glob.glob", MockGlob.glob):
            with patch("pyarrow.parquet.read_metadata", MockReadMetadataEmpty):
                # check that error is raised if empty input file(s)
                with pytest.raises(AssertionError, match="All .parquet files must be non-empty"):
                    converters.parquet_to_sonata(inpath, outfile, nodes, nodefiles)
            with patch("pyarrow.parquet.read_metadata", MockReadMetadata):
                with patch("subprocess.Popen", MockPopen):
                    with patch(
                        "connectome_manipulator.connectome_manipulation.converters.create_parquet_metadata",
                        mock_create_parquet_metadata,
                    ):
                        # monkeypatch.setattr(test_module, 'create_parquet_metadata',
                        #                    mock_create_parquet_metadata)

                        # check that error is raised if file does not exist after
                        with pytest.raises(AssertionError, match="SONATA"):
                            converters.parquet_to_sonata(inpath, outfile, nodes, nodefiles)

                        # check that the file was removed
                        assert not os.path.exists(outfile)


def test_create_new_file_from_template():
    with setup_tempdir(__name__) as tempdir:
        template = os.path.join(tempdir, "template")
        new_file = os.path.join(tempdir, "new_file")
        orig_text = " # comment here\nnormal line here"
        replacements = {"here": "somewhere", "normal": "cool"}

        with open(template, "w") as fd:
            fd.write(orig_text)

        def read_new_file():
            with open(new_file) as fd:
                return fd.read()

        test_module.create_new_file_from_template(
            new_file, template, replacements, skip_comments=True
        )
        content = read_new_file()

        assert content.find("comment here") > -1
        assert content.find("cool line somewhere") > -1

        test_module.create_new_file_from_template(
            new_file, template, replacements, skip_comments=False
        )
        content = read_new_file()

        assert content.find("comment somewhere") > -1


def test_create_sonata_config():
    def json_read(path):
        with open(path, "r") as fd:
            return json.load(fd)

    config_path = os.path.join(TEST_DATA_DIR, "circuit_sonata.json")
    config = Circuit(config_path).config

    orig_edges = "edges.h5"
    new_edges_name = "edges-test.h5"
    population_name = "my_edges"

    with setup_tempdir(__name__) as tempdir:
        new_config_path = os.path.join(tempdir, "circuit_sonata.json")
        new_edges_path = os.path.join(tempdir, new_edges_name)

        test_module.create_sonata_config(config, new_config_path, new_edges_path, population_name)
        res = json_read(new_config_path)

        assert os.path.basename(res["networks"]["edges"][0]["edges_file"]) == new_edges_name

        assert res["manifest"] == {"$BASE_DIR": "."}
        assert res["networks"]["nodes"][0]["nodes_file"] == os.path.join(TEST_DATA_DIR, "nodes.h5")
        assert res["networks"]["edges"][0]["edges_file"] == "$BASE_DIR/edges-test.h5"


def test_create_workflow_config():
    mapping = {
        v: f"${v}"
        for v in [
            "CIRCUIT_NAME",
            "CIRCUIT_DESCRIPTION",
            "CIRCUIT_TYPE",
            "CIRCUIT_CONFIG",
            "DATE",
            "FILE_NAME",
        ]
    }

    with setup_tempdir(__name__) as tempdir:
        template_file = os.path.join(tempdir, "template.json")
        with open(template_file, "w") as fd:
            json.dump(mapping, fd)

        circuit_path = os.path.join(tempdir, "fake_circuit.json")
        blue_config = os.path.join(tempdir, "fake_CircuitConfig")
        output_path = os.path.join(tempdir, "fake_outpath")
        workflow_path = os.path.join(output_path, "workflows")

        for manip_name in ["fake_manipulation", None]:
            test_module.create_workflow_config(
                circuit_path, blue_config, manip_name, output_path, template_file
            )

            assert os.path.isdir(workflow_path)

            dirlist = [os.path.join(workflow_path, p) for p in os.listdir(workflow_path)]
            assert len(dirlist) == 1

            with open(dirlist[0], "r") as fd:
                res = json.load(fd)

            os.remove(dirlist[0])

            for k in mapping:
                assert res[k] != mapping[k]
