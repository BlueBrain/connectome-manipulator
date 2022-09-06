import os
import json

import pytest
from mock import patch, Mock

from bluepysnap import Circuit
from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation

from utils import create_fake_module, setup_tempdir, TEST_DATA_DIR

import connectome_manipulator.connectome_manipulation.connectome_manipulation as test_module


def test_load_circuit():
    # Check error handling if no edge population exists
    with patch(f'connectome_manipulator.connectome_manipulation.connectome_manipulation.Circuit'
               ) as patched:
        patched.return_value = Mock(edges=Mock(population_names=[]))

        expected_error = 'No edge population found!'
        with pytest.raises(AssertionError, match=expected_error):
            test_module.load_circuit('fake_config')

    # Check error handling if more than one non-default populations exist and no population name is specified
    with patch(f'connectome_manipulator.connectome_manipulation.connectome_manipulation.Circuit'
               ) as patched:
        patched.return_value = Mock(edges=Mock(population_names=['name1', 'name2']))

        expected_error = 'Population "default" not found in edges file!'
        with pytest.raises(AssertionError, match=expected_error):
            test_module.load_circuit('fake_config')

    config_path = os.path.join(TEST_DATA_DIR, 'circuit_sonata.json')
    n_split = 2
    res = test_module.load_circuit(config_path, N_split=n_split)

    # Check that there are 2 instances of NodePopulation in first item
    assert len(res[0]) == 2
    assert all(isinstance(r, NodePopulation) for r in res[0])

    # Check that two paths are returned and they're correct
    assert len(res[1]) == 2
    assert all(r == os.path.join(TEST_DATA_DIR, 'nodes.h5') for r in res[1])

    # Check that number of splits is correct
    # Just a thought, but maybe N_split should be restricted with min(N_split, len(tgt_node_ids))
    assert len(res[2]) == n_split

    # Check that last edge population is returned
    assert isinstance(res[3], EdgePopulation)

    # Check that a correct edge path is returned
    assert res[4] == os.path.join(TEST_DATA_DIR, 'edges.h5')


def test_apply_manipulation():
    module_name = 'fake_apply'
    manip_config = {'manip': {'name': 'test',
                              'fcts': [{'source': module_name, 'kwargs': {}}]}}
    full_module_name = f'connectome_manipulator.connectome_manipulation.{module_name}'

    # Create fake module in which apply is an instance of Mock()
    fake_module = create_fake_module(full_module_name, 'from mock import Mock; apply=Mock()')

    # check that the apply function gets called
    test_module.apply_manipulation(None, None, manip_config, None)
    fake_module.apply.assert_called()

    with patch('importlib.import_module', Mock(return_value=object)):
        with pytest.raises(AssertionError, match=r'Manipulation module .* requires apply\(\) function'):
            test_module.apply_manipulation(None, None, manip_config, None)


def test_edges_to_parquet():
    edges = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json')).edges
    edges_table = edges.afferent_edges([0, 1], properties=sorted(edges.property_names))

    with setup_tempdir(__name__) as tempdir:
        outfile = os.path.join(tempdir, 'test.parquet')
        test_module.edges_to_parquet(edges_table, outfile)
        assert os.path.isfile(outfile)


def test_parquet_to_sonata():

    class FakeNode():
        name = ''

    infiles = ['dummy']
    nodes = [FakeNode(), FakeNode()]
    nodefiles = ['fake/path', 'another/one']

    class MockPopen():
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def communicate():
            return [''.encode()]

    class MockReadMetadataEmpty():
        def __init__(self, *args, **kwargs):
            pass
        num_rows = 0

    class MockReadMetadata():
        def __init__(self, *args, **kwargs):
            pass
        num_rows = 1

    with setup_tempdir(__name__) as tempdir:
        outfile = os.path.join(tempdir, 'test.sonata')

        with open(outfile, 'w') as fd:
            fd.write('')

        with patch('pyarrow.parquet.read_metadata', MockReadMetadataEmpty):
            # check that error is raised if empty input file(s)
            with pytest.raises(AssertionError, match='All .parquet files empty'):
                test_module.parquet_to_sonata(infiles, outfile, nodes, nodefiles) 

        with patch('pyarrow.parquet.read_metadata', MockReadMetadata):
            with patch('subprocess.Popen', MockPopen):

                # check that error is raised if file does not exist after
                with pytest.raises(AssertionError, match='SONATA'):
                    test_module.parquet_to_sonata(infiles, outfile, nodes, nodefiles) 

                # check that the file was removed
                assert not os.path.exists(outfile)


def test_create_new_file_from_template():
    with setup_tempdir(__name__) as tempdir:
        template = os.path.join(tempdir, 'template')
        new_file = os.path.join(tempdir, 'new_file')
        orig_text = ' # comment here\nnormal line here'
        replacements = {'here': 'somewhere',
                        'normal': 'cool'}

        with open(template, 'w') as fd:
            fd.write(orig_text)

        def read_new_file():
            with open(new_file) as fd:
                return fd.read()

        test_module.create_new_file_from_template(new_file, template, replacements, skip_comments=True)
        content = read_new_file()

        assert content.find('comment here') > -1
        assert content.find('cool line somewhere') > -1

        test_module.create_new_file_from_template(new_file, template, replacements, skip_comments=False)
        content = read_new_file()

        assert content.find('comment somewhere') > -1


def test_create_sonata_config():

    def json_read(path):
        with open(path, 'r') as fd:
            return json.load(fd)

    config_path = os.path.join(TEST_DATA_DIR, 'circuit_sonata.json')
    orig_edges = 'edges.h5'
    new_edges = 'edges-test.h5'

    with setup_tempdir(__name__) as tempdir:
        new_config_path = os.path.join(tempdir, 'circuit_sonata.json')

        test_module.create_sonata_config(new_config_path, new_edges, config_path, orig_edges)
        res = json_read(new_config_path)

        assert os.path.basename(res['networks']['edges'][0]['edges_file']) == new_edges

        test_module.create_sonata_config(new_config_path, new_edges, config_path, orig_edges, orig_base_dir=os.path.split(config_path)[0])
        res = json_read(new_config_path)
        config = json_read(config_path)

        # NOTE by herttuai on 06/10/2021:
        # To me it seems that the $BASE_DIR and $ORIG_BASE_DIR should be the other way around in manifest.
        # i.e., res['manifest']['$ORIG_BASE_DIR'] should equal to config['manifest']['$BASE_DIR']
        # Or maybe I misunderstood something
        # NOTE by chr-pok on 10/02/2022:
        # Yes indeed, there was some confusion. "rebase_dir" was meant to contain the original dir from which
        # to re-base the config. It has been renamed to "orig_base_dir" to avoid confusion.
        assert res['manifest']['$BASE_DIR'] == '.'
        if os.path.isabs(config['manifest']['$BASE_DIR']):
            assert res['manifest']['$ORIG_BASE_DIR'] == config['manifest']['$BASE_DIR']
        else:
            assert res['manifest']['$ORIG_BASE_DIR'] == os.path.normpath(os.path.join(os.path.split(config_path)[0], config['manifest']['$BASE_DIR']))
        assert os.path.dirname(res['networks']['nodes'][0]['nodes_file']) == '$ORIG_BASE_DIR'
        assert os.path.dirname(res['networks']['edges'][0]['edges_file']) == '$BASE_DIR'


def test_create_workflow_config():
    mapping = {v: f'${v}' for v in ['CIRCUIT_NAME',
                                    'CIRCUIT_DESCRIPTION',
                                    'CIRCUIT_TYPE',
                                    'CIRCUIT_CONFIG',
                                    'DATE',
                                    'FILE_NAME',
                                    ]}

    with setup_tempdir(__name__) as tempdir:
        template_file = os.path.join(tempdir, 'template.json')
        with open(template_file, 'w') as fd:
            json.dump(mapping, fd)

        circuit_path = os.path.join(tempdir, 'fake_circuit.json')
        blue_config = os.path.join(tempdir, 'fake_CircuitConfig')
        output_path = os.path.join(tempdir, 'fake_outpath')
        workflow_path = os.path.join(output_path, 'workflows')

        for manip_name in ['fake_manipulation', None]:
            test_module.create_workflow_config(circuit_path,
                                               blue_config,
                                               manip_name,
                                               output_path,
                                               template_file)

            assert os.path.isdir(workflow_path)

            dirlist = [os.path.join(workflow_path, p) for p in os.listdir(workflow_path)]
            assert len(dirlist) == 1

            with open(dirlist[0], 'r') as fd:
                res = json.load(fd)

            os.remove(dirlist[0])

            for k in mapping:
                assert res[k] != mapping[k]
