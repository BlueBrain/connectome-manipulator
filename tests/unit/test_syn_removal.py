# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import connectome_manipulator.connectome_manipulation.manipulation.syn_removal as test_module
import numpy as np
import os
import pytest

from bluepysnap import Circuit
from numpy.testing import assert_allclose, assert_approx_equal, assert_array_equal

from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator import log


@pytest.fixture
def manipulation():
    m = Manipulation.get("syn_removal")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    # Check that corrent number of synapses is removed
    total_syns = len(edges_table)
    for pct in range(0, 101, 25):
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(tgt_ids, amount_pct=pct)
        res = writer.to_pandas()
        assert len(res) == int(total_syns * (100 - pct) / 100)

    # Check that only given ids are considered
    src_id = 0
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, sel_src={*src_ids} - {src_id}, amount_pct=100)
    res = writer.to_pandas()
    assert np.all(res["@source_node"] == src_id)

    tgt_id = 9
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, sel_dest={*tgt_ids} - {tgt_id}, amount_pct=100)
    res = writer.to_pandas()
    assert np.all(res["@target_node"] == tgt_id)

    # Check when both sel_src and sel_dest are given
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(
        tgt_ids, sel_src={*src_ids} - {src_id}, sel_dest={*tgt_ids} - {tgt_id}, amount_pct=100
    )
    res = writer.to_pandas()
    assert np.all(np.logical_or(res["@source_node"] == src_id, res["@target_node"] == tgt_id))

    # Check that one synapse per connection is left when keep-cons=True
    total_conns = len(np.unique(edges_table[["@source_node", "@target_node"]], axis=0))
    writer = EdgeWriter(None, existing_edges=edges_table.copy())
    manipulation(nodes, writer).apply(tgt_ids, amount_pct=100, keep_conns=True, rescale_gsyn=True)
    res = writer.to_pandas()
    res_conn = np.unique(res[["@source_node", "@target_node"]], axis=0)
    assert len(res_conn) == total_conns
    assert len(res) == total_conns

    # Check that conductance per connection is maintained with rescale_gsyn=True
    assert_allclose(
        res.groupby(["@source_node", "@target_node"])["conductance"].sum(),
        edges_table.groupby(["@source_node", "@target_node"])["conductance"].sum(),
    )


def test_get_gsyn_sum_per_conn():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    # Test full edges table
    gsyn_table = test_module.SynapseRemoval._get_gsyn_sum_per_conn(edges_table, src_ids, tgt_ids)
    edge_groups = edges_table.groupby(["@source_node", "@target_node"])
    for sgid in src_ids:
        for tgid in tgt_ids:
            if (sgid, tgid) in edge_groups.groups:
                assert_approx_equal(
                    gsyn_table[sgid, tgid], edge_groups.get_group((sgid, tgid)).conductance.sum()
                )

    # Test subset
    src_sel = [3, 9]
    tgt_sel = [2, 4, 7]
    gsyn_table = test_module.SynapseRemoval._get_gsyn_sum_per_conn(edges_table, src_sel, tgt_sel)
    mask = np.logical_and(
        np.in1d(edges_table["@source_node"], src_sel),
        np.in1d(edges_table["@target_node"], tgt_sel),
    )
    assert np.sum(mask) > 0 and np.sum(mask) < len(mask), "Bad choice for testing!"
    edge_groups = edges_table.loc[mask].groupby(["@source_node", "@target_node"])
    for sgid in src_sel:
        for tgid in tgt_sel:
            if (sgid, tgid) in edge_groups.groups:
                assert_approx_equal(
                    gsyn_table[sgid, tgid], edge_groups.get_group((sgid, tgid)).conductance.sum()
                )


def test_rescale_gsyn_per_conn():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = circuit.edges[circuit.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table_orig = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    # Basically test rescaling the conductance of every connection by a factor of 2
    ratio = 2
    edges_table = edges_table_orig.copy()
    gsyn_table = test_module.SynapseRemoval._get_gsyn_sum_per_conn(edges_table, src_ids, tgt_ids)
    gsyn_table_manip = np.array(gsyn_table) * ratio

    test_module.SynapseRemoval._rescale_gsyn_per_conn(
        edges_table, src_ids, tgt_ids, gsyn_table, gsyn_table_manip
    )
    assert_array_equal(edges_table.conductance, edges_table_orig.conductance / ratio)

    # Now the same but only for a subset
    src_sel = [3, 9]
    tgt_sel = [2, 4, 7]
    edges_table = edges_table_orig.copy()
    gsyn_table = test_module.SynapseRemoval._get_gsyn_sum_per_conn(edges_table, src_sel, tgt_sel)
    gsyn_table_manip = np.array(gsyn_table) * ratio

    test_module.SynapseRemoval._rescale_gsyn_per_conn(
        edges_table, src_sel, tgt_sel, gsyn_table, gsyn_table_manip
    )

    # Check that only the connections defined by node_ids are changed
    mask = np.logical_and(
        np.in1d(edges_table["@source_node"], src_sel),
        np.in1d(edges_table["@target_node"], tgt_sel),
    )
    assert np.sum(mask) > 0 and np.sum(mask) < len(mask), "Bad choice for testing!"
    assert_array_equal(edges_table.conductance[mask], edges_table_orig.conductance[mask] / ratio)
    assert_array_equal(edges_table.conductance[~mask], edges_table_orig.conductance[~mask])
