import os

import numpy as np
import pandas as pd

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
import connectome_manipulator.connectome_manipulation.conn_wiring as test_module


def test_apply():
    c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    node_ids = [n.ids() for n in nodes]
    edges_table = edges.afferent_edges(node_ids[1], properties=edges.property_names)

    aux_dict = {'split_ids': node_ids[1]}
    prob_model_file = os.path.join(TEST_DATA_DIR, 'model_config__ConnProb0p0.json')
    nsynconn_model_file = os.path.join(TEST_DATA_DIR, 'model_config__NSynPerConn2.json')
    pct = 100.0
    # np.random.seed(0)
    res = test_module.apply(edges_table, nodes, aux_dict, amount_pct=pct, prob_model_file=prob_model_file, nsynconn_model_file=nsynconn_model_file)

    assert False, 'NYI'
    # TODO: Check if existing synapses unchanged
    #       Check if number of new synapses valid
    #       Check if new synapse property values valid
    #       Check if required properties are returned
    #       Test with different sets of input parameters:
    #         *Full/empty edges table: edges_table = edges_table.loc[[]]
    #         *p = 0.0/0.1/1.0
    #         *pct = 0/50/100
    #         *...
