import os

import numpy as np
import pandas as pd

from bluepysnap import Circuit

from utils import TEST_DATA_DIR
from connectome_manipulator.model_building import model_types
import connectome_manipulator.connectome_manipulation.conn_rewiring as test_module


def test_apply():
    c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    aux_dict = {'N_split': 1, 'split_ids': tgt_ids}
    delay_model_file = os.path.join(TEST_DATA_DIR, f'model_config__DistDepDelay.json') # Deterministic delay model w/o variation
    delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
    pct = 100.0

    prob_model_file = os.path.join(TEST_DATA_DIR, 'model_config__ConnProb1p0.json')
    res = test_module.apply(edges_table, nodes, aux_dict, sel_src={'synapse_class': 'EXC'}, syn_class='EXC', keep_indegree=True, amount_pct=pct, prob_model_file=prob_model_file, delay_model_file=delay_model_file)
