import os

import numpy as np
import pandas as pd

from bluepysnap import Circuit

import pytest
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
    props_model_file = os.path.join(TEST_DATA_DIR, f'model_config__ConnProps.json') # Deterministic connection properties model w/o variation
    props_model = model_types.AbstractModel.model_from_file(props_model_file)
    pct = 100.0
    syn_class = 'EXC'

    def check_delay(res, nodes, syn_class, delay_model):
        """ Check delays (from PRE neuron (soma) to POST synapse position) """
        for idx in res[np.isin(res['@source_node'], nodes[0].ids({'synapse_class': syn_class}))].index:
            delay_offset, delay_scale = delay_model.get_param_dict()['delay_mean_coefs']
            src_pos = nodes[0].positions(res.loc[idx]['@source_node']).to_numpy()
            syn_pos = res.loc[idx][['afferent_center_x', 'afferent_center_y', 'afferent_center_z']].to_numpy()
            dist = np.sqrt(np.sum((src_pos - syn_pos)**2))
            delay = delay_scale * dist + delay_offset
            assert np.isclose(res.loc[idx]['delay'], delay), 'ERROR: Delay mismatch!'

    def check_nsyn(ref, res):
        """ Check number of synapses per connection """
        nsyn_per_conn1 = np.unique(ref[['@source_node', '@target_node']], axis=0, return_counts=True)[1]
        nsyn_per_conn2 = np.unique(res[['@source_node', '@target_node']], axis=0, return_counts=True)[1]
        assert np.array_equal(np.sort(nsyn_per_conn1), np.sort(nsyn_per_conn2)), 'ERROR: Synapses per connection mismatch!'

    def check_indegree(ref, res, nodes, check_not_equal=False):
        """ Check indegree """
        indeg1 = [len(np.unique(ref['@source_node'][ref['@target_node'] == tid])) for tid in nodes[1].ids()]
        indeg2 = [len(np.unique(res['@source_node'][res['@target_node'] == tid])) for tid in nodes[1].ids()]
        if check_not_equal:
            assert not np.array_equal(indeg1, indeg2), 'ERROR: Indegree should be different!'
        else:
            assert np.array_equal(indeg1, indeg2), 'ERROR: Indegree mismatch!'

    def check_unchanged(ref, res, nodes, syn_class):
        """ Check that non-target connections unchanged """
        unch_tab1 = ref[~np.isin(ref['@source_node'], nodes[0].ids({'synapse_class': syn_class}))]
        unch_tab2 = res[~np.isin(res['@source_node'], nodes[0].ids({'synapse_class': syn_class}))]
        assert np.all([np.sum(np.all(unch_tab1.iloc[idx] == unch_tab2, 1)) == 1 for idx in range(unch_tab1.shape[0])]), f'ERROR: Non-{syn_class} connections changed!'

    def check_sampling(ref, res, col_sel):
        """ Check if synapse properties (incl. #syn/conn) sampled from existing values """
        nsyn_per_conn1 = np.unique(ref[['@source_node', '@target_node']], axis=0, return_counts=True)[1]
        nsyn_per_conn2 = np.unique(res[['@source_node', '@target_node']], axis=0, return_counts=True)[1]
        assert np.all(np.isin(nsyn_per_conn2, nsyn_per_conn1)), 'ERROR: Synapses per connection sampling error!' # Check duplicate_sample (#syn/conn)
        assert np.all([np.all(np.isin(np.unique(res[col]), np.unique(ref[col]))) for col in col_sel]), 'ERROR: Synapse properties sampling error!' # Check duplicate_sample (w/o #syn/conn)

    # Case 1: Rewire connectivity with conn. prob. p=0.0 (no connectivity)
    prob_model_file = os.path.join(TEST_DATA_DIR, 'model_config__ConnProb0p0.json')

    ## (a) Keeping indegree => NOT POSSIBLE with p=0.0
    with pytest.raises(AssertionError, match='Keeping indegree not possible since connection probability zero!'):
        res = test_module.apply(edges_table.copy(), nodes, aux_dict, syn_class=syn_class, prob_model_file=prob_model_file, delay_model_file=delay_model_file, sel_src=None, sel_dest=None, amount_pct=pct, keep_indegree=True, reuse_conns=True, gen_method=None, props_model_file=None, pos_map_file=None)

    ## (b) Not keeping indegree => All EXC connections should be removed
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, syn_class=syn_class, prob_model_file=prob_model_file, delay_model_file=delay_model_file, sel_src=None, sel_dest=None, amount_pct=pct, keep_indegree=False, reuse_conns=False, gen_method='duplicate_sample', props_model_file=None, pos_map_file=None)
    assert edges_table[~np.isin(edges_table['@source_node'], nodes[0].ids({'synapse_class': syn_class}))].equals(res), 'ERROR: Results table mismatch!'

    # Case 2: Rewire connectivity with conn. prob. p=1.0 (full connectivity, w/o autapses)
    prob_model_file = os.path.join(TEST_DATA_DIR, 'model_config__ConnProb1p0.json')

    ## (a) Keeping indegree & reusing connections
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, syn_class=syn_class, prob_model_file=prob_model_file, delay_model_file=delay_model_file, sel_src=None, sel_dest=None, amount_pct=pct, keep_indegree=True, reuse_conns=True, gen_method=None, props_model_file=None, pos_map_file=None)
    assert np.array_equal(edges_table.shape, res.shape), 'ERROR: Number of synapses mismatch!'
    
    col_sel = np.setdiff1d(edges_table.columns, ['@source_node', '@target_node', 'delay'])
    assert edges_table[col_sel].equals(res[col_sel]), 'ERROR: Synapse properties mismatch!'

    check_nsyn(edges_table, res) # Check reuse_conns option
    check_indegree(edges_table, res, nodes) # Check keep_indegree option
    check_unchanged(edges_table, res, nodes, syn_class) # Check that non-EXC connections unchanged

    ## (b) Keeping indegree & w/o reusing connections
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, syn_class=syn_class, prob_model_file=prob_model_file, delay_model_file=delay_model_file, sel_src=None, sel_dest=None, amount_pct=pct, keep_indegree=True, reuse_conns=False, gen_method='duplicate_sample', props_model_file=None, pos_map_file=None)

    check_indegree(edges_table, res, nodes) # Check keep_indegree option
    check_unchanged(edges_table, res, nodes, syn_class) # Check that non-EXC connections unchanged
    check_sampling(edges_table, res, col_sel) # Check duplicate_sample method
    check_delay(res, nodes, syn_class, delay_model) # Check synaptic delays

    ## (c) W/o keeping indegree & w/o reusing connections
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, syn_class=syn_class, prob_model_file=prob_model_file, delay_model_file=delay_model_file, sel_src=None, sel_dest=None, amount_pct=pct, keep_indegree=False, reuse_conns=False, gen_method='duplicate_sample', props_model_file=None, pos_map_file=None)

    assert False, 'TODO...'
    # TODO: Check all-to-all connectivity
#     src_sel = nodes[0].ids({'synapse_class': syn_class})
#     tgt_sel = nodes[1].ids()
#     adj_mat = np.full((len(src_sel), len(tgt_sel)), False)
#     for sidx, s in enumerate(src_sel):
#         for tidx, t in enumerate(tgt_sel):
#             if np.sum(np.logical_and(res['@source_node'] == s, res['@target_node'] == t)) > 0:
#                 adj_mat[sidx, tidx] = True

    check_indegree(edges_table, res, nodes, check_not_equal=True) # Check if keep_indegree changed
    check_unchanged(edges_table, res, nodes, syn_class) # Check that non-EXC connections unchanged
    check_sampling(edges_table, res, col_sel) # Check duplicate_sample method
    check_delay(res, nodes, syn_class, delay_model) # Check synaptic delays
