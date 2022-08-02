import os

import numpy as np
import pandas as pd

from bluepysnap import Circuit

import pytest
import re
from utils import TEST_DATA_DIR
from connectome_manipulator.model_building import model_types
import connectome_manipulator.connectome_manipulation.syn_prop_alteration as test_module


def test_apply():
    c = Circuit(os.path.join(TEST_DATA_DIR, 'circuit_sonata.json'))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    aux_dict = {'N_split': 1, 'split_ids': tgt_ids}

    # Case 1: Check prop_sel
    pct = 100.0
    new_value = {'mode': 'setval', 'value': -1.0}
    for prop_sel in ['u_syn', 'n_rrp_vesicles', 'delay', 'afferent_section_pos', '@source_node']:
        props_nonsel = np.setdiff1d(edges_table.columns, prop_sel)
        res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
        assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
        assert np.all(res[prop_sel] == new_value['value']), 'ERROR: Selected property values not modified correctly!'
    
    # Case 2: Check pct
    prop_sel = 'u_syn'
    props_nonsel = np.setdiff1d(edges_table.columns, prop_sel)
    new_value = {'mode': 'setval', 'value': -1.0}
    for pct in np.linspace(0, 100, 6):
        res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
        assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
        assert 100.0 * np.sum(res[prop_sel] == new_value['value']) / res.shape[0] == pct, 'ERROR: Wrong percentage of synapses changed to target value!'
        assert 100.0 * np.sum(res[prop_sel] == edges_table[prop_sel]) / res.shape[0] == 100.0 - pct, 'ERROR: Wrong percentage of synapses unchanged!'

    # Case 3: Check sel_src/sel_dest
    pct = 100.0
    for src_class in ['EXC', 'INH']:
        for tgt_class in ['EXC', 'INH']:
            sel_src = {'synapse_class': src_class}
            sel_dest = {'synapse_class': tgt_class}
            res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=sel_src, sel_dest=sel_dest, syn_filter=None, amount_pct=pct)
            assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
            sel_idx = np.logical_and(np.isin(edges_table['@source_node'], nodes[0].ids(sel_src)), np.isin(edges_table['@target_node'], nodes[1].ids(sel_dest)))
            assert np.all(res[prop_sel][sel_idx] == new_value['value']), 'ERROR: Selected property values not modified correctly!'
            assert np.all(res[prop_sel][~sel_idx] == edges_table[prop_sel][~sel_idx]), 'ERROR: Non-selected property values changed!'

    # Case 4: Check syn_filter
    filt_prop = 'n_rrp_vesicles'
    filt_val = [2, 3]
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter={filt_prop: filt_val}, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    sel_idx = np.isin(edges_table[filt_prop], filt_val)
    assert np.all(res[prop_sel][sel_idx] == new_value['value']), 'ERROR: Selected property values not modified correctly!'
    assert np.all(res[prop_sel][~sel_idx] == edges_table[prop_sel][~sel_idx]), 'ERROR: Non-selected property values changed!'

    # Case 5: Check modes
    ## (a) Constant value
    new_value = {'mode': 'setval', 'value': -1.0}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == new_value['value']), 'ERROR: Selected property values not modified correctly!'

    ## (b) Scaling factor
    new_value = {'mode': 'scale', 'factor': -1.0}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == edges_table[prop_sel] * new_value['factor']), 'ERROR: Selected property values not modified correctly!'

    ## (c) Shuffling across synapses
    new_value = {'mode': 'shuffle'}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(np.isin(res[prop_sel], edges_table[prop_sel])) and not res[prop_sel].equals(edges_table[prop_sel]), 'ERROR: Selected property values not modified correctly!'

    ## (d) Random value (constant)
    new_value = {'mode': 'randval', 'rng': 'normal', 'kwargs': {'loc': -1.0, 'scale': 0.0}}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == new_value['kwargs']['loc']), 'ERROR: Selected property values not modified correctly!'

    ## (e) Random scaling factor (constant)
    new_value = {'mode': 'randscale', 'rng': 'normal', 'kwargs': {'loc': -1.0, 'scale': 0.0}}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == edges_table[prop_sel] * new_value['kwargs']['loc']), 'ERROR: Selected property values not modified correctly!'

    ## (f) Random additive value (constant)
    new_value = {'mode': 'randadd', 'rng': 'normal', 'kwargs': {'loc': -1.0, 'scale': 0.0}}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == edges_table[prop_sel] + new_value['kwargs']['loc']), 'ERROR: Selected property values not modified correctly!'

    # Case 5: Check range
    ## (a) Lower bound
    new_value = {'mode': 'scale', 'factor': -1.0, 'range': [0.0, 1.0]}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == new_value['range'][0]), 'ERROR: Selected property values not modified correctly!'

    ## (b) Upper bound
    new_value = {'mode': 'scale', 'factor': 100.0, 'range': [0.0, 1.0]}
    res = test_module.apply(edges_table.copy(), nodes, aux_dict, prop_sel, new_value, sel_src=None, sel_dest=None, syn_filter=None, amount_pct=pct)
    assert edges_table[props_nonsel].equals(res[props_nonsel]), 'ERROR: Non-selected property values changed!'
    assert np.all(res[prop_sel] == new_value['range'][1]), 'ERROR: Selected property values not modified correctly!'
