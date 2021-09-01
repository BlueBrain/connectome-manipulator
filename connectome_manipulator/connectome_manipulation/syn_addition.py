'''TODO: improve description'''
# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

# TODO: Refactor into sub-functions for separate methods (to be accessed internally and externally by other manipulations)
# TODO: Add duplicate_sample method from conn_rewiring

import os.path
import pickle

import numpy as np
from bluepysnap.circuit import Circuit

from connectome_manipulator import log
from connectome_manipulator.connectome_manipulation.helper_functions import (
    get_gsyn_sum_per_conn, rescale_gsyn_per_conn)
from connectome_manipulator.model_building import model_building


def apply(edges_table, nodes, aux_dict, amount, sel_src=None, sel_dest=None, rescale_gsyn=False, method='duplicate', props_model_file=None, ext_edges_config_file=None):
    """Add certain amount of synapses to existing connections, optionally keeping sum of g_syns per connection constant."""
    # Input checks
    log.log_assert(isinstance(amount, dict) and 'type' in amount.keys() and 'value' in amount.keys(), 'Amount must be specified as dict with "type"/"value" entries!')
    log.log_assert(amount['type'] in ['pct', 'pct_per_conn', 'rnd_per_conn', 'minval_per_conn'], f'Synapse amount type "{amount["type"]}" not supported (must be "pct", "pct_per_conn", "rnd_per_conn", or "minval_per_conn")!')
    # amount: pct ... Overall increase by percentage of total number of existing synapses
    #         pct_per_conn ... Increase each connection by percentage of existing synapses
    #         rnd_per_conn ... Increase each connection by random number of synapses within given range
    #         minval_per_conn ... Increase until minimum target value of synapses per connection is reached
    log.log_assert(method in ['duplicate', 'derive', 'load'], f'Synapse addition method "{method}" not supported (must be "duplicate", "derive", or "load")!')
    # method: duplicate ... Duplicate existing synapses
    #         derive ... Derive from existing synapses (duplicate and randomize certain properties based on a model)
    #         load ... Load from external connectome, e.g. structural connectome
    #         [NYI] randomize ... Fully randomized parameterization, based on model-based distributions

    # Select connections to add synapses to
    gids_src = nodes[0].ids(sel_src)
    gids_dest = nodes[1].ids(sel_dest)
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest))
    num_syn = np.sum(syn_sel_idx)

    conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)

    # Determine number of synapses to be added
    if amount['type'] == 'pct': # Overall increase by percentage of total number of existing synapses between src and dest
        log.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn / 100).astype(int)
    elif amount['type'] == 'pct_per_conn': # Increase each connection by percentage of existing synapses
        log.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn_per_conn / 100).astype(int)
    elif amount['type'] == 'rnd_per_conn': # Increase each connection by random number of synapses within given range
        log.log_assert(not np.isscalar(amount['value']) and len(amount['value']) == 2, f'Amount value for type "{amount["type"]}" needs to be a range with two elements!')
        log.log_assert(amount['value'][0] >= 0 and amount['value'][1] >= amount['value'][0], f'Amount value range for type "{amount["type"]}" out of range!')
        num_add = np.random.randint(low=np.round(amount['value'][0]).astype(int), high=np.round(amount['value'][1]).astype(int) + 1, size=conns.shape[0])
    elif amount['type'] == 'minval_per_conn': # Increase until minimum target value of synapses per connection is reached
        log.log_assert(amount['value'] >= 0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.maximum(np.round(amount['value']).astype(int) - num_syn_per_conn, 0)
    else:
        log.log_assert(False, f'Amount type "{amount["type"]}" not supported!')

    log.info(f'Adding {np.sum(num_add)} synapses to {edges_table.shape[0]} total synapses (sel_src={sel_src}, sel_dest={sel_dest}, amount[type]={amount["type"]}, amount[value]={amount["value"]}, method={method}, rescale_gsyn={rescale_gsyn})')

    # Load synapse properties model (required for "derive" method)
    if method == 'derive':
        # >example: props_model_file = '../working_dir/model_building/circuit-build-S1_v1/model/ConnPropsPerMType.pickle'
        log.log_assert(props_model_file is not None, f'Synaptic properties model file required for "{method}" method!')
        log.log_assert(os.path.exists(props_model_file), 'Synaptic properties model file not found!')
        log.info(f'Loading synaptic properties model from {props_model_file}')
        with open(props_model_file, 'rb') as f:
            props_model_dict = pickle.load(f)
        props_model = model_building.get_model(props_model_dict['model'], props_model_dict['model_inputs'], props_model_dict['model_params'])

    # Load external connectome (required for "load" method)
    if method == 'load':
        # >example: ext_edges_config_file = '/gpfs/bbp.cscs.ch/data/scratch/proj83/home/pokorny/circuit-build-S1_v1/sonata/struct_circuit_config.json'
        log.log_assert(ext_edges_config_file is not None, f'External connectome edges file required for "{method}" method!')
        log.log_assert(os.path.exists(ext_edges_config_file), 'External connectome edges file not found!')
        log.info(f'Loading external connectome edges from {ext_edges_config_file}')

        # Load external circuit
        c_ext = Circuit(ext_edges_config_file)

        # Select external edge population [assuming exactly one edge population in given edges file]
        log.log_assert(len(c_ext.edges.population_names) == 1, 'Only a single edge population per file supported as external connectome!')
        edges_ext = c_ext.edges[c_ext.edges.population_names[0]]

        # Select corresponding external source/target nodes populations => Must be consistent with circuit nodes
        log.log_assert(edges_ext.source.name == nodes[0].name and edges_ext.target.name == nodes[1].name, f'External nodes populations inconsistent ({edges_ext.source.name}->{edges_ext.target.name} instead of {nodes[0].name}->{nodes[1].name})!')
        nodes_ext = [edges_ext.source, edges_ext.target]
        log.log_assert(np.all([np.array_equal(n.ids(), n_ext.ids()) for (n, n_ext) in zip(nodes, nodes_ext)]), 'External connectome not consistent with circuit nodes!')

        # Extract external edges table between selected nodes
        edges_table_ext = edges_ext.afferent_edges(aux_dict['split_ids'], properties=list(edges_table.columns))
        edges_table_ext = edges_table_ext[np.logical_and(np.isin(edges_table_ext['@source_node'], gids_src), np.isin(edges_table_ext['@target_node'], gids_dest))]
        num_syn_ext = edges_table_ext.shape[0]

        # Remove duplicate synapses that are already in circuit edges table
        edges_table_ext = edges_table_ext.append(edges_table.append(edges_table)).drop_duplicates(keep=False, ignore_index=True)
        log.info(f'Loaded external edges table with {edges_table_ext.shape[0]} synapses ({num_syn_ext} synapses initially; {num_syn_ext - edges_table_ext.shape[0]} duplicates dropped)')
        log.log_assert(edges_table_ext.shape[0] > 0, 'External synapse table is empty!')

        # Select only synapses from connections that are also part of the selected circuit edges table (to not introduce new connections)
        # (Get existing connections in the external connectome)
        conns_ext, syn_conn_idx_ext = np.unique(edges_table_ext[['@source_node', '@target_node']], axis=0, return_inverse=True)
        # (Filter connections, keeping only the ones that are also present in the selected circuit connectome)
        conns_set = {tuple(c) for c in conns}
        conn_sel_idx_ext = np.apply_along_axis(lambda c: tuple(c) in conns_set, 1, conns_ext)
        conn_sel_idx_ext = np.where(conn_sel_idx_ext)[0]
        conns_ext = conns_ext[conn_sel_idx_ext]
        # (Find corresponding external synapses, belonging to any of the existing circuit connections)
        syn_sel_idx_ext = np.isin(syn_conn_idx_ext, conn_sel_idx_ext)
        log.log_assert(np.sum(syn_sel_idx_ext) > 0, 'No external synapses corresponding to existing connections found!')
        # (Find mapping between circuit connections and external connections; -1 if connection does not exist is external connectome)
        if not np.isscalar(num_add): # Required only for connection-specific additions
            conn_map = np.full(conns.shape[0], -1)
            for cidx in range(conns.shape[0]):
                sel_idx = np.where(np.all(conns_ext == conns[cidx, :], 1))[0]
                if len(sel_idx) > 0:
                    conn_map[cidx] = conn_sel_idx_ext[sel_idx[0]]
                    # >example how to access all external synapses belonging to given connection c, i.e. conns[c, :] => edges_table_ext[syn_conn_idx_ext == conn_map[c]]

    # Add num_add synapses between src and dest nodes
    if np.sum(num_add) > 0:
        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) BEFORE adding synapses
            gsyn_table = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)

        if method in ('duplicate', 'derive', ): # Duplicate or derive from existing synapses
            if np.isscalar(num_add): # Overall number of synapses to add
                sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_add) # Random sampling from existing synapses with replacement
            else: # Number of synapses per connection to add [requires syn_conn_idx for mapping between connections and synapses]
                sel_dupl = np.full(np.sum(num_add), -1) # Empty selection
                dupl_idx = np.hstack((0, np.cumsum(num_add))) # Index vector where to save selected indices
                for cidx, num in enumerate(num_add):
                    if num == 0: # Nothing to add for this connection
                        continue
                    conn_sel = np.zeros_like(syn_sel_idx, dtype=bool) # Empty selection mask
                    conn_sel[syn_sel_idx] = syn_conn_idx == cidx # Sub-select (mask) synapses belonging to given connection from all selected synapses
                    sel_dupl[dupl_idx[cidx]:dupl_idx[cidx + 1]] = np.random.choice(np.where(conn_sel)[0], num) # Random sampling from existing synapses per connection with replacement
            new_edges = edges_table.iloc[sel_dupl].copy()

            if method == 'derive': # Derive from existing synapses (duplicate and randomize certain properties based on a model)
                src_mtypes = nodes[0].get(new_edges['@source_node'].to_numpy(), properties='mtype')
                tgt_mtypes = nodes[1].get(new_edges['@target_node'].to_numpy(), properties='mtype')

                # Access props. statistics >example: props_model('u_syn', src_mtypes.iloc[0], tgt_mtypes.iloc[0], 'mean')
                model_props = sorted(props_model(None, None, None, None)) # List of properties in the model
                model_props = sorted(np.intersect1d(edges_table.keys(), model_props)) # Select model props which are part of the edges table

                # Randomize within connections
                new_conns, new_syn_conn_idx = np.unique(new_edges[['@source_node', '@target_node']], axis=0, return_inverse=True)
                for new_idx in range(new_conns.shape[0]):
                    # Estimate current property means per connection
                    conn_idx = np.where(np.all(conns == new_conns[new_idx], 1))[0][0]
                    prop_means = [np.mean(edges_table[syn_sel_idx].loc[syn_conn_idx == conn_idx, p]) for p in model_props]

                    # Get within-connection statistics from model
                    prop_stds = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'std-within') for p in model_props]
                    prop_mins = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'min') for p in model_props]
                    prop_maxs = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'max') for p in model_props]

                    # Randomize new property values per connection
                    # TODO: Take actual distributions (and data types) into account!
                    new_vals = np.random.randn(np.sum(new_syn_conn_idx == new_idx), len(model_props)) * prop_stds + prop_means
                    new_vals = np.maximum(new_vals, prop_mins)
                    new_vals = np.minimum(new_vals, prop_maxs)
                    new_edges.loc[new_syn_conn_idx == new_idx, model_props] = new_vals
                new_edges = new_edges.astype(edges_table.dtypes)

        elif method == 'load': # Load addditional synapses from external connectome
            if np.isscalar(num_add): # Overall number of external synapses to add
                num_syn_sel = np.sum(syn_sel_idx_ext)
                if num_syn_sel < num_add:
                    sel_ext = np.where(syn_sel_idx_ext)[0] # Add all available synapses
                    log.warning(f'{num_add - num_syn_sel} of {num_add} synapses could not be added, since not enough external synapses provided!')
                else:
                    sel_ext = np.random.choice(np.where(syn_sel_idx_ext)[0], num_add, replace=False) # Random sampling from external synapses without replacement
            else: # Number of external synapses per connection to add [requires syn_conn_idx_ext and conn_map for mapping between connections and external synapses]
                sel_ext = np.full(np.sum(num_add), -1) # Empty selection
                ext_idx = np.hstack((0, np.cumsum(num_add))) # Index vector where to save selected indices
                for cidx, num in enumerate(num_add):
                    if num == 0: # Nothing to add for this connection
                        continue
                    conn_sel_ext = syn_conn_idx_ext == conn_map[cidx] # Select all external synapses belonging to a given connection
                    num_syn_sel = np.sum(conn_sel_ext)
                    if num_syn_sel < num:
                        sel_ext[ext_idx[cidx]:(ext_idx[cidx] + num_syn_sel)] = np.where(conn_sel_ext)[0] # Add all available synapses
                    else:
                        sel_ext[ext_idx[cidx]:ext_idx[cidx + 1]] = np.random.choice(np.where(conn_sel_ext)[0], num, replace=False) # Random sampling from external synapses per connection without replacement
                if np.sum(sel_ext == -1) > 0:
                    log.warning(f'{np.sum(sel_ext == -1)} of {np.sum(num_add)} synapses could not be added, since not enough external synapses provided!')
                    sel_ext = sel_ext[sel_ext >= 0]
            new_edges = edges_table_ext.iloc[sel_ext].copy()
        else:
            log.log_assert(False, f'Method "{method}" not supported!')

        # Add new synapses to table, re-sort, and assign new index
        edges_table = edges_table.append(new_edges)
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required when merging files in block-based processing]

        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) AFTER adding synapses ...
            gsyn_table_manip = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)

            # ... and rescale g_syn so that the sum of g_syns per connections BEFORE and AFTER the manipulation is kept the same
            rescale_gsyn_per_conn(edges_table, gids_src, gids_dest, gsyn_table, gsyn_table_manip)
    else:
        log.warning('Nothing to add!')

    return edges_table
