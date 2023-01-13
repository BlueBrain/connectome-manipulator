"""
Manipulation name: conn_rewiring
Description: Rewiring of existing connectome, based on a given model of connection probability (which can
             optionally be scaled by a global probability scaling factor p_scale; p_scale=1.0 by default)
             Optionally, number of ingoing connections (and #synapses/connection) can be kept the same
             by re-using existing connections/synapses. Otherwise, existing synapses may be deleted or new ones
             created, based on the selected generation method ("duplicate_sample" and "duplicate_randomize"
             supported so far!).

             Connection/synapse generation methods (gen_method):
                 "duplicate_sample" ... Duplicate existing synapse position & sample (non-morphology-related; w/o delay) property values independently from existing synapses
                 "duplicate_randomize" ... Duplicate existing synapse position & randomize (non-morphology-related; w/o delay) property values based on pathway-specific model distributions

             Estimation run (optional): By setting estimation_run=True (False by default), an early stopping
                 criterion is applied to just estimate the resulting number of connections (on average, i.e.,
                 independent on random seed) based on the given connection probability model (and scaling).
                 No actual connectome will be generated and such a run will not produce any output file.

             NOTE: Input edges_table assumed to be sorted by @target_node.
                   Output edges_table will again be sorted by @target_node (But not by [@target_node, @source_node]!!).
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types, conn_prob
from connectome_manipulator.access_functions import get_node_ids


def apply(edges_table, nodes, aux_dict, syn_class, prob_model_file, delay_model_file, sel_src=None, sel_dest=None, pos_map_file=None, keep_indegree=True, reuse_conns=True, gen_method=None, amount_pct=100.0, props_model_file=None, estimation_run=False, p_scale=1.0):
    """Rewiring (interchange) of connections between pairs of neurons based on given conn. prob. model (re-using ingoing connections and optionally, creating/deleting synapses)."""
    log.log_assert(np.all(np.diff(edges_table['@target_node']) >= 0), 'Edges table must be ordered by @target_node!')
    log.log_assert(syn_class in ['EXC', 'INH'], f'Synapse class "{syn_class}" not supported (must be "EXC" or "INH")!')
    log.log_assert(0.0 <= amount_pct <= 100.0, '"amount_pct" out of range!')
    log.log_assert(p_scale >= 0.0, '"p_scale" cannot be negative!')

    if estimation_run:
        log.log_assert(keep_indegree == False, 'Connectivity estimation not supported with "keep_indegree" option!')
        log.info('*** Estimation run enabled ***')

    if keep_indegree and reuse_conns:
        log.log_assert(gen_method is None, f'No generation method required for "keep_indegree" and "reuse_conns" options!')
    else:
        log.log_assert(gen_method in ['duplicate_sample', 'duplicate_randomize'], 'Valid generation method required (must be "duplicate_sample" or "duplicate_randomize")!')

    if gen_method == 'duplicate_sample' and aux_dict['N_split'] > 1:
        log.warning(f'"{gen_method}" method samples only from synapses within same data split! Reduce number of splits to 1 to sample from all synapses!')

    # Load connection probability model
    log.log_assert(os.path.exists(prob_model_file), 'Conn. prob. model file not found!')
    log.info(f'Loading conn. prob. model from {prob_model_file}')
    p_model = model_types.AbstractModel.model_from_file(prob_model_file)
    log.log_assert(p_model.input_names == ['src_pos', 'tgt_pos'], 'Conn. prob. model must have "src_pos" and "tgt_pos" as inputs!')
    log.info(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')
    if p_scale != 1.0:
        log.info(f'Using probability scaling factor p_scale={p_scale}')

    # Load delay model
    log.log_assert(os.path.exists(delay_model_file), 'Delay model file not found!')
    log.info(f'Loading delay model from {delay_model_file}')
    delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
    log.info(f'Loaded delay model of type "{delay_model.__class__.__name__}"')

    # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
    _, pos_acc = conn_prob.load_pos_mapping_model(pos_map_file)    
    if pos_acc is None:
        log.info('No position mapping model provided')

    # Load connection/synapse properties model [required for "duplicate_randomize" method]
    if gen_method == 'duplicate_randomize':
        log.log_assert(props_model_file is not None, f'Properties model required for generation method "{gen_method}"!')
        log.log_assert(os.path.exists(props_model_file), 'Properties model file not found!')
        log.info(f'Loading properties model from {props_model_file}')
        props_model = model_types.AbstractModel.model_from_file(props_model_file)
        log.info(f'Loaded properties model of type "{props_model.__class__.__name__}"')
    else:
        log.log_assert(props_model_file is None, f'Properties model incompatible with generation method "{gen_method}"!')
        props_model = None

    # Determine source/target nodes for rewiring
    stats_dict = {} # Keep track of statistics
    stats_dict['num_syn_removed'] = []
    stats_dict['num_conn_removed'] = []
    stats_dict['num_syn_rewired'] = []
    stats_dict['num_conn_rewired'] = []
    stats_dict['num_syn_added'] = []
    stats_dict['num_conn_added'] = []
    src_node_ids = get_node_ids(nodes[0], sel_src)
    src_class = nodes[0].get(src_node_ids, properties='synapse_class')
    src_node_ids = src_class[src_class == syn_class].index.to_numpy() # Select only source nodes with given synapse class (EXC/INH)
    log.log_assert(len(src_node_ids) > 0, f'No {syn_class} source nodes found!')
    syn_sel_idx_src = np.isin(edges_table['@source_node'], src_node_ids)
    log.log_assert(np.all(edges_table.loc[syn_sel_idx_src, 'syn_type_id'] >= 100) if syn_class == 'EXC'
                   else np.all(edges_table.loc[syn_sel_idx_src, 'syn_type_id'] < 100), 'Synapse class error!')
    src_pos = conn_prob.get_neuron_positions(nodes[0].positions if pos_acc is None else pos_acc, [src_node_ids])[0] # Get neuron positions (incl. position mapping, if provided)

    tgt_node_ids = get_node_ids(nodes[1], sel_dest)
    num_tgt_total = len(tgt_node_ids)
    tgt_node_ids = np.intersect1d(tgt_node_ids, aux_dict['split_ids']) # Only select target nodes that are actually in current split of edges_table
    num_tgt = np.round(amount_pct * len(tgt_node_ids) / 100).astype(int)
    tgt_sel = np.random.permutation([True] * num_tgt + [False] * (len(tgt_node_ids) - num_tgt))
    if len(tgt_node_ids) > 0:
        tgt_node_ids = tgt_node_ids[tgt_sel] # Select subset of neurons (keeping order)
    if np.sum(tgt_sel) == 0: # Nothing to rewire
        log.info('No target nodes selected, nothing to rewire')
        log.data(f'RewiringIndices_{aux_dict["i_split"] + 1}_{aux_dict["N_split"]}',
                 i_split=aux_dict['i_split'], N_split=aux_dict['N_split'], split_ids=aux_dict['split_ids'], tgt_node_ids=tgt_node_ids, tgt_sel=tgt_sel)
        return edges_table

    log.info(f'Rewiring afferent {syn_class} connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest}, keep_indegree={keep_indegree}, gen_method={gen_method})')

    # Init/reset static variables (function attributes) related to duplicate_... methods which need only be initialized once [for better performance]
    duplicate_sample_synapses.reset = True
    duplicate_randomize_synapses.reset = True

    # Index of input connections (before rewiring) [for data logging]
    inp_conns, inp_syn_conn_idx, inp_syn_per_conn = np.unique(edges_table[['@target_node', '@source_node']], axis=0, return_inverse=True, return_counts=True)
    inp_conns = np.fliplr(inp_conns) # Restore ['@source_node', '@target_node'] order of elements
    stats_dict['input_syn_count'] = edges_table.shape[0]
    stats_dict['input_conn_count'] = len(inp_syn_per_conn)
    stats_dict['input_syn_per_conn'] = list(inp_syn_per_conn)

    # Run connection rewiring
    syn_del_idx = np.full(edges_table.shape[0], False) # Global synapse indices to keep track of all unused synapses to be deleted
    syn_rewire_idx = np.full(edges_table.shape[0], False) # Global synapse indices to keep track of all rewired synapses [for data logging]
    all_new_edges = edges_table.loc[[]].copy() # New edges table to collect all generated synapses
    stats_dict['source_nrn_count_all'] = len(src_node_ids) # All source neurons (corresponding to chosen sel_src and syn_class)
    stats_dict['target_nrn_count_all'] = len(tgt_node_ids) # All target neurons in current split (corresponding to chosen sel_dest)
    stats_dict['target_nrn_count_sel'] = num_tgt # Selected target neurons in current split (based on amount_pct)
    stats_dict['unable_to_rewire_nrn_count'] = 0 # (Neurons)
    stats_dict['input_conn_count_sel'] = [] # Number of input connections within src/tgt node selection
    stats_dict['output_conn_count_sel'] = [] # Number of output connections within src/tgt node selection (based on prob. model; for specific seed)
    stats_dict['output_conn_count_sel_avg'] = [] # Average number of output connections within src/tgt node selection (based on prob. model)
    progress_pct = np.round(100 * np.arange(len(tgt_node_ids)) / (len(tgt_node_ids) - 1)).astype(int)
    for tidx, tgt in enumerate(tgt_node_ids):
        if tidx == 0 or progress_pct[tidx - 1] != progress_pct[tidx]:
            print(f'{progress_pct[tidx]}%', end=' ' if progress_pct[tidx] < 100.0 else '\n') # Just for console, no logging

        syn_sel_idx_tgt = edges_table['@target_node'] == tgt
        syn_sel_idx = np.logical_and(syn_sel_idx_tgt, syn_sel_idx_src)
        num_sel = np.sum(syn_sel_idx)

        if (keep_indegree and num_sel == 0) or np.sum(syn_sel_idx_tgt) == 0:
            stats_dict['unable_to_rewire_nrn_count'] += 1 # (Neurons)
            continue # Nothing to rewire (no synapses on target node)

        # Determine conn. prob. of all source nodes to be connected with target node
        tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_acc is None else pos_acc, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
        p_src = p_model.apply(src_pos=src_pos, tgt_pos=tgt_pos).flatten() * p_scale
        p_src[np.isnan(p_src)] = 0.0 # Exclude invalid values
        p_src[src_node_ids == tgt] = 0.0 # Exclude autapses [ASSUMING node IDs are unique across src/tgt node populations!]

        # Currently existing sources for given target node
        src, src_syn_idx = np.unique(edges_table.loc[syn_sel_idx, '@source_node'], return_inverse=True)
        num_src = len(src)
        stats_dict['input_conn_count_sel'] = stats_dict['input_conn_count_sel'] + [num_src]

        # Sample new presynaptic neurons from list of source nodes according to conn. prob.
        if keep_indegree: # Keep the same number of ingoing connections
            log.log_assert(len(src_node_ids) >= num_src, f'Not enough source neurons for target neuron {tgt} available for rewiring!')
            log.log_assert(np.sum(p_src) > 0.0, f'Keeping indegree not possible since connection probability zero!')
            src_new = np.random.choice(src_node_ids, size=num_src, replace=False, p=p_src / np.sum(p_src)) # New source node IDs per connection
        else: # Number of ingoing connections NOT kept the same

            stats_dict['output_conn_count_sel_avg'] = stats_dict['output_conn_count_sel_avg'] + [np.round(np.mean(p_src) * len(p_src)).astype(int)]
            if estimation_run:
                continue

#             ##### TESTING: OPTIMIZING #CONNECTIONS [Repeat random generation N times and keep the one with #connestions closest to average] #####
#             num_opt = 10
#             num_conns_expected = np.round(np.mean(p_src) * len(p_src)).astype(int) # Expected number of connections (=target count)
#             new_conn_count = -np.inf
#             for n_opt in range(num_opt):
#                 src_new_sel_tmp = np.random.rand(len(src_node_ids)) < p_src
#                 if np.abs(np.sum(src_new_sel_tmp) - num_conns_expected) < np.abs(new_conn_count - num_conns_expected): # Keep closest value among all tries
#                     src_new_sel = src_new_sel_tmp
#                     new_conn_count = np.sum(src_new_sel)
#                 if new_conn_count == num_conns_expected:
#                     break # Optimum found
#             ##### ###### ##### ###### ##### ###### #####

            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel] # New source node IDs per connection
            stats_dict['output_conn_count_sel'] = stats_dict['output_conn_count_sel'] + [np.sum(src_new_sel)]

        num_new = len(src_new)

        # Re-use (up to) num_src existing connections (incl. #synapses/connection) for rewiring of (up to) num_new new connections (optional)
        if reuse_conns:
            num_src_to_reuse = num_src
            num_new_reused = num_new
        else:
            num_src_to_reuse = 0
            num_new_reused = 0

        if num_src > num_new_reused: # Delete unused connections/synapses
            syn_del_idx[syn_sel_idx] = src_syn_idx >= num_new_reused # Set global indices of connections to be deleted
            syn_sel_idx[syn_del_idx] = False # Remove to-be-deleted indices from selection
            stats_dict['num_syn_removed'] = stats_dict['num_syn_removed'] + [np.sum(src_syn_idx >= num_new_reused)] # (Synapses)
            stats_dict['num_conn_removed'] = stats_dict['num_conn_removed'] + [num_src - num_new_reused] # (Connections)
            src_syn_idx = src_syn_idx[src_syn_idx < num_new_reused]
        else:
            stats_dict['num_syn_removed'] = stats_dict['num_syn_removed'] + [0] # (Synapses)
            stats_dict['num_conn_removed'] = stats_dict['num_conn_removed'] + [0] # (Connections)

        if num_src_to_reuse < num_new: # Generate new synapses/connections, if needed
            num_gen_conn = num_new - num_src_to_reuse # Number of new connections to generate
            src_gen = src_new[-num_gen_conn:] # Split new sources into ones used for newly generated ...
            src_new = src_new[:num_src_to_reuse] # ... and existing connections

            if gen_method == 'duplicate_sample': # Duplicate existing synapse position & sample (non-morphology-related) property values independently from existing synapses
                new_edges = duplicate_sample_synapses(src_gen, tidx, edges_table, nodes, syn_sel_idx, syn_sel_idx_tgt, tgt_node_ids, syn_class, delay_model)
            elif gen_method == 'duplicate_randomize': # Duplicate existing synapse position & randomize (non-morphology-related) property values based on pathway-specific model distributions
                new_edges = duplicate_randomize_synapses(src_gen, edges_table, nodes, syn_sel_idx, syn_sel_idx_tgt, tgt, delay_model, props_model)
            else:
                log.log_assert(False, f'Generation method {gen_method} unknown!')

            # Add new_edges to global new edges table [ignoring duplicate indices]
            all_new_edges = all_new_edges.append(new_edges)

            stats_dict['num_syn_added'] = stats_dict['num_syn_added'] + [new_edges.shape[0]] # (Synapses)
            stats_dict['num_conn_added'] = stats_dict['num_conn_added'] + [len(src_gen)] # (Connections)
        else:
            stats_dict['num_syn_added'] = stats_dict['num_syn_added'] + [0] # (Synapses)
            stats_dict['num_conn_added'] = stats_dict['num_conn_added'] + [0] # (Connections)

        # Assign new source nodes = rewiring of existing connections
        syn_rewire_idx = np.logical_or(syn_rewire_idx, syn_sel_idx) # [for data logging]
        edges_table.loc[syn_sel_idx, '@source_node'] = src_new[src_syn_idx] # Source node IDs per connection expanded to synapses
        stats_dict['num_syn_rewired'] = stats_dict['num_syn_rewired'] + [len(src_syn_idx)] # (Synapses)
        stats_dict['num_conn_rewired'] = stats_dict['num_conn_rewired'] + [len(src_new)] # (Connections)

        # Assign new distance-dependent delays (in-place), based on (generative) delay model
        assign_delays_from_model(delay_model, nodes, edges_table, src_new, src_syn_idx, syn_sel_idx)

    # Estimate resulting number of connections for computing a global probability scaling factor [returns empty edges table!!]
    if estimation_run:
        stat_sel = ['input_syn_count', 'input_conn_count', 'input_syn_per_conn', 'source_nrn_count_all', 'target_nrn_count_all', 'target_nrn_count_sel', 'unable_to_rewire_nrn_count', 'input_conn_count_sel', 'output_conn_count_sel_avg']
        stat_str = [f'      {k}: COUNT {len(v)}, MEAN {np.mean(v):.2f}, MIN {np.min(v)}, MAX {np.max(v)}, SUM {np.sum(v)}' if isinstance(v, list) and len(v) > 0 else f'      {k}: {v}' for k, v in stats_dict.items() if k in stat_sel]
        log.info('CONNECTIVITY ESTIMATION:\n%s', '\n'.join(stat_str))
        log.data(f'EstimationStats_{aux_dict["i_split"] + 1}_{aux_dict["N_split"]}', **{k: v for k, v in stats_dict.items() if k in stat_sel})
        return edges_table.iloc[[]].copy()

    # Update statistics
    stats_dict['num_syn_unchanged'] = stats_dict['input_syn_count'] - np.sum(stats_dict['num_syn_removed']) - np.sum(stats_dict['num_syn_rewired'])

    # Delete unused synapses (if any)
    if np.any(syn_del_idx):
        edges_table = edges_table[~syn_del_idx].copy()
        log.info(f'Deleted {np.sum(syn_del_idx)} unused synapses')

    # Add new synapses to table, re-sort, and assign new index
    if all_new_edges.size > 0:
        syn_new_dupl_idx = np.array(all_new_edges.index) # Index of duplicated synapses [for data logging]
        min_idx = np.min(edges_table.index)
        max_idx = np.max(edges_table.index)
        all_new_edges.index = range(max_idx + 1, max_idx + 1 + all_new_edges.shape[0]) # Set index to new range, so as to keep track of new edges
        edges_table = edges_table.append(all_new_edges)
        edges_table.sort_values('@target_node', kind='mergesort', inplace=True) # Stable sorting, i.e., preserving order of input edges!!
        syn_new_idx = edges_table.index > max_idx # Global synapse indices to keep track of all new synapses [for data logging]
        syn_new_dupl_idx = syn_new_dupl_idx[edges_table.index[syn_new_idx] - max_idx - 1] # Restore sorting, so that in same order as in merged & sorted edges table
        log.info(f'Generated {all_new_edges.shape[0]} new synapses')
    else: # No new synapses
        syn_new_dupl_idx = np.array([])
        syn_new_idx = np.full(edges_table.shape[0], False)

    # Reset index
    edges_table.reset_index(inplace=True, drop=True) # Reset index [No index offset required when merging files in block-based processing]

    ##### [TESTING] #####
    # Check if output indeed sorted
    log.log_assert(np.all(np.diff(edges_table['@target_node']) >= 0), 'ERROR: Output edges table not sorted by @target_node!')
    ##### ######### #####

    # Index of output connections (after rewiring) [for data logging]
    out_conns, out_syn_conn_idx, out_syn_per_conn = np.unique(edges_table[['@target_node', '@source_node']], axis=0, return_inverse=True, return_counts=True)
    out_conns = np.fliplr(out_conns) # Restore ['@source_node', '@target_node'] order of elements
    stats_dict['output_syn_count'] = edges_table.shape[0]
    stats_dict['output_conn_count'] = len(out_syn_per_conn)
    stats_dict['output_syn_per_conn'] = list(out_syn_per_conn)

    # Log statistics
    stat_str = [f'      {k}: COUNT {len(v)}, MEAN {np.mean(v):.2f}, MIN {np.min(v)}, MAX {np.max(v)}, SUM {np.sum(v)}' if isinstance(v, list) and len(v) > 0 else f'      {k}: {v}' for k, v in stats_dict.items()]
    log.info('STATISTICS:\n%s', '\n'.join(stat_str))
    log.log_assert(stats_dict['num_syn_unchanged'] == stats_dict['output_syn_count'] - np.sum(stats_dict['num_syn_added']) - np.sum(stats_dict['num_syn_rewired']), 'ERROR: Unchanged synapse count mismtach!') # Consistency check
    log.data(f'RewiringStats_{aux_dict["i_split"] + 1}_{aux_dict["N_split"]}', **stats_dict)

    # Write index data log [book-keeping for validation purposes]
    inp_syn_unch_idx = np.zeros_like(syn_del_idx) # Global synapse indices to keep track of all unchanged synapses [for data logging]
    inp_syn_unch_idx = np.logical_and(~syn_del_idx, ~syn_rewire_idx)
    out_syn_rew_idx = np.zeros_like(syn_new_idx) # Global output synapse indices to keep track of all rewired synapses [for data logging]
    out_syn_rew_idx[~syn_new_idx] = syn_rewire_idx[~syn_del_idx] # [ASSUME: Input edges table order preserved]
    out_syn_unch_idx = np.zeros_like(syn_new_idx) # Global output synapse indices to keep track of all unchanged synapses [for data logging]
    out_syn_unch_idx[~syn_new_idx] = inp_syn_unch_idx[~syn_del_idx] # [ASSUME: Input edges table order preserved]
    log.log_assert(np.sum(stats_dict['num_syn_rewired']) == np.sum(syn_rewire_idx), 'ERROR: Rewired (input) synapse count mismtach!')
    log.log_assert(np.sum(stats_dict['num_syn_rewired']) == np.sum(out_syn_rew_idx), 'ERROR: Rewired (output) synapse count mismtach!')
    log.log_assert(stats_dict['num_syn_unchanged'] == np.sum(inp_syn_unch_idx), 'ERROR: Unchanged (input) synapse count mismtach!')
    log.log_assert(stats_dict['num_syn_unchanged'] == np.sum(out_syn_unch_idx), 'ERROR: Unchanged (output) synapse count mismtach!')

    log.data(f'RewiringIndices_{aux_dict["i_split"] + 1}_{aux_dict["N_split"]}',
             inp_syn_del_idx=syn_del_idx, inp_syn_rew_idx=syn_rewire_idx, inp_syn_unch_idx=inp_syn_unch_idx,
             out_syn_new_idx=syn_new_idx, syn_new_dupl_idx=syn_new_dupl_idx, out_syn_rew_idx=out_syn_rew_idx, out_syn_unch_idx=out_syn_unch_idx,
             inp_conns=inp_conns, inp_syn_conn_idx=inp_syn_conn_idx, inp_syn_per_conn=inp_syn_per_conn,
             out_conns=out_conns, out_syn_conn_idx=out_syn_conn_idx, out_syn_per_conn=out_syn_per_conn,
             i_split=aux_dict['i_split'], N_split=aux_dict['N_split'], split_ids=aux_dict['split_ids'], src_node_ids=src_node_ids, tgt_node_ids=tgt_node_ids, tgt_sel=tgt_sel)
    # inp_syn_del_idx ... Binary index vector of deleted synapses w.r.t. input edges table (of current block)
    # inp_syn_rew_idx ... Binary index vector of rewired synapses w.r.t. input edges table (of current block)
    # inp_syn_unch_idx ... Binary index vector of unchanged synapses w.r.t. input edges table (of current block)
    # out_syn_new_idx ... Binary index vector of new synapses w.r.t. output edges table (of current block)
    # syn_new_dupl_idx ... Index vector of duplicated synapses (positions) w.r.t. input edges table (globally, i.e., across all blocks), corresponding to new synapses in out_syn_new_idx
    # out_syn_rew_idx ... Binary index vector of rewired synapses w.r.t. output edges table (of current block)
    # out_syn_unch_idx ... Binary index vector of unchanged synapses w.r.t. output edges table (of current block)
    # inp_conns ... Input connections (of current block)
    # inp_syn_conn_idx ... Index vector of input connections w.r.t. inp_conns (of current block)
    # inp_syn_per_conn: Number of synapses per connection w.r.t. inp_conns (of current block)
    # out_conns ... Input connections (of current block)
    # out_syn_conn_idx ... Index vector of input connections w.r.t. out_conns (of current block)
    # out_syn_per_conn ... Number of synapses per connection w.r.t. out_conns (of current block)
    # i_split ... Index of current block
    # N_split ... Total number of splits (blocks)
    # split_ids ... Neuron ids of current block
    # src_node_ids ... Selected source neuron ids
    # tgt_node_ids ... Selected target neuron ids within current block
    # tgt_sel ... Binary (random) target neuron selection index within current block, according to given amount_pct

    ##### [TESTING] #####
    # Overflow/value check
    log.log_assert(np.all(np.abs(edges_table.max()) < 1e9), 'Value overflow in edges table!')
    if 'n_rrp_vesicles' in edges_table.columns:
        log.log_assert(np.all(edges_table['n_rrp_vesicles'] >= 1), 'Value error in edges table (n_rrp_vesicles)!')
    ##### ######### #####

    return edges_table


def duplicate_synapses(syn_sel_idx, syn_sel_idx_tgt, syn_conn_idx, num_gen_syn, tgt):
    """ Duplicate num_gen_syn synapse positions on target neuron
        ['efferent_...' properties will no longer be consistent with actual source neuron's axon morphology!] """

#     # => Duplicated synapses may belong to same connection!!
#     num_sel = np.sum(syn_sel_idx)
#     if num_sel > 0: # Duplicate only synapses of syn_class type
#         sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_gen_syn) # Random sampling from existing synapses with replacement
#     else: # Include all synapses, if no synapses of syn_class type are available
#         sel_dupl = np.random.choice(np.where(syn_sel_idx_tgt)[0], num_gen_syn) # Random sampling from existing synapses with replacement

    # => Unique per connection, so that no duplicated synapses belong to same connection (if possible)!!
    unique_per_conn_warning = False
    num_sel = np.sum(syn_sel_idx)
    sel_dupl = []
    conns = np.unique(syn_conn_idx)
    for cidx in conns:
        dupl_count = np.sum(syn_conn_idx == cidx)
        if num_sel > 0: # Duplicate only synapses of syn_class type
            draw_from = np.where(syn_sel_idx)[0]
        else: # Include all synapses, if no synapses of syn_class type are available
            draw_from = np.where(syn_sel_idx_tgt)[0]
        if len(draw_from) >= dupl_count:
            sel_dupl.append(np.random.choice(draw_from, dupl_count, replace=False)) # Random sampling from existing synapses WITHOUT replacement, if possible
        else:
            sel_dupl.append(np.random.choice(draw_from, dupl_count, replace=True)) # Random sampling from existing synapses WITH replacement, otherwise
            unique_per_conn_warning = True
    sel_dupl = np.hstack(sel_dupl)
    log.log_assert(len(sel_dupl) == num_gen_syn, 'ERROR: Wrong number of duplicated synapse positions!')

    if unique_per_conn_warning:
        log.warning(f'Duplicated synapse position belonging to same connection (target neuron {tgt})! Unique synapse positions per connection not possible!')

    ##### [TESTING] #####
    # Check if indeed no duplicates per connection
    if not unique_per_conn_warning:
        for cidx in conns:
            log.log_assert(np.all(np.unique(sel_dupl[syn_conn_idx == cidx], return_counts=True)[1] == 1), f'ERROR: Duplicated synapse positions within connection (target neuron {tgt})!')
    ##### ######### #####

    return sel_dupl


def duplicate_sample_synapses(src_gen, tidx, edges_table, nodes, syn_sel_idx, syn_sel_idx_tgt, tgt_node_ids, syn_class, delay_model):
    """Method to generate new synapses from source neurons <src_gen> to target neuron <tgt_node_ids[tidx]>, by duplicating existing
       synapse position & sampling (non-morphology-related) property values independently from existing synapses."""
    # Init static variables (function attributes) related to this method which need only be initialized once [for better performance]
    if duplicate_sample_synapses.reset:
        duplicate_sample_synapses.per_mtype_dict = {} # Dict to keep computed values per target m-type (instead of re-computing them for each target neuron)
        duplicate_sample_synapses.props_sel = list(filter(lambda x: not np.any([excl in x for excl in ['_node', '_x', '_y', '_z', '_section', '_segment', '_length', 'delay']]), edges_table.columns)) # Non-morphology-related property selection (to be sampled/randomized)
        if syn_class == 'EXC':
            duplicate_sample_synapses.syn_sel_idx_type = edges_table['syn_type_id'] >= 100
        elif syn_class == 'INH':
            duplicate_sample_synapses.syn_sel_idx_type = edges_table['syn_type_id'] < 100
        else:
            log.log_assert(False, f'Synapse class {syn_class} not supported!')
        duplicate_sample_synapses.reset = False

    # Sample #synapses/connection from other existing synapses targetting neurons of the same mtype (or layer) as tgt (incl. tgt)
    tgt = tgt_node_ids[tidx]
    tgt_layers = nodes[1].get(tgt_node_ids, properties='layer').to_numpy()
    tgt_mtypes = nodes[1].get(tgt_node_ids, properties='mtype').to_numpy()
    tgt_mtype = tgt_mtypes[tidx]
    num_gen_conn = len(src_gen)
    if tgt_mtype in duplicate_sample_synapses.per_mtype_dict.keys(): # Load from dict, if already exists [optimized for speed]
        syn_sel_idx_mtype = duplicate_sample_synapses.per_mtype_dict[tgt_mtype]['syn_sel_idx_mtype']
        num_syn_per_conn = duplicate_sample_synapses.per_mtype_dict[tgt_mtype]['num_syn_per_conn']
    else: # Otherwise compute
        syn_sel_idx_mtype = np.logical_and(duplicate_sample_synapses.syn_sel_idx_type, np.isin(edges_table['@target_node'], tgt_node_ids[tgt_mtypes == tgt_mtype]))
        if np.sum(syn_sel_idx_mtype) == 0: # Ignore m-type, consider matching layer
            syn_sel_idx_mtype = np.logical_and(duplicate_sample_synapses.syn_sel_idx_type, np.isin(edges_table['@target_node'], tgt_node_ids[tgt_layers == tgt_layers[tidx]]))
        if np.sum(syn_sel_idx_mtype) == 0: # Otherwise, ignore m-type & layer
            syn_sel_idx_mtype = duplicate_sample_synapses.syn_sel_idx_type
            log.warning(f'No synapses with matching m-type or layer to sample connection property values for target neuron {tgt} from!')
        log.log_assert(np.sum(syn_sel_idx_mtype) > 0, f'No synapses to sample connection property values for target neuron {tgt} from!')
        _, num_syn_per_conn = np.unique(edges_table[syn_sel_idx_mtype][['@source_node', '@target_node']], axis=0, return_counts=True)
        duplicate_sample_synapses.per_mtype_dict[tgt_mtype] = {'syn_sel_idx_mtype': syn_sel_idx_mtype, 'num_syn_per_conn': num_syn_per_conn}
    num_syn_per_conn = num_syn_per_conn[np.random.choice(len(num_syn_per_conn), num_gen_conn)] # Sample #synapses/connection
    syn_conn_idx = np.concatenate([[i] * n for i, n in enumerate(num_syn_per_conn)]) # Create mapping from synapses to connections
    num_gen_syn = len(syn_conn_idx) # Number of synapses to generate

    # Duplicate num_gen_syn synapse positions on target neuron ['efferent_...' properties will no longer be consistent with actual source neuron's axon morphology!]
    sel_dupl = duplicate_synapses(syn_sel_idx, syn_sel_idx_tgt, syn_conn_idx, num_gen_syn, tgt)
    new_edges = edges_table.iloc[sel_dupl].copy()

    # Sample (non-morphology-related) property values independently from other existing synapses targetting neurons of the same mtype as tgt (incl. tgt)
    # => Assume identical (non-morphology-related) property values for synapses belonging to same connection (incl. delay, but this will (optionally) be re-assigned)!!
    for p in duplicate_sample_synapses.props_sel:
        new_edges[p] = edges_table.loc[syn_sel_idx_mtype, p].sample(num_gen_conn, replace=True).to_numpy()[syn_conn_idx]

    # Assign num_gen_syn synapses to num_gen_conn connections from src_gen to tgt
    new_edges['@source_node'] = src_gen[syn_conn_idx]

    # Assign distance-dependent delays (in-place), based on (generative) delay model (optional)
    assign_delays_from_model(delay_model, nodes, new_edges, src_gen, syn_conn_idx)

    return new_edges


def duplicate_randomize_synapses(src_gen, edges_table, nodes, syn_sel_idx, syn_sel_idx_tgt, tgt_id, delay_model, props_model):
    """Method to generate new synapses from source neurons <src_gen> to target neuron <tgt_id>, by duplicating existing
       synapse position & randomizing (non-morphology-related) properties based on pathway-specific model distributions."""
    # Init static variables (function attributes) related to this method which need only be initialized once [for better performance]
    if duplicate_randomize_synapses.reset:
        duplicate_randomize_synapses.props_sel = list(filter(lambda x: not np.any([excl in x for excl in ['_node', '_x', '_y', '_z', '_section', '_segment', '_length', 'delay']]), edges_table.columns)) # Non-morphology-related property selection (to be sampled/randomized)
        log.log_assert(np.all(np.isin(duplicate_randomize_synapses.props_sel, props_model.get_prop_names())), f'Required properties missing in properties model (must include: {duplicate_randomize_synapses.props_sel})!')
        duplicate_randomize_synapses.reset = False

    # Generate new synapse properties based on properties model
    src_mtypes = nodes[0].get(src_gen, properties='mtype').to_numpy()
    tgt_mtype = nodes[1].get(tgt_id, properties='mtype')
    new_syn_props = [props_model.apply(src_type=s, tgt_type=tgt_mtype) for s in src_mtypes]
    num_syn_per_conn = [syn.shape[0] for syn in new_syn_props]
    syn_conn_idx = np.concatenate([[i] * n for i, n in enumerate(num_syn_per_conn)]) # Create mapping from synapses to connections
    num_gen_syn = len(syn_conn_idx) # Total number of generated synapses

    # Duplicate num_gen_syn synapse positions on target neuron ['efferent_...' properties will no longer be consistent with actual source neuron's axon morphology!]
    sel_dupl = duplicate_synapses(syn_sel_idx, syn_sel_idx_tgt, syn_conn_idx, num_gen_syn, tgt_id)
    new_edges = edges_table.iloc[sel_dupl].copy()

    # Assign new synapse values for non-morphology-related properties
    new_edges[duplicate_randomize_synapses.props_sel] = pd.concat(new_syn_props, ignore_index=True)[duplicate_randomize_synapses.props_sel].to_numpy()

    # Assign num_gen_syn synapses to num_gen_conn connections from src_gen to tgt
    new_edges['@source_node'] = src_gen[syn_conn_idx]

    # Assign distance-dependent delays (in-place), based on (generative) delay model
    assign_delays_from_model(delay_model, nodes, new_edges, src_gen, syn_conn_idx)

    ##### [TESTING] #####
    for cidx in range(len(src_gen)):
        log.log_assert(len(np.unique(new_edges[syn_conn_idx == cidx]['u_syn'])) == 1, f'u_syn not unique within connection {src_gen[cidx]}-{tgt_id}!')
    ##### ######### #####

    return new_edges


def assign_delays_from_model(delay_model, nodes, edges_table, src_new, src_syn_idx, syn_sel_idx=None):
    """Assign new distance-dependent delays, drawn from truncated normal distribution, to new synapses within edges_table (in-place)."""
    log.log_assert(delay_model is not None, 'Delay model required!')

    if syn_sel_idx is None:
        syn_sel_idx = np.full(edges_table.shape[0], True)

    if len(src_new) == 0 or len(src_syn_idx) == 0 or np.sum(syn_sel_idx) == 0: # No synapses specified
        return

    # Determine distance from source neuron (soma) to synapse on target neuron
    src_new_pos = nodes[0].positions(src_new).to_numpy()
    syn_pos = edges_table.loc[syn_sel_idx, ['afferent_center_x', 'afferent_center_y', 'afferent_center_z']].to_numpy() # Synapse position on post-synaptic dendrite
    syn_dist = np.sqrt(np.sum((syn_pos - src_new_pos[src_syn_idx, :])**2, 1))

    # Obtain delay values from (generative) model
    delay_new = delay_model.apply(distance=syn_dist)

    # Assign to edges_table (in-place)
    edges_table.loc[syn_sel_idx, 'delay'] = delay_new
