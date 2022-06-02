"""
Manipulation name: conn_rewiring
Description: Rewiring of existing connectome, based on a given model of connection probability.
             Optionally, number of ingoing connections (and #synapses/connection) can be kept the same
             by re-using existing synapses. Otherwise, existing synapses may be deleted or new ones
             created, based on the selected generation method (Only "duplicate_sample" supported so far!).
"""

import os
import pickle

import numpy as np
from scipy.stats import truncnorm

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types, conn_prob
from connectome_manipulator.access_functions import get_node_ids


def apply(edges_table, nodes, aux_dict, syn_class, prob_model_file, sel_src=None, sel_dest=None, delay_model_file=None, pos_map_file=None, keep_indegree=True, gen_method=None, amount_pct=100.0):
    """Rewiring (interchange) of connections between pairs of neurons based on given conn. prob. model (re-using ingoing connections and optionally, creating/deleting synapses)."""
    log.log_assert(syn_class in ['EXC', 'INH'], f'Synapse class "{syn_class}" not supported (must be "EXC" or "INH")!')
    log.log_assert(0.0 <= amount_pct <= 100.0, 'amount_pct out of range!')
    if keep_indegree:
        log.log_assert(gen_method is None, f'Generation method {gen_method} not compatible with "keep_indegree" option!')
    else:
        log.log_assert(gen_method in ['duplicate_sample'], 'Valid generation method required (must be "duplicate_sample")!')
        # 'duplicate_sample' ... duplicate existing synapse position & sample (non-morphology-related) property values independently from existing synapses
        # 'duplicate_randomize' [NYI] ... duplicate existing synapse position & model-based randomization using pathway-specific distributions

    if gen_method == 'duplicate_sample' and aux_dict['N_split'] > 1:
        log.warning(f'"{gen_method}" method samples only from synapses within same data split! Reduce number of splits to 1 to sample from all synapses!')

    # Load connection probability model
    log.log_assert(os.path.exists(prob_model_file), 'Conn. prob. model file not found!')
    log.info(f'Loading conn. prob. model from {prob_model_file}')
    p_model = model_types.AbstractModel.model_from_file(prob_model_file)
    log.log_assert(p_model.input_names == ['src_pos', 'tgt_pos'], 'Conn. prob. model must have "src_pos" and "tgt_pos" as inputs!')
    log.info(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')

    # Load delay model (optional)
    if delay_model_file is not None:
        log.log_assert(os.path.exists(delay_model_file), 'Delay model file not found!')
        log.info(f'Loading delay model from {delay_model_file}')
        d_model = model_types.AbstractModel.model_from_file(delay_model_file)
        log.info(f'Loaded delay model of type "{d_model.__class__.__name__}"')
    else:
        d_model = None
        log.info('No delay model provided')

    # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
    _, pos_acc = conn_prob.load_pos_mapping_model(pos_map_file)    
    if pos_acc is None:
        log.info('No position mapping model provided')

    # Determine source/target nodes for rewiring
    stats_dict = {} # Keep track of statistics
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
    if np.sum(tgt_sel) == 0: # Nothing to rewire
        logging.info('No target nodes selected, nothing to rewire')
        return edges_table
    tgt_node_ids = tgt_node_ids[tgt_sel] # Select subset of neurons (keeping order)
    tgt_mtypes = nodes[1].get(tgt_node_ids, properties='mtype').to_numpy()
    tgt_layers = nodes[1].get(tgt_node_ids, properties='layer').to_numpy()

    if gen_method == 'duplicate_sample':
        if syn_class == 'EXC':
            syn_sel_idx_type = edges_table['syn_type_id'] >= 100
        elif syn_class == 'INH':
            syn_sel_idx_type = edges_table['syn_type_id'] < 100
        else:
            log.log_assert(False, f'Synapse class {syn_class} not supported!')

    log.info(f'Rewiring afferent {syn_class} connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest}, keep_indegree={keep_indegree}, gen_method={gen_method})')

    # Run connection rewiring
    props_sel = list(filter(lambda x: not np.any([excl in x for excl in ['_node', '_x', '_y', '_z', '_section', '_segment', '_length']]), edges_table.columns)) # Non-morphology-related property selection (to be sampled/randomized)
    syn_del_idx = np.full(edges_table.shape[0], False) # Global synapse indices to keep track of all unused synapses to be deleted
    all_new_edges = edges_table.loc[[]].copy() # New edges table to collect all generated synapses
    per_mtype_dict = {} # Dict to keep computed values per target m-type (instead of re-computing them for each target neuron)
    stats_dict['target_count'] = num_tgt # (Neurons)
    stats_dict['unable_to_rewire_count'] = 0 # (Neurons)
    progress_pct = np.round(100 * np.arange(len(tgt_node_ids)) / (len(tgt_node_ids) - 1)).astype(int)
    for tidx, tgt in enumerate(tgt_node_ids):
        if tidx == 0 or progress_pct[tidx - 1] != progress_pct[tidx]:
            print(f'{progress_pct[tidx]}%', end=' ' if progress_pct[tidx] < 100.0 else '\n') # Just for console, no logging

        syn_sel_idx_tgt = edges_table['@target_node'] == tgt
        syn_sel_idx = np.logical_and(syn_sel_idx_tgt, syn_sel_idx_src)
        num_sel = np.sum(syn_sel_idx)

        if (keep_indegree and num_sel == 0) or np.sum(syn_sel_idx_tgt) == 0:
            stats_dict['unable_to_rewire_count'] += 1 # (Neurons)
            continue # Nothing to rewire (no synapses on target node)

        # Determine conn. prob. of all source nodes to be connected with target node
        tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_acc is None else pos_acc, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
        p_src = p_model.apply(src_pos=src_pos, tgt_pos=tgt_pos).flatten()
        p_src[np.isnan(p_src)] = 0.0 # Exclude invalid values
        p_src[src_node_ids == tgt] = 0.0 # Exclude autapses [ASSUMING node IDs are unique across src/tgt node populations!]

        # Currently existing sources for given target node
        src, src_syn_idx = np.unique(edges_table.loc[syn_sel_idx, '@source_node'], return_inverse=True)
        num_src = len(src)

        # Sample new presynaptic neurons from list of source nodes according to conn. prob.
        if keep_indegree: # Keep the same number of ingoing connections (and #synapses/connection)
            log.log_assert(len(src_node_ids) >= num_src, f'Not enough source neurons for target neuron {tgt} available for rewiring!')
            src_new = np.random.choice(src_node_ids, size=num_src, replace=False, p=p_src / np.sum(p_src)) # New source node IDs per connection
        else: # Number of ingoing connections (and #synapses/connection) NOT kept the same
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel] # New source node IDs per connection
            num_new = len(src_new)

            # Re-use (up to) num_src existing connections for rewiring
            if num_src > num_new: # Delete unused connections/synapses
                syn_del_idx[syn_sel_idx] = src_syn_idx >= num_new # Set global indices of connections to be deleted
                syn_sel_idx[syn_del_idx] = False # Remove to-be-deleted indices from selection
                stats_dict['num_syn_removed'] = stats_dict.get('num_syn_removed', []) + [np.sum(src_syn_idx >= num_new)] # (Synapses)
                stats_dict['num_conn_removed'] = stats_dict.get('num_conn_removed', []) + [num_src - num_new] # (Connections)
                src_syn_idx = src_syn_idx[src_syn_idx < num_new]
            elif num_src < num_new: # Generate new synapses/connections, if needed
                num_gen_conn = num_new - num_src # Number of new connections to generate
                src_gen = src_new[-num_gen_conn:] # Split new sources into ones used for newly generated ...
                src_new = src_new[:num_src] # ... and existing connections

                if gen_method == 'duplicate_sample': # Duplicate existing synapse position & sample (non-morphology-related) property values independently from existing synapses
                    # Sample #synapses/connection from other existing synapses targetting neurons of the same mtype (or layer) as tgt (incl. tgt)
                    if tgt_mtypes[tidx] in per_mtype_dict.keys(): # Load from dict, if already exists [optimized for speed]
                        syn_sel_idx_mtype = per_mtype_dict[tgt_mtypes[tidx]]['syn_sel_idx_mtype']
                        num_syn_per_conn = per_mtype_dict[tgt_mtypes[tidx]]['num_syn_per_conn']
                    else: # Otherwise compute
                        syn_sel_idx_mtype = np.logical_and(syn_sel_idx_type, np.isin(edges_table['@target_node'], tgt_node_ids[tgt_mtypes == tgt_mtypes[tidx]]))
                        if np.sum(syn_sel_idx_mtype) == 0: # Ignore m-type, consider matching layer
                            syn_sel_idx_mtype = np.logical_and(syn_sel_idx_type, np.isin(edges_table['@target_node'], tgt_node_ids[tgt_layers == tgt_layers[tidx]]))
                        log.log_assert(np.sum(syn_sel_idx_mtype) > 0, f'No synapses to sample connection property values for target neuron {tgt} from!')
                        _, num_syn_per_conn = np.unique(edges_table[syn_sel_idx_mtype][['@source_node', '@target_node']], axis=0, return_counts=True)
                        per_mtype_dict[tgt_mtypes[tidx]] = {'syn_sel_idx_mtype': syn_sel_idx_mtype, 'num_syn_per_conn': num_syn_per_conn}
                    num_syn_per_conn = num_syn_per_conn[np.random.choice(len(num_syn_per_conn), num_gen_conn)] # Sample #synapses/connection
                    syn_conn_idx = np.concatenate([[i] * n for i, n in enumerate(num_syn_per_conn)]) # Create mapping from synapses to connections
                    num_gen_syn = len(syn_conn_idx) # Number of synapses to generate

                    # Duplicate num_gen_syn synapse positions on target neuron ['efferent_...' properties will no longer be consistent with actual source neuron's axon morphology!]
                    if num_sel > 0: # Duplicate only synapses of syn_class type
                        sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_gen_syn) # Random sampling from existing synapses with replacement
                    else: # Include all synapses, if no synapses of syn_class type are available
                        sel_dupl = np.random.choice(np.where(syn_sel_idx_tgt)[0], num_gen_syn) # Random sampling from existing synapses with replacement
                    new_edges = edges_table.iloc[sel_dupl].copy()

                    # Sample (non-morphology-related) property values independently from other existing synapses targetting neurons of the same mtype as tgt (incl. tgt)
                    # => Assume identical (non-morphology-related) property values for synapses belonging to same connection (incl. delay)!!
                    for p in props_sel:
                        new_edges[p] = edges_table.loc[syn_sel_idx_mtype, p].sample(num_gen_conn, replace=True).to_numpy()[syn_conn_idx]

                    # Assign num_gen_syn synapses to num_gen_conn connections from src_gen to tgt
                    new_edges['@source_node'] = src_gen[syn_conn_idx]
                    stats_dict['num_syn_added'] = stats_dict.get('num_syn_added', []) + [len(syn_conn_idx)] # (Synapses)
                    stats_dict['num_conn_added'] = stats_dict.get('num_conn_added', []) + [len(src_gen)] # (Connections)

                    # Assign distance-dependent delays (in-place), based on (generative) delay model (optional)
                    if d_model is not None:
                        assign_delays_from_model(d_model, nodes, new_edges, src_gen, syn_conn_idx)

                    # Add new_edges to global new edges table [ignoring duplicate indices]
                    all_new_edges = all_new_edges.append(new_edges)
                else:
                    log.log_assert(False, f'Generation method {gen_method} unknown!')
            else: # num_src == num_new
                # Exact match: nothing to add, nothing to delete
                pass

        # Assign new source nodes = rewiring of existing connections
        edges_table.loc[syn_sel_idx, '@source_node'] = src_new[src_syn_idx] # Source node IDs per connection expanded to synapses
        stats_dict['num_syn_rewired'] = stats_dict.get('num_syn_rewired', []) + [len(src_syn_idx)] # (Synapses)
        stats_dict['num_conn_rewired'] = stats_dict.get('num_conn_rewired', []) + [len(src_new)] # (Connections)

        # Assign new distance-dependent delays (in-place), based on (generative) delay model (optional)
        if d_model is not None:
            assign_delays_from_model(d_model, nodes, edges_table, src_new, src_syn_idx, syn_sel_idx)

    # Delete unused synapses (if any)
    if np.any(syn_del_idx):
        edges_table = edges_table[~syn_del_idx].copy()
        log.info(f'Deleted {np.sum(syn_del_idx)} unused synapses')

    # Add new synapses to table, re-sort, and assign new index
    if all_new_edges.size > 0:
        edges_table = edges_table.append(all_new_edges)
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required when merging files in block-based processing]
        log.info(f'Generated {all_new_edges.shape[0]} new synapses')

    # Print statistics
    stat_str = [f'      {k}: COUNT {len(v)}, MEAN {np.mean(v):.2f}, MIN {np.min(v)}, MAX {np.max(v)}, SUM {np.sum(v)}' if isinstance(v, list) else f'      {k}: {v}' for k, v in stats_dict.items()]
    log.info('STATISTICS:\n%s', '\n'.join(stat_str))

    return edges_table


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
