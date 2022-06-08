"""
Manipulation name: conn_wiring
Description: Special case of connectome rewiring, which wires an empty connectome from scratch, or simply
             adds connections to an existing connectome (edges table)!
             Only specific properties like source/target node, afferent synapse positions, synapse type
             (INH: 0, EXC: 100), and delay (optional) will be generated.
"""

import os

import neurom as nm
import numpy as np

from bluepysnap.morph import MorphHelper
from connectome_manipulator import log
from connectome_manipulator.model_building import model_types, conn_prob
from connectome_manipulator.access_functions import get_node_ids

# IDEAs for improvements:
#   Accept model file name OR model dict for prob_model_file, nsynconn_model_file, delay_model_file
#   Add model for synapse placement

def apply(edges_table, nodes, aux_dict, prob_model_file, nsynconn_model_file, sel_src=None, sel_dest=None, delay_model_file=None, pos_map_file=None, amount_pct=100.0, morph_ext='swc'):
    """Wiring (generation) of structural connections between pairs of neurons based on given conn. prob. model.
       => Only structural synapse properties will be set: PRE/POST neuron IDs, synapse positions, type, axonal delays"""
    log.log_assert(0.0 <= amount_pct <= 100.0, 'amount_pct out of range!')
    if edges_table.shape[0] > 0:
        log.warning(f'Initial connectome not empty ({edges_table.shape[0]} synapses, {edges_table.shape[1]} properties)! Connections will be added to existing connectome. Existing properties may be removed to match newly generated synapses.')

    # Load connection probability model
    log.log_assert(os.path.exists(prob_model_file), 'Conn. prob. model file not found!')
    log.info(f'Loading conn. prob. model from {prob_model_file}')
    p_model = model_types.AbstractModel.model_from_file(prob_model_file)
    log.log_assert(p_model.input_names == ['src_pos', 'tgt_pos'], 'Conn. prob. model must have "src_pos" and "tgt_pos" as inputs!')
    log.info(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')

    # Load #synapses/connection model
    log.log_assert(os.path.exists(nsynconn_model_file), '#synapses/connection model file not found!')
    log.info(f'Loading #synapses/connection model from {nsynconn_model_file}')
    nsynconn_model = model_types.AbstractModel.model_from_file(nsynconn_model_file)

    # Load delay model (optional)
    if delay_model_file is not None:
        log.log_assert(os.path.exists(delay_model_file), 'Delay model file not found!')
        log.info(f'Loading delay model from {delay_model_file}')
        delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
        log.info(f'Loaded delay model of type "{delay_model.__class__.__name__}"')
    else:
        delay_model = None
        log.info('No delay model provided')

    # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
    _, pos_acc = conn_prob.load_pos_mapping_model(pos_map_file)    
    if pos_acc is None:
        log.info('No position mapping model provided')

    # Determine source/target nodes for wiring
    src_node_ids = get_node_ids(nodes[0], sel_src)
    src_class = nodes[0].get(src_node_ids, properties='synapse_class')
    src_mtypes = nodes[0].get(src_node_ids, properties='mtype').to_numpy()
    log.log_assert(len(src_node_ids) > 0, f'No source nodes selected!')
    src_pos = conn_prob.get_neuron_positions(nodes[0].positions if pos_acc is None else pos_acc, [src_node_ids])[0] # Get neuron positions (incl. position mapping, if provided)

    tgt_node_ids = get_node_ids(nodes[1], sel_dest)
    num_tgt_total = len(tgt_node_ids)
    tgt_node_ids = np.intersect1d(tgt_node_ids, aux_dict['split_ids']) # Only select target nodes that are actually in current split of edges_table
    num_tgt = np.round(amount_pct * len(tgt_node_ids) / 100).astype(int)
    tgt_sel = np.random.permutation([True] * num_tgt + [False] * (len(tgt_node_ids) - num_tgt))
    if np.sum(tgt_sel) == 0: # Nothing to wire
        logging.info('No target nodes selected, nothing to wire')
        return edges_table
    tgt_node_ids = tgt_node_ids[tgt_sel] # Select subset of neurons (keeping order)
    tgt_mtypes = nodes[1].get(tgt_node_ids, properties='mtype').to_numpy()
    tgt_layers = nodes[1].get(tgt_node_ids, properties='layer').to_numpy()

    log.info(f'Generating afferent connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest})')

    # Prepare to load target (dendritic) morphologies
    # tgt_morph = nodes[1].morph ### ERROR with path/file format!
    # get_tgt_morph = lambda node_id: tgt_morph.get(node_id, transform=True) # Access function
    morph_dir = nodes[1].config['morphologies_dir']
    tgt_morph = MorphHelper(morph_dir, nodes[1], {'h5v1': os.path.join(morph_dir, 'h5v1'), 'neurolucida-asc': os.path.join(morph_dir, 'ascii')})
    get_tgt_morph = lambda node_id: tgt_morph.get(node_id, transform=True, extension=morph_ext) # Access function (incl. transformation!), using specified format (swc/h5/...)

    # Run connection wiring
    all_new_edges = edges_table.loc[[]].copy() # New edges table to collect all generated synapses
    progress_pct = np.round(100 * np.arange(len(tgt_node_ids)) / (len(tgt_node_ids) - 1)).astype(int)
    for tidx, tgt in enumerate(tgt_node_ids):
        if tidx == 0 or progress_pct[tidx - 1] != progress_pct[tidx]:
            print(f'{progress_pct[tidx]}%', end=' ' if progress_pct[tidx] < 100.0 else '\n') # Just for console, no logging

        # Determine conn. prob. of all source nodes to be connected with target node
        tgt_pos = conn_prob.get_neuron_positions(nodes[1].positions if pos_acc is None else pos_acc, [[tgt]])[0] # Get neuron positions (incl. position mapping, if provided)
        p_src = p_model.apply(src_pos=src_pos, tgt_pos=tgt_pos).flatten()
        p_src[np.isnan(p_src)] = 0.0 # Exclude invalid values
        p_src[src_node_ids == tgt] = 0.0 # Exclude autapses [ASSUMING node IDs are unique across src/tgt node populations!]

        # Sample new presynaptic neurons from list of source nodes according to conn. prob.
        src_new_sel = np.random.rand(len(src_node_ids)) < p_src
        src_new = src_node_ids[src_new_sel] # New source node IDs per connection
        num_new = len(src_new)
        if num_new == 0:
            continue # Nothing to wire

        # Sample number of synapses per connection
        num_syn_per_conn = [nsynconn_model.draw(model_types.N_SYN_PER_CONN_NAME, src_type=s, tgt_type=tgt_mtypes[tidx])[0] for s in src_mtypes[src_new_sel]]
        syn_conn_idx = np.concatenate([[i] * n for i, n in enumerate(num_syn_per_conn)]) # Create mapping from synapses to connections
        num_gen_syn = len(syn_conn_idx) # Number of synapses to generate

        # Create new synapses
        new_edges = edges_table.loc[[]].copy() # Initialize empty
        new_edges['@source_node'] = src_new[syn_conn_idx] # Source node IDs per connection expanded to synapses
        new_edges['@target_node'] = tgt
        
        # Place synapses randomly on soma/dendrite sections
        # [TODO: Add model for synapse placement??]
        morph = get_tgt_morph(tgt)
        sec_ind = np.hstack([[-1], np.where(np.isin(morph.section_types, [nm.BASAL_DENDRITE, nm.APICAL_DENDRITE]))[0]]) # Soma/dendrite section indices; soma...-1

        sec_sel = np.random.choice(sec_ind, len(syn_conn_idx)) # Randomly choose section indices
        off_sel = np.random.rand(len(syn_conn_idx)) # Randomly choose fractional offset within each section
        off_sel[sec_sel == -1] = 0.0 # Soma offsets must be zero
        type_sel = [int(morph.section(sec).type) if sec >= 0 else 0 for sec in sec_sel] # Type 0: Soma (1: Axon, 2: Basal, 3: Apical)
        pos_sel = np.array([nm.morphmath.path_fraction_point(morph.section(sec).points, off) if sec >= 0 else morph.soma.center.astype(float) for sec, off in zip(sec_sel, off_sel)]) # Synapse positions, computed from section & offset
        syn_type = np.select([src_class[new_edges['@source_node']].to_numpy() == 'INH', src_class[new_edges['@source_node']].to_numpy() == 'EXC'], [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)]) # INH: 0-99 (Using 0); EXC: >=100 (Using 100)

        new_edges['afferent_section_id'] = sec_sel + 1 # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
        new_edges['afferent_section_pos'] = off_sel
        new_edges['afferent_section_type'] = type_sel
        new_edges[['afferent_center_x', 'afferent_center_y', 'afferent_center_z']] = pos_sel
        new_edges['syn_type_id'] = syn_type

        # Assign distance-dependent delays, based on (generative) delay model (optional)
        if delay_model is not None:
            src_new_pos = nodes[0].positions(src_new).to_numpy()
            syn_dist = np.sqrt(np.sum((pos_sel - src_new_pos[syn_conn_idx, :])**2, 1)) # Distance from source neurons (soma) to synapse positions on target neuron
            new_edges['delay'] = delay_model.apply(distance=syn_dist)

        # Add new_edges to edges table
        all_new_edges = all_new_edges.append(new_edges)

    # Drop empty (NaN) columns [OTHERWISE: Problem converting to SONATA]
    init_prop_count = all_new_edges.shape[1]
    if all_new_edges.shape[0] > 0:
        all_new_edges.dropna(axis=1, inplace=True, how='all') # Drop empty/unused columns
    unused_props = np.setdiff1d(edges_table.keys(), all_new_edges.keys())
    edges_table = edges_table.drop(unused_props, axis=1) # Drop in original table as well, to avoid inconsistencies!
    final_prop_count = all_new_edges.shape[1]

    # Add new synapses to table, re-sort, and assign new index
    init_edge_count = edges_table.shape[0]
    edges_table = edges_table.append(all_new_edges)
    final_edge_count = edges_table.shape[0]
    if final_edge_count > init_edge_count:
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required when merging files in block-based processing]

    log.info(f'Generated {final_edge_count - init_edge_count} (of {edges_table.shape[0]}) new synapses with {final_prop_count} properties ({init_prop_count - final_prop_count} removed)')

    return edges_table
