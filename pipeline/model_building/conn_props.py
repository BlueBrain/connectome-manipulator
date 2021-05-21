# Model building function
#
# Three functions need to be defined 
# (1) extract(...): extracting connectivity specific data
# (2) build(...): building a data-based model
# (3) plot(...): visualizing data vs. model

from model_building import model_building
import os.path
import progressbar
import numpy as np
import matplotlib.pyplot as plt

# TODO: Visualize and capture actual distributions of synaptic properties
#       Visualize and capture correlations between synaptic properties

""" Extract statistics for synaptic properties between samples of neurons for each pair of m-types """
def extract(circuit, min_sample_size_per_group=None, max_sample_size_per_group=None, **_):
    
    # Select edge population [assuming exactly one edge population in given edges file]
    assert len(circuit.edges.population_names) == 1, 'ERROR: Only a single edge population per file supported for modelling!'
    edges = circuit.edges[circuit.edges.population_names[0]]
    
    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]
    
    m_types = [sorted(n.property_values('mtype', is_present=True)) for n in nodes]
    m_type_class = [[nodes[i].get({'mtype': m}, properties='synapse_class').iloc[0] for m in m_types[i]] for i in range(len(nodes))]
    m_type_layer = [[nodes[i].get({'mtype': m}, properties='layer').iloc[0] for m in m_types[i]] for i in range(len(nodes))]
    syn_props = list(filter(lambda x: not np.any([excl in x for excl in ['@', 'delay', 'afferent', 'efferent', 'spine_length']]), edges.property_names))
    
    print(f'INFO: Estimating statistics for {len(syn_props)} properties between {len(m_types[0])}x{len(m_types[1])} m-types', flush=True)
    
    # Statistics for #syn/conn
    syns_per_conn_data = {'mean': np.full((len(m_types[0]), len(m_types[1])), np.nan),
                          'std': np.full((len(m_types[0]), len(m_types[1])), np.nan),
                          'min': np.full((len(m_types[0]), len(m_types[1])), np.nan),
                          'max': np.full((len(m_types[0]), len(m_types[1])), np.nan)}
    
    # Statistics for synapse/connection properties
    conn_prop_data = {'mean': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value means across connections
                      'std': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value stds across connections
                      'std-within': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value stds across synapses within connections
                      'min': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value overall min
                      'max': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan)} # Property value overall max
    
    # Extract statistics
    conn_counts = {'min': np.inf, 'max': -np.inf, 'sel': 0} # Count connections for reporting
    pbar = progressbar.ProgressBar()
    for sidx in pbar(range(len(m_types[0]))):
        sids = src_nodes.ids({'mtype': m_types[0][sidx]})
        for tidx in range(len(m_types[1])):
            tids = tgt_nodes.ids({'mtype': m_types[1][tidx]})
            edges_sel = edges.pathway_edges(sids, tids, ['@source_node', '@target_node'] + syn_props)
            if edges_sel.shape[0] == 0: # No synapses between pair of m-types
                continue
            
            conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_sel[['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
            conn_counts['min'] = min(conn_counts['min'], len(num_syn_per_conn))
            conn_counts['max'] = max(conn_counts['max'], len(num_syn_per_conn))
            conn_sel = range(len(num_syn_per_conn)) # Select all connections
            if not min_sample_size_per_group is None and min_sample_size_per_group > 0 and len(conn_sel) < min_sample_size_per_group: # Not enough connections available
                continue
            if not max_sample_size_per_group is None and max_sample_size_per_group > 0 and len(conn_sel) > max_sample_size_per_group: # Subsample connections
                conn_sel = sorted(np.random.choice(conn_sel, max_sample_size_per_group, replace=False))
            conn_counts['sel'] += 1
            
            syns_per_conn_data['mean'][sidx, tidx] = np.mean(num_syn_per_conn[conn_sel])
            syns_per_conn_data['std'][sidx, tidx] = np.std(num_syn_per_conn[conn_sel])
            syns_per_conn_data['min'][sidx, tidx] = np.min(num_syn_per_conn[conn_sel])
            syns_per_conn_data['max'][sidx, tidx] = np.max(num_syn_per_conn[conn_sel])
            
            means_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            stds_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            mins_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            maxs_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            for cidx, c in enumerate(conn_sel):
                means_within[cidx, :] = np.mean(edges_sel.loc[syn_conn_idx==c, syn_props], 0)
                stds_within[cidx, :] = np.std(edges_sel.loc[syn_conn_idx==c, syn_props], 0)
                mins_within[cidx, :] = np.min(edges_sel.loc[syn_conn_idx==c, syn_props], 0)
                maxs_within[cidx, :] = np.max(edges_sel.loc[syn_conn_idx==c, syn_props], 0)
            
            conn_prop_data['mean'][sidx, tidx, :] = np.mean(means_within, 0)
            conn_prop_data['std'][sidx, tidx, :] = np.std(means_within, 0)
            conn_prop_data['std-within'][sidx, tidx, :] = np.mean(stds_within, 0)
            conn_prop_data['min'][sidx, tidx, :] = np.min(mins_within, 0)
            conn_prop_data['max'][sidx, tidx, :] = np.max(maxs_within, 0)
    
    print(f'INFO: Between {conn_counts["min"]} and {conn_counts["max"]} connections per pathway found. {conn_counts["sel"]} of {len(m_types[0])}x{len(m_types[1])} pathways selected.')
    
    return {'syns_per_conn_data': syns_per_conn_data, 'conn_prop_data': conn_prop_data, 'm_types': m_types, 'm_type_class': m_type_class, 'm_type_layer': m_type_layer, 'syn_props': syn_props}


""" Build model from data (lookup table with missing values interpolated at different levels of granularity) """
def build(syns_per_conn_data, conn_prop_data, m_types, m_type_class, m_type_layer, syn_props, **_):
    
    # Interpolate missing values in lookup tables
    syns_per_conn_model = {k: v.copy() for (k, v) in syns_per_conn_data.items()}
    conn_prop_model = {k: v.copy() for (k, v) in conn_prop_data.items()}
    missing_list = np.array(np.where(np.logical_not(np.isfinite(syns_per_conn_model['mean'])))).T
    level_counts = {} # Count interpolation levels for reporting
    for (sidx, tidx) in missing_list:
        # Select level of granularity
        for level in range(5):
            if level == 0: # Use source m-type/target layer/synapse class value, if existent
                src_sel = [sidx]
                tgt_sel = np.logical_and(np.array(m_type_layer[1]) == m_type_layer[1][tidx], np.array(m_type_class[1]) == m_type_class[1][tidx])
            elif level == 1: # Use source m-type/target synapse class value, if existent
                src_sel = [sidx]
                tgt_sel = np.array(m_type_class[1]) == m_type_class[1][tidx]
            elif level == 2: # Use per layer/synapse class value, if existent
                src_sel = np.logical_and(np.array(m_type_layer[0]) == m_type_layer[0][sidx], np.array(m_type_class[0]) == m_type_class[0][sidx])
                tgt_sel = np.logical_and(np.array(m_type_layer[1]) == m_type_layer[1][tidx], np.array(m_type_class[1]) == m_type_class[1][tidx])
            elif level == 3: # Use per synapse class value, if existent
                src_sel = np.array(m_type_class[0]) == m_type_class[0][sidx]
                tgt_sel = np.array(m_type_class[1]) == m_type_class[1][tidx]
            else: # Otherwise: Use overall value
                src_sel = range(len(m_types[0]))
                tgt_sel = range(len(m_types[1]))
            if np.any(np.isfinite(syns_per_conn_data['mean'][src_sel, :][:, tgt_sel])):
                level_counts[f'Level{level}'] = level_counts.get(f'Level{level}', 0) + 1
                break
        
        # Interpolate missing values
        syns_per_conn_model['mean'][sidx, tidx] = np.nanmean(syns_per_conn_data['mean'][src_sel, :][:, tgt_sel])
        syns_per_conn_model['std'][sidx, tidx] = np.nanmean(syns_per_conn_data['std'][src_sel, :][:, tgt_sel])
        syns_per_conn_model['min'][sidx, tidx] = np.nanmin(syns_per_conn_data['min'][src_sel, :][:, tgt_sel])
        syns_per_conn_model['max'][sidx, tidx] = np.nanmax(syns_per_conn_data['max'][src_sel, :][:, tgt_sel])
        conn_prop_model['mean'][sidx, tidx, :] = [np.nanmean(conn_prop_data['mean'][src_sel, :, p][:, tgt_sel]) for p in range(len(syn_props))]
        conn_prop_model['std'][sidx, tidx, :] = [np.nanmean(conn_prop_data['std'][src_sel, :, p][:, tgt_sel]) for p in range(len(syn_props))]
        conn_prop_model['std-within'][sidx, tidx, :] = [np.nanmean(conn_prop_data['std-within'][src_sel, :, p][:, tgt_sel]) for p in range(len(syn_props))]
        conn_prop_model['min'][sidx, tidx, :] = [np.nanmin(conn_prop_data['min'][src_sel, :, p][:, tgt_sel]) for p in range(len(syn_props))]
        conn_prop_model['max'][sidx, tidx, :] = [np.nanmax(conn_prop_data['max'][src_sel, :, p][:, tgt_sel]) for p in range(len(syn_props))]
    
    print(f'INFO: Interpolated {missing_list.shape[0]} missing values. Interpolation level counts: { {k: level_counts[k] for k in sorted(level_counts.keys())} }')
    
    # Create model dictionary (lookup-table)
    prop_model_dict = {syn_props[p]: {m_types[0][s]: {m_types[1][t]: {k: conn_prop_model[k][s, t, p] for k in conn_prop_model.keys()} for t in range(len(m_types[1]))} for s in range(len(m_types[0]))} for p in range(len(syn_props))}
    prop_model_dict.update({'n_syn_per_conn': {m_types[0][s]: {m_types[1][t]: {k: syns_per_conn_model[k][s, t] for k in syns_per_conn_model.keys()} for t in range(len(m_types[1]))} for s in range(len(m_types[0]))}})
    
    print(f'MODEL FIT for synapse/connection properties ({len(m_types[0])}x{len(m_types[1])} m-types):')
    print(list(prop_model_dict.keys()))
    
    return {'model': 'prop_model_dict[prop_name][src_type][tgt_type][stat_type]',
            'model_inputs': ['prop_name', 'src_type', 'tgt_type', 'stat_type'],
            'model_params': {'prop_model_dict': prop_model_dict}}


""" Visualize data vs. model """
def plot(out_dir, syns_per_conn_data, conn_prop_data, m_types, syn_props, model, model_inputs, model_params, **_):
    
    model_fct = model_building.get_model(model, model_inputs, model_params)
    prop_names = syn_props + ['n_syn_per_conn']
    
    # Draw figure
    data_sel = 'mean' # Plot mean only
    for pidx, p in enumerate(prop_names):
        plt.figure(figsize=(8, 3), dpi=300)
        for didx, data in enumerate([conn_prop_data[data_sel][:, :, pidx] if pidx < conn_prop_data[data_sel].shape[2] else syns_per_conn_data[data_sel], np.array([[model_fct(p, s, t, data_sel) for t in m_types[1]] for s in m_types[0]])]):
            plt.subplot(1, 2, didx + 1)
            plt.imshow(data, interpolation='nearest', cmap='jet')
            plt.xticks(range(len(m_types[1])), m_types[0], rotation=90, fontsize=3)
            plt.yticks(range(len(m_types[1])), m_types[0], rotation=0, fontsize=3)
            plt.colorbar()
        plt.suptitle(p)
        plt.tight_layout()
        
        out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model__{p}.png'))
        print(f'INFO: Saving {out_fn}...')
        plt.savefig(out_fn)
    
    return
