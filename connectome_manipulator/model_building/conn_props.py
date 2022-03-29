"""
Module for building connection/synapse properties model, consisting of three basic functions:
  -extract(...): Extracts statistics for connection/synaptic properties between samples of neurons for each pair of m-types
  -build(...): Fit model distribution to data, incl. missing values interpolated at different levels of granularity
  -plot(...): Visualizes extracted data vs. actual model output
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_edges_population

DISTRIBUTION_ATTRIBUTES = {'constant': ['mean'],
                           'normal': ['mean', 'std'],
                           'truncnorm': ['mean', 'std', 'min', 'max'],
                           'gamma': ['mean', 'std'],
                           'poisson': ['mean']}

# Ideas for improvement:
#   *Restrict to given node set!!
#   *Detect actual distributions of synaptic properties (incl. data type!)
#   *Capture cross-correlations between synaptic properties


def extract(circuit, min_sample_size_per_group=None, max_sample_size_per_group=None, hist_bins=50, **_):
    """Extract statistics for synaptic properties between samples of neurons for each pair of m-types."""
    # Select edge population [assuming exactly one edge population in given edges file]
    edges = get_edges_population(circuit)

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
                          'max': np.full((len(m_types[0]), len(m_types[1])), np.nan),
                          'hist': [[[] for j in range(len(m_types[1]))] for i in range(len(m_types[0]))]}

    # Statistics for synapse/connection properties
    conn_prop_data = {'mean': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value means across connections
                      'std': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value stds across connections
                      'std-within': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value stds across synapses within connections
                      'min': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value overall min
                      'max': np.full((len(m_types[0]), len(m_types[1]), len(syn_props)), np.nan), # Property value overall max
                      'hist': [[[[] for k in range(len(syn_props))] for j in range(len(m_types[1]))] for i in range(len(m_types[0]))]} # Histogram of distribution

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

            _, syn_conn_idx, num_syn_per_conn = np.unique(edges_sel[['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
            conn_counts['min'] = min(conn_counts['min'], len(num_syn_per_conn))
            conn_counts['max'] = max(conn_counts['max'], len(num_syn_per_conn))
            conn_sel = range(len(num_syn_per_conn)) # Select all connections
            if min_sample_size_per_group is not None and min_sample_size_per_group > 0 and len(conn_sel) < min_sample_size_per_group: # Not enough connections available
                continue
            if max_sample_size_per_group is not None and 0 < max_sample_size_per_group < len(conn_sel): # Subsample connections
                conn_sel = sorted(np.random.choice(conn_sel, max_sample_size_per_group, replace=False))
            conn_counts['sel'] += 1

            syns_per_conn_data['mean'][sidx, tidx] = np.mean(num_syn_per_conn[conn_sel])
            syns_per_conn_data['std'][sidx, tidx] = np.std(num_syn_per_conn[conn_sel])
            syns_per_conn_data['min'][sidx, tidx] = np.min(num_syn_per_conn[conn_sel])
            syns_per_conn_data['max'][sidx, tidx] = np.max(num_syn_per_conn[conn_sel])
            syns_per_conn_data['hist'][sidx][tidx] = np.histogram(num_syn_per_conn[conn_sel], bins=hist_bins)

            means_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            stds_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            mins_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            maxs_within = np.full((len(conn_sel), len(syn_props)), np.nan)
            for cidx, c in enumerate(conn_sel):
                means_within[cidx, :] = np.mean(edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0, dtype=np.float64) # [float64 required, otherwise rounding problems!!!]
                stds_within[cidx, :] = np.std(edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0, dtype=np.float64)
                mins_within[cidx, :] = np.min(edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0).astype(np.float64)
                maxs_within[cidx, :] = np.max(edges_sel.loc[syn_conn_idx == c, syn_props].to_numpy(), 0).astype(np.float64)

            conn_prop_data['mean'][sidx, tidx, :] = np.mean(means_within, 0)
            conn_prop_data['std'][sidx, tidx, :] = np.std(means_within, 0)
            conn_prop_data['std-within'][sidx, tidx, :] = np.mean(stds_within, 0)
            conn_prop_data['min'][sidx, tidx, :] = np.min(mins_within, 0)
            conn_prop_data['max'][sidx, tidx, :] = np.max(maxs_within, 0)
            for pidx in range(len(syn_props)):
                conn_prop_data['hist'][sidx][tidx][pidx] = np.histogram(means_within[:, pidx], bins=hist_bins)

    print(f'INFO: Between {conn_counts["min"]} and {conn_counts["max"]} connections per pathway found. {conn_counts["sel"]} of {len(m_types[0])}x{len(m_types[1])} pathways selected.')

    return {'syns_per_conn_data': syns_per_conn_data, 'conn_prop_data': conn_prop_data, 'm_types': m_types, 'm_type_class': m_type_class, 'm_type_layer': m_type_layer, 'syn_props': syn_props, 'hist_bins': hist_bins}


def build(syns_per_conn_data, conn_prop_data, m_types, m_type_class, m_type_layer, syn_props, distr_types={}, data_types={}, data_bounds={}, **_):
    """Build model from data (lookup table with missing values interpolated at different levels of granularity)."""
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

    # Create model properties dictionary
    prop_model_dict = {}
    for pidx, prop in enumerate(syn_props + ['n_syn_per_conn']):
        prop_model_dict[prop] = {}
        if not prop in distr_types:
            print(f'WARNING: No distribution type for "{prop}" specified - Using "normal"!')
        distr_type = distr_types.get(prop, 'normal')
        assert distr_type in DISTRIBUTION_ATTRIBUTES, f'ERROR: Distribution type "{distr_type}" not supported!'
        dtype = data_types.get(prop)
        bounds = data_bounds.get(prop)
        for sidx, src in enumerate(m_types[0]):
            prop_model_dict[prop][src] = {}
            for tidx, tgt in enumerate(m_types[1]):
                attr_dict = {'type': distr_type}
                distr_attr = DISTRIBUTION_ATTRIBUTES[distr_type]
                if prop == 'n_syn_per_conn':
                    attr_dict.update({attr: syns_per_conn_model[attr][sidx, tidx] for attr in distr_attr})
                else:
                    if np.any(conn_prop_model['std-within'][sidx, tidx, pidx] > 0.0):
                        distr_attr = distr_attr + ['std-within']
                    attr_dict.update({attr: conn_prop_model[attr][sidx, tidx, pidx] for attr in distr_attr})
                if dtype is not None:
                    attr_dict.update({'dtype': dtype})
                if bounds is not None and hasattr(bounds, '__iter__') and len(bounds) == 2:
                    if bounds[0] is not None:
                        attr_dict.update({'lower_bound': bounds[0]})
                    if bounds[1] is not None:
                        attr_dict.update({'upper_bound': bounds[1]})
                prop_model_dict[prop][src][tgt] = attr_dict

    # Create model
    model = model_types.ConnPropsModel(src_types=m_types[0], tgt_types=m_types[1], prop_stats=prop_model_dict)
    print('MODEL:', end=' ')
    print(model.get_model_str())

    return model


def plot(out_dir, syns_per_conn_data, conn_prop_data, m_types, syn_props, model, hist_bins, **_):
    """Visualize data vs. model."""
    model_params = model.get_param_dict()
    prop_names = syn_props + ['n_syn_per_conn']

    # Plot data vs. model: property maps
    title_str = ['Data', 'Model']
    for stat_sel in ['mean', 'std']:
        for pidx, p in enumerate(prop_names):
            plt.figure(figsize=(8, 3), dpi=300)
            for didx, data in enumerate([conn_prop_data[stat_sel][:, :, pidx] if pidx < conn_prop_data[stat_sel].shape[2] else syns_per_conn_data[stat_sel], np.array([[model_params['prop_stats'][p][s][t][stat_sel] for t in m_types[1]] for s in m_types[0]])]):
                plt.subplot(1, 2, didx + 1)
                plt.imshow(data, interpolation='nearest', cmap='jet')
                plt.xticks(range(len(m_types[1])), m_types[0], rotation=90, fontsize=3)
                plt.yticks(range(len(m_types[1])), m_types[0], rotation=0, fontsize=3)
                plt.colorbar()
                plt.title(title_str[didx])
            plt.suptitle(f'{p} ({stat_sel})', fontweight='bold')
            plt.tight_layout()

            out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model_map_{stat_sel}__{p}.png'))
            print(f'INFO: Saving {out_fn}...')
            plt.savefig(out_fn)

    # Plot data vs. model: Distribution histogram examples (generative model)
    N = 1000 # Number of samples
    conn_counts = [[np.sum(syns_per_conn_data['hist'][sidx][tidx][0]) if len(syns_per_conn_data['hist'][sidx][tidx]) > 0 else 0 for sidx in range(len(m_types[0]))] for tidx in range(len(m_types[1]))]
    max_pathways = np.where(np.array(conn_counts) == np.max(conn_counts)) # Select pathway(s) with maximum number of connections (i.e., most robust statistics)
    sidx, tidx = [max_pathways[i][0] for i in range(len(max_pathways))] # Select first of these pathways for plotting
    src, tgt = [m_types[0][sidx], m_types[1][tidx]]
    for pidx, p in enumerate(prop_names):
        plt.figure(figsize=(5, 3), dpi=300)
        if pidx < len(syn_props):
            data_hist = conn_prop_data['hist'][sidx][tidx][pidx]
        else:
            data_hist = syns_per_conn_data['hist'][sidx][tidx]
        plt.bar(data_hist[1][:-1], data_hist[0] / np.sum(data_hist[0]), align='edge', width=np.min(np.diff(data_hist[1])), label=f'Data (N={np.max(conn_counts)})')
        model_data = np.hstack([model.draw(prop_name=p, src_type=src, tgt_type=tgt) for n in range(N)])
        model_hist = np.histogram(model_data, bins=hist_bins)
        plt.step(model_hist[1], np.hstack([model_hist[0][0], model_hist[0]]) / np.sum(model_hist[0]), where='pre', color='tab:orange', label=f'Model (N={N})')
        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.title(f'{src} to {tgt}', fontweight='bold')
        plt.xlabel(p)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model_hist__{p}.png'))
        print(f'INFO: Saving {out_fn}...')
        plt.savefig(out_fn)
