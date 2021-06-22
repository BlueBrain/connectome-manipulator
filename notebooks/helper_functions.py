# Helper functions

import sys
import os.path
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import json
from toposample import Config
from toposample.db import get_column_from_database

""" Combined plotting of all topological parameters in one figure with num_rows rows
    [Modified from plot_comparison(...) in topological_comparator/bin/compare_topo_db.py] """
def plot_topodb_comparison(db_dict, param_dict, groupby, out_dir, num_rows=1, show_fig=False):
    
    cname1, cname2 = list(db_dict.keys())
    out_dir = os.path.join(out_dir, cname1 + '_vs_' + cname2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for circ, db in db_dict.items():
        db['Circuit'] = circ
        db[''] = '' # Add dummy column, so that empty groupby is supported
    full_db = pandas.concat(db_dict.values(), axis=0)
    
    if not num_rows:
        num_rows = 1
    if num_rows < 1:
        num_rows = 1
    if num_rows > len(param_dict):
        num_rows = len(param_dict)
    
    num_cols = np.ceil(len(param_dict) / num_rows).astype(int)
    figsize_x = 3 * num_cols
    figsize_y = num_rows * (4 + len(full_db[groupby].drop_duplicates()) / 3)
    plt.figure(figsize=(figsize_x, figsize_y))
    for param_idx, (param_name, param_spec) in enumerate(param_dict.items()):
        try:
            ax = plt.subplot(num_rows, num_cols, param_idx + 1)
            plot_frame = full_db[['Circuit', groupby]].copy()
            plot_frame[param_name] = get_column_from_database(full_db, param_spec['column'],
                                                              index=param_spec.get('index', None),
                                                              function=param_spec.get('function', None))
            sns.violinplot(y=groupby, x=param_name, hue='Circuit', split=True, data=plot_frame, ax=ax, orient='h')
            if param_idx % num_cols != 0:
                ax.set_ylabel(None)
            ax.legend(loc='lower right', fontsize=6, bbox_to_anchor=(1.0, 1.0))
        except:
            print(f'Error occured when trying to compare {param_name}: {sys.exc_info()}')
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'topo_comp-all_params' + (('-per_' + groupby) if groupby else '') + '.pdf'))
    print(f'Saving {out_fn}...')
    plt.gcf().savefig(out_fn)
    if show_fig:
        plt.show()
    else:
        plt.close("all")


""" Plot comparison of over-/underexpression of triad motifs from a single sample per circuit as specified (e.g., spec='Radius/All/0') """
def plot_triads_comparison(topocomp_config, topocomp_config_files, spec, out_dir, plot_single_samples=True, show_fig=False, fig_format='png', fig_dpi=600):
    
    circuit_ids = sorted(topocomp_config['circuits'].keys())
    circuit_names = [topocomp_config['circuits'][cidx]['circuit_name'] for cidx in circuit_ids]
    out_dir = os.path.join(out_dir, '_vs_'.join(circuit_names))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cfg_dict = dict([(k, Config(v)) for k, v in topocomp_config_files.items()])
    
    # Load triad overexpressions of selected sample
    spec_split = spec.split('/')
    sample_type, sample_name = spec_split[:2]
    if len(spec_split) == 3:
        sample_nr = spec_split[2].split(',')
    else:
        sample_nr = []
    triads = [] # Actual triad counts
    triads_er = [] # ER control model counts
    num_samples = []
    for cidx in circuit_ids:
        triad_fn = cfg_dict[cidx].stage('count_triads')['outputs']['triads']
        with open(triad_fn, 'r') as f:
            triads_tmp = json.load(f)[sample_type][sample_name]
        if len(sample_nr) == 0:
            sample_nr = list(triads_tmp.keys())
        else:
            assert np.all(np.isin(sample_nr, list(triads_tmp.keys()))), 'ERROR: Sample number error!'
        triads.append(np.array([triads_tmp[n]['overexpression'][0] for n in sample_nr]))
        triads_er.append(np.array([triads_tmp[n]['overexpression'][1] for n in sample_nr]))
        num_samples.append(len(sample_nr))
    
    if len(np.unique(num_samples)) == 1:
        num_samples_str = str(num_samples[0])
    else:
        num_samples_str = '/'.join([str(n) for n in num_samples])
    
    # Rel. triad counts w.r.t. ER control model
    triads_rel = [(triads[idx] - triads_er[idx]) / (triads[idx] + triads_er[idx]) for idx in range(len(circuit_ids))] # Normalize such that overexpression is constrained 0 < x <= 1 and underexpression to -1 <= x < 0
    #triads_rel = [(triads[idx] - triads_er[idx]) / triads_er[idx] for idx in range(len(circuit_ids))]
    
    # Plot rel. triad overexpressions (normalized)
    markers = ['s', 'o']
    colors = ['b', 'r']
    minor_colors = ['lightsteelblue', 'lightcoral']
    plt.figure()
    if plot_single_samples:
        for idx in range(len(circuit_ids)):
            plt.plot(range(1, triads_rel[idx].shape[1] + 1), triads_rel[idx].T, color=minor_colors[np.mod(idx, len(minor_colors))], linewidth=1, alpha=0.25, zorder=0)
    for idx in range(len(circuit_ids)):
        plt.errorbar(range(1, triads_rel[idx].shape[1] + 1), np.mean(triads_rel[idx], 0), yerr=np.std(triads_rel[idx], 0), color=colors[np.mod(idx, len(colors))], linewidth=3, elinewidth=2, marker=markers[np.mod(idx, len(markers))], label=circuit_names[idx])
    plt.xlim(plt.xlim())
    plt.plot(plt.xlim(), [0, 0], 'k', linewidth=0.5, zorder=0)
    plt.xticks(range(1, triads_rel[0].shape[1] + 1))
    plt.title(f'Triad motifs ({spec}; {num_samples_str} sample(s))')
    plt.xlabel('Triad motif')
    plt.ylabel('Norm. rel. count (w.r.t. ER model)')
    plt.legend(loc='upper left', ncol=len(circuit_ids))
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'triad_comp-' + spec.replace('/', '_').replace(',', '_') + f'.{fig_format}'))
    print(f'Saving {out_fn}...')
    plt.gcf().savefig(out_fn, dpi=fig_dpi)
    if show_fig:
        plt.show()
    else:
        plt.close("all")
