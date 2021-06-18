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
def plot_triads_comparison(topocomp_config, topocomp_config_files, spec, out_dir, show_fig=False):
    
    circuit_ids = sorted(topocomp_config['circuits'].keys())
    circuit_names = [topocomp_config['circuits'][cidx]['circuit_name'] for cidx in circuit_ids]
    out_dir = os.path.join(out_dir, '_vs_'.join(circuit_names))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cfg_dict = dict([(k, Config(v)) for k, v in topocomp_config_files.items()])
    
    # Load triad overexpressions of selected sample
    sample_type, sample_name, sample_nr = spec.split('/')
    triads = []
    for cidx in circuit_ids:
        triad_fn = cfg_dict[cidx].stage('count_triads')['outputs']['triads']
        with open(triad_fn, 'r') as f:
            triads.append(np.array(json.load(f)[sample_type][sample_name][sample_nr]['overexpression']))
    
    # Rel. triad counts w.r.t. ER control model
    triads_rel = [(triads[idx][0, :] - triads[idx][1, :]) / triads[idx][1, :] for idx in range(len(circuit_ids))]
    
    # Plot rel. triad overexpressions (normalized)
    markers = ['s', 'o']
    plt.figure()
    for idx in range(len(circuit_ids)):
        plt.plot(range(1, len(triads_rel[0]) + 1), triads_rel[idx] / np.sum(triads_rel[idx]), marker=markers[np.mod(idx, len(markers))], label=circuit_names[idx])
    plt.xticks(range(1, len(triads_rel[0]) + 1))
    plt.title('Over-/underexpression of triad motifs (w.r.t. ER model)')
    plt.xlabel('Triad motif')
    plt.ylabel('Norm. rel. count')
    plt.legend()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'triad_comp-' + spec.replace('/', '_') + '.pdf'))
    print(f'Saving {out_fn}...')
    plt.gcf().savefig(out_fn)
    if show_fig:
        plt.show()
    else:
        plt.close("all")
