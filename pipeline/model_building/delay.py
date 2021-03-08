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
from sklearn.linear_model import LinearRegression

""" Extract distance-dependent synaptic delays from a sample of pairs of neurons """
def extract(circuit, bin_size_um, max_range_um=None, sample_size=None, **_):
    
    # Extract distance/delay values
    nodes = circuit.nodes['All']
    node_ids = nodes.ids()
    if sample_size is None or sample_size <= 0 or sample_size >= len(node_ids):
        sample_size = len(node_ids)
    node_ids_sel = node_ids[np.random.permutation([True] * sample_size + [False] * (len(node_ids) - sample_size))]
    
    edges = circuit.edges['default']
    edges_table = edges.pathway_edges(source=node_ids_sel, target=node_ids_sel, properties=['@source_node', 'delay', 'afferent_center_x', 'afferent_center_y', 'afferent_center_z'])
    
    print(f'INFO: Extracting delays from {edges_table.shape[0]} synapses between {sample_size} neurons')
    
    src_pos = nodes.positions(edges_table['@source_node'].to_numpy()).to_numpy() # Soma position of pre-synaptic neuron
    tgt_pos = edges_table[['afferent_center_x', 'afferent_center_y', 'afferent_center_z']].to_numpy() # Synapse position on post-synaptic dendrite
    src_tgt_dist = np.sqrt(np.sum((tgt_pos - src_pos)**2, 1))
    src_tgt_delay = edges_table['delay'].to_numpy()
    
    # Extract distance-dependent delays
    if max_range_um is None:
        max_range_um = np.max(src_tgt_dist)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um
    dist_delays_mean = np.full(num_bins, np.nan)
    dist_delays_std = np.full(num_bins, np.nan)
    dist_count = np.zeros(num_bins).astype(int)

    print('Extracting distance-dependent synaptic delays...', flush=True)
    pbar = progressbar.ProgressBar()
    for idx in pbar(range(num_bins)):
        d_sel = np.logical_and(src_tgt_dist >= dist_bins[idx], (src_tgt_dist < dist_bins[idx + 1]) if idx < num_bins - 1 else (src_tgt_dist <= dist_bins[idx + 1])) # Including last edge
        dist_count[idx] = np.sum(d_sel)
        if dist_count[idx] > 0:
            dist_delays_mean[idx] = np.mean(src_tgt_delay[d_sel])
            dist_delays_std[idx] = np.std(src_tgt_delay[d_sel])
    
    return {'dist_bins': dist_bins, 'dist_delays_mean': dist_delays_mean, 'dist_delays_std': dist_delays_std, 'dist_count': dist_count}


""" Build distance-dependent synaptic delay model (linear model for delay mean, const model for delay std) """
def build(dist_bins, dist_delays_mean, dist_delays_std, bin_size_um, **_):
    
    assert np.all((np.diff(dist_bins) - bin_size_um) < 1e-12), 'ERROR: Bin size mismatch!'
    bin_offset = 0.5 * bin_size_um
    
    # Mean delay model (linear)
    X = np.array(dist_bins[:-1][np.isfinite(dist_delays_mean)] + bin_offset, ndmin=2).T
    y = dist_delays_mean[np.isfinite(dist_delays_mean)]
    dist_delays_mean_model = LinearRegression().fit(X, y)
    
    # Std delay model (const)
    dist_delays_std_model = np.mean(dist_delays_std)
    
    print(f'MODEL FIT: dist_delays_mean_model(x) = {dist_delays_mean_model.coef_[0]:.3f} * x + {dist_delays_mean_model.intercept_:.3f}')
    print(f'           dist_delays_std_model(x)  = {dist_delays_std_model:.3f}')
    
    return {'model': 'dist_delays_mean_model.predict(d) if type=="mean" else (np.full_like(d, dist_delays_std_model, dtype=np.double) if type=="std" else None)',
            'model_inputs': ['d', 'type'],
            'model_params': {'dist_delays_mean_model': dist_delays_mean_model, 'dist_delays_std_model': dist_delays_std_model}}


""" Visualize data vs. model """
def plot(out_dir, dist_bins, dist_delays_mean, dist_delays_std, dist_count, model, model_inputs, model_params, **_):
    
    bin_width = np.diff(dist_bins[:2])[0]
    
    mean_model_str = f'f(x) = {model_params["dist_delays_mean_model"].coef_[0]:.3f} * x + {model_params["dist_delays_mean_model"].intercept_:.3f}'
    std_model_str = f'f(x) = {model_params["dist_delays_std_model"]:.3f}'
    model_fct = model_building.get_model(model, model_inputs, model_params)
    
    # Draw figure
    plt.figure(figsize=(8, 4), dpi=300)
    plt.bar(dist_bins[:-1] + 0.5 * bin_width, dist_delays_mean, width=0.95 * bin_width, facecolor='tab:blue', label=f'Data mean: N = {np.sum(dist_count)} synapses')
    plt.bar(dist_bins[:-1] + 0.5 * bin_width, dist_delays_std, width=0.5 * bin_width, facecolor='tab:red', label=f'Data std: N = {np.sum(dist_count)} synapses')
    plt.plot(dist_bins, model_fct(np.array(dist_bins, ndmin=2).T, 'mean'), '--', color='tab:brown', label='Model mean: ' + mean_model_str)
    plt.plot(dist_bins, model_fct(np.array(dist_bins, ndmin=2).T, 'std'), '--', color='tab:olive', label='Model std: ' + std_model_str)
    plt.xlim((dist_bins[0], dist_bins[-1]))
    plt.xlabel('Distance [um]')
    plt.ylabel('Delay [ms]')
    plt.title(f'Distance-dependent synaptic delays', fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    
    # Add second axis with bin counts
    count_color = 'tab:orange'
    ax_count = plt.gca().twinx()
    ax_count.set_yscale('log')
    ax_count.step(dist_bins, np.concatenate((dist_count[:1], dist_count)), color=count_color)
    ax_count.set_ylabel('Count', color=count_color)
    ax_count.tick_params(axis='y', which='both', colors=count_color)
    ax_count.spines['right'].set_color(count_color)
    
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    return

