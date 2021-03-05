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
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix

""" Extract connection probability from a sample of pairs of neurons """
def extract(circuit, order, sample_size=None, **kwargs):
    
    #TODO: Add cell selection criteria (layers, mtypes, ...)
    
    print(f'INFO: Running order-{order} data extraction...')
    
    nodes = circuit.nodes['All']
    node_ids = nodes.ids()
    if sample_size is None or sample_size <= 0 or sample_size >= len(node_ids):
        sample_size = len(node_ids)
    node_ids_sel = node_ids[np.random.permutation([True] * sample_size + [False] * (len(node_ids) - sample_size))]
    
    edges = circuit.edges['default']
    
    if order == 1:
        return extract_1st_order(nodes, edges, node_ids_sel, node_ids_sel, **kwargs)
    else:
        assert False, f'ERROR: Order-{order} data extraction not supported!'


""" Build connection probability model from data """
def build(order, **kwargs):
    
    print(f'INFO: Running order-{order} model building...')
    
    if order == 1:
        return build_1st_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} model building not supported!'


""" Visualize data vs. model """
def plot(order, **kwargs):
    
    print(f'INFO: Running order-{order} data/model visualization...')
    
    if order == 1:
        return plot_1st_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} data/model visualization not supported!'


# Generative models for circuit connectivity from [Gal et al. 2020]
# *** 1st order model (Erdos-Renyi) ***
""" Extract average connection probability (1st order) from a sample of pairs of neurons """
def extract_1st_order(nodes, edges, src_node_ids, tgt_node_ids, **_):
    
    src_cell_count = len(src_node_ids)
    tgt_cell_count = len(tgt_node_ids)
    conn_count = len(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids)))
    
    print(f'INFO: Found {conn_count} connections between {src_cell_count}x{tgt_cell_count} neurons')
    
    p_conn = conn_count / (src_cell_count * tgt_cell_count)
    
    return {'p_conn': p_conn, 'src_cell_count': src_cell_count, 'tgt_cell_count': tgt_cell_count}


""" Build 1st order model (Erdos-Renyi, capturing average conn. prob.) """
def build_1st_order(p_conn, **_):
    
    p_conn_model = p_conn # Constant model

    print(f'MODEL FIT: p_conn_model()  = {p_conn_model:.3f}')
    
    return {'model': 'p', 'model_inputs': [], 'model_params': {'p': p_conn_model}}


""" Visualize data vs. model (1st order) """
def plot_1st_order(out_dir, p_conn, src_cell_count, tgt_cell_count, model, model_inputs, model_params, **_):
    
    model_str = f'f(x) = {model_params["p"]:.3f}'
    model_fct = model_building.get_model(model, model_inputs, model_params)
    
    # Draw figure
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(0.5, p_conn, width=1, facecolor='tab:blue', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot([-0.5, 1.5], np.ones(2) * model_fct(), '--', color='tab:red', label=f'Model: ' + model_str)
    plt.text(0.5, 0.99 * p_conn, f'p = {p_conn:.3f}', color='k', ha='center', va='top')
    plt.xticks([])
    plt.ylabel('Conn. prob.')
    plt.title(f'Average conn. prob. (1st-order)', fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    return

# *** 2nd order (distance-dependent) ***
""" Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons """
def extract_2nd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, **_):
    
    # Compute distance matrix
    src_nrn_pos = nodes.positions(src_node_ids).to_numpy()
    tgt_nrn_pos = nodes.positions(tgt_node_ids).to_numpy()
    dist_mat = distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections
    
    # Extract adjacency
    conns = np.array(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids)))
    adj_mat = csr_matrix((np.full(conns.shape[0], True), conns.T.tolist()))
    assert not np.any(adj_mat.diagonal()), 'ERROR: Autaptic connection(s) found!'
    
    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.max(dist_mat)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um
    dist_count_all = np.full(num_bins, -1) # Count of all pairs of neurons within given distance
    dist_count_conn = np.full(num_bins, -1) # Count of connected pairs of neurons withing given distance
    
    print('Extracting distance-dependent connection probabilities...', flush=True)
    pbar = progressbar.ProgressBar()
    for idx in pbar(range(num_bins)):
        d_sel = np.logical_and(dist_mat >= dist_bins[idx], (dist_mat < dist_bins[idx + 1]) if idx < num_bins - 1 else (dist_mat <= dist_bins[idx + 1])) # Including last edge
        sidx, tidx = np.where(d_sel)
        dist_count_all[idx] = np.sum(d_sel)
        dist_count_conn[idx] = np.sum(adj_mat[src_node_ids[sidx], tgt_node_ids[tidx]])
    p_conn_dist = dist_count_conn / dist_count_all
    
    return {'p_conn': p_conn_dist, 'dist_bins': dist_bins, 'dist_count_conn': dist_count_conn, 'dist_count_all': dist_count_all}


""" Build 2nd order model (exponential distance-dependent conn. prob.) """
def build_2nd_order(p_conn_dist, dist_bins, **_):
    return


""" Visualize data vs. model (2nd order) """
def plot_2nd_order(out_dir, p_conn_dist, p_conn_dist_model, dist_bins, **_):
    return
