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
import itertools
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit

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
    elif order == 2:
        return extract_2nd_order(nodes, edges, node_ids_sel, node_ids_sel, **kwargs)
    elif order == 3:
        return extract_3rd_order(nodes, edges, node_ids_sel, node_ids_sel, **kwargs)
    else:
        assert False, f'ERROR: Order-{order} data extraction not supported!'


""" Build connection probability model from data """
def build(order, **kwargs):
    
    print(f'INFO: Running order-{order} model building...')
    
    if order == 1:
        return build_1st_order(**kwargs)
    elif order == 2:
        return build_2nd_order(**kwargs)
    elif order == 3:
        return build_3rd_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} model building not supported!'


""" Visualize data vs. model """
def plot(order, **kwargs):
    
    print(f'INFO: Running order-{order} data/model visualization...')
    
    if order == 1:
        return plot_1st_order(**kwargs)
    elif order == 2:
        return plot_2nd_order(**kwargs)
    elif order == 3:
        return plot_3rd_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} data/model visualization not supported!'


""" Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins """
def extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, dep_matrices, dep_bins):
    
    num_dep = len(dep_matrices)
    assert len(dep_bins) == num_dep, 'ERROR: Dependencies/bins mismatch!'
    assert np.all([dep_matrices[dim].shape == (len(src_node_ids), len(tgt_node_ids)) for dim in range(num_dep)]), 'ERROR: Matrix dimension mismatch!'
    
    # Extract adjacency
    conns = np.array(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids)))
    adj_mat = csr_matrix((np.full(conns.shape[0], True), conns.T.tolist()), shape=(max(src_node_ids) + 1, max(tgt_node_ids) + 1))
    if np.any(adj_mat.diagonal()):
        print('WARNING: Autaptic connection(s) found!')
    
    # Extract connection probability
    num_bins = [len(b) - 1 for b in dep_bins]
    bin_indices = [list(range(n)) for n in num_bins]
    count_all = np.full(num_bins, -1) # Count of all pairs of neurons for each combination of dependencies
    count_conn = np.full(num_bins, -1) # Count of connected pairs of neurons for each combination of dependencies
    
    print(f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...', flush=True)
    pbar = progressbar.ProgressBar(maxval=np.prod(num_bins) - 1)
    for idx in pbar(itertools.product(*bin_indices)):
        dep_sel = np.full((len(src_node_ids), len(tgt_node_ids)), True)
        for dim in range(num_dep):
            lower = dep_bins[dim][idx[dim]]
            upper = dep_bins[dim][idx[dim] + 1]
            dep_sel = np.logical_and(dep_sel, np.logical_and(dep_matrices[dim] >= lower, (dep_matrices[dim] < upper) if idx[dim] < num_bins[dim] - 1 else (dep_matrices[dim] <= upper))) # Including last edge
        sidx, tidx = np.nonzero(dep_sel)
        count_all[idx] = np.sum(dep_sel)
        count_conn[idx] = np.sum(adj_mat[src_node_ids[sidx], tgt_node_ids[tidx]])
    p_conn = count_conn / count_all
    
    return p_conn, count_conn, count_all


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   1st order model (Erdos-Renyi)
###################################################################################################

""" Extract average connection probability (1st order) from a sample of pairs of neurons """
def extract_1st_order(nodes, edges, src_node_ids, tgt_node_ids, **_):
    
    p_conn, conn_count, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [], [])
    
    src_cell_count = len(src_node_ids)
    tgt_cell_count = len(tgt_node_ids)
    print(f'INFO: Found {conn_count} connections between {src_cell_count}x{tgt_cell_count} neurons (p = {p_conn:.3f})')
    
    return {'p_conn': p_conn, 'src_cell_count': src_cell_count, 'tgt_cell_count': tgt_cell_count}


""" Build 1st order model (Erdos-Renyi, capturing average conn. prob.) """
def build_1st_order(p_conn, **_):
    
    p_conn_model = p_conn # Constant model

    print(f'MODEL FIT: p_conn_model()  = {p_conn_model:.3f}')
    
    return {'model': 'p',
            'model_inputs': [],
            'model_params': {'p': p_conn_model}}


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


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent)
###################################################################################################

""" Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons """
def extract_2nd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, **_):
    
    # Compute distance matrix
    src_nrn_pos = nodes.positions(src_node_ids).to_numpy()
    tgt_nrn_pos = nodes.positions(tgt_node_ids).to_numpy()
    dist_mat = distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections
    
    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um
    
    p_conn_dist, dist_count_conn, dist_count_all = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat], [dist_bins])
    
    return {'p_conn_dist': p_conn_dist, 'dist_bins': dist_bins, 'dist_count_conn': dist_count_conn, 'dist_count_all': dist_count_all, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


""" Build 2nd order model (exponential distance-dependent conn. prob.) """
def build_2nd_order(p_conn_dist, dist_bins, **_):
    
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    
    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    X = dist_bins[:-1][np.isfinite(p_conn_dist)] + bin_offset
    y = p_conn_dist[np.isfinite(p_conn_dist)]
    (a_opt, b_opt), _ = curve_fit(exp_model, X, y, p0=[0.0, 0.0])
    
    print(f'MODEL FIT: f(x) = {a_opt:.3f} * exp(-{b_opt:.3f} * x)')
    
    return {'model': 'a_opt * np.exp(-b_opt * np.array(d))',
            'model_inputs': ['d'],
            'model_params': {'a_opt': a_opt, 'b_opt': b_opt}}


""" Visualize data vs. model (2nd order) """
def plot_2nd_order(out_dir, p_conn_dist, dist_bins, src_cell_count, tgt_cell_count, model, model_inputs, model_params, **_):
    
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)
    
    model_str = f'f(x) = {model_params["a_opt"]:.3f} * exp(-{model_params["b_opt"]:.3f} * x)'
    model_fct = model_building.get_model(model, model_inputs, model_params)
    
    plt.figure(figsize=(12, 4), dpi=300)
    
    # Data vs. model
    plt.subplot(1, 2, 1)
    plt.plot(dist_bins[:-1] + bin_offset, p_conn_dist, '.-', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot(dist_model, model_fct(dist_model), '--', label='Model: ' + model_str)
    plt.grid()
    plt.xlabel('Distance [um]')
    plt.ylabel('Conn. prob.')
    plt.title(f'Dist-dep. conn. prob. (2nd order)')
    plt.legend()
    
    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500 # (um)
    r_markers = [200, 400] # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model_fct(vdist)
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=plt.cm.hot)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\Delta$x')
    plt.ylabel('$\Delta$z')
    plt.title('2D conn. prob. (2nd order model)')
    plt.colorbar(label='Conn. prob.')
    
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    return


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   3rd order (bipolar distance-dependent)
###################################################################################################

""" Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons """
def extract_3rd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, **_):
    
    # Compute distance matrix
    src_nrn_pos = nodes.positions(src_node_ids).to_numpy()
    tgt_nrn_pos = nodes.positions(tgt_node_ids).to_numpy()
    dist_mat = distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections
    
    # Compute bipolar matrix (along z-axis; post-synaptic neuron below (delta_z < 0) or above (delta_z > 0) pre-synaptic neuron)
    bip_mat = np.sign(np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 2], tgt_nrn_pos[:, 2], indexing='ij'), axis=0))) # Bipolar distinction based on difference in z coordinate
    
    # Extract bipolar distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
    bip_bins = [np.min(bip_mat), 0, np.max(bip_mat)]
    
    p_conn_dist_bip, dist_bip_count_conn, dist_bip_count_all = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat, bip_mat], [dist_bins, bip_bins])
    
    return {'p_conn_dist_bip': p_conn_dist_bip, 'dist_bins': dist_bins, 'bip_bins': bip_bins, 'dist_bip_count_conn': dist_bip_count_conn, 'dist_bip_count_all': dist_bip_count_all, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


""" Build 3rd order model (bipolar exp. distance-dependent conn. prob.) """
def build_3rd_order(p_conn_dist_bip, dist_bins, bip_bins, **_):
    
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    
    X = dist_bins[:-1][np.all(np.isfinite(p_conn_dist_bip), 1)] + bin_offset
    y = p_conn_dist_bip[np.all(np.isfinite(p_conn_dist_bip), 1), :]
    
    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    (aN_opt, bN_opt), _ = curve_fit(exp_model, X, y[:, 0], p0=[0.0, 0.0])
    (aP_opt, bP_opt), _ = curve_fit(exp_model, X, y[:, 1], p0=[0.0, 0.0])
    opt_model = lambda d, dz: np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], [aN_opt * np.exp(-bN_opt * np.array(d)), aP_opt * np.exp(-bP_opt * np.array(d)), 0.5 * (aN_opt * np.exp(-bN_opt * np.array(d)) + aP_opt * np.exp(-bP_opt * np.array(d)))])
    
    print(f'BIPOLAR MODEL FIT: f(x, dz) = {aN_opt:.3f} * exp(-{bN_opt:.3f} * x) if dz < 0')
    print(f'                              {aP_opt:.3f} * exp(-{bP_opt:.3f} * x) if dz > 0')
    print(f'                              AVERAGE OF BOTH MODELS  if dz == 0')
    
    return {'model': 'np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], [aN_opt * np.exp(-bN_opt * np.array(d)), aP_opt * np.exp(-bP_opt * np.array(d)), 0.5 * (aN_opt * np.exp(-bN_opt * np.array(d)) + aP_opt * np.exp(-bP_opt * np.array(d)))])',
            'model_inputs': ['d, dz'],
            'model_params': {'aN_opt': aN_opt, 'bN_opt': bN_opt, 'aP_opt': aP_opt, 'bP_opt': bP_opt}}


""" Visualize data vs. model (3rd order) """
def plot_3rd_order(out_dir, p_conn_dist_bip, dist_bins, bip_bins, src_cell_count, tgt_cell_count, model, model_inputs, model_params, **_):
    
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)
    
    model_strN = f'{model_params["aN_opt"]:.3f} * exp(-{model_params["bN_opt"]:.3f} * x)'
    model_strP = f'{model_params["aP_opt"]:.3f} * exp(-{model_params["bP_opt"]:.3f} * x)'
    model_fct = model_building.get_model(model, model_inputs, model_params)
    
    plt.figure(figsize=(12, 4), dpi=300)
    
    # Data vs. model
    plt.subplot(1, 2, 1)
    bip_dist = np.concatenate((-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset))
    bip_data = np.concatenate((p_conn_dist_bip[::-1, 0], [np.nan], p_conn_dist_bip[:, 1]))
    plt.plot(bip_dist, bip_data, '.-', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot(-dist_model, model_fct(dist_model, np.sign(-dist_model)), '--', label='Model: ' + model_strN)
    plt.plot(dist_model, model_fct(dist_model, np.sign(dist_model)), '--', label='Model: ' + model_strP)
    plt.grid()
    plt.xlabel('sign($\Delta$z) * Distance [um]')
    plt.ylabel('Conn. prob.')
    plt.title(f'Bipolar dist-dep. conn. prob. (3rd order)')
    plt.legend(loc='upper left')
    
    # 2D connection probability (model)
    plt.subplot(1, 2, 2)
    plot_range = 500 # (um)
    r_markers = [200, 400] # (um)
    dx = np.linspace(-plot_range, plot_range, 201)
    dz = np.linspace(plot_range, -plot_range, 201)
    xv, zv = np.meshgrid(dx, dz)
    vdist = np.sqrt(xv**2 + zv**2)
    pdist = model_fct(vdist, np.sign(zv))
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=plt.cm.hot)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\Delta$x')
    plt.ylabel('$\Delta$z')
    plt.title('2D conn. prob. (3rd order model)')
    plt.colorbar(label='Conn. prob.')
    
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    return
