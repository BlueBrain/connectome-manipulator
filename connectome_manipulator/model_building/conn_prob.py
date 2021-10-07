'''TODO: improve description'''
# Model building function
#
# Three functions need to be defined
# (1) extract(...): extracting connectivity specific data
# (2) build(...): building a data-based model
# (3) plot(...): visualizing data vs. model

import itertools
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor

from connectome_manipulator.model_building import model_building


JET = plt.cm.get_cmap('jet')
HOT = plt.cm.get_cmap('hot')


def extract(circuit, order, sel_src=None, sel_dest=None, sample_size=None, **kwargs):
    """Extract connection probability between samples of neurons."""
    print(f'INFO: Running order-{order} data extraction (sel_src={sel_src}, sel_dest={sel_dest}, sample_size={sample_size} neurons)...')

    # Select edge population [assuming exactly one edge population in given edges file]
    assert len(circuit.edges.population_names) == 1, 'ERROR: Only a single edge population per file supported for modelling!'
    edges = circuit.edges[circuit.edges.population_names[0]]

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]

    node_ids_src = src_nodes.ids(sel_src)
    node_ids_dest = tgt_nodes.ids(sel_dest)

    if not kwargs.get('pos_map_file') is None:
        assert (src_nodes.name == tgt_nodes.name) or (src_nodes.ids().min() > tgt_nodes.ids().max()) or (src_nodes.ids().max() < tgt_nodes.ids().min()), 'ERROR: Position mapping only supported for same source/taget node population or non-overlapping id ranges!'

    if sample_size is None or sample_size <= 0:
        sample_size = np.inf # Select all nodes
    sample_size_src = min(sample_size, len(node_ids_src))
    sample_size_dest = min(sample_size, len(node_ids_dest))
    node_ids_src_sel = node_ids_src[np.random.permutation([True] * sample_size_src + [False] * (len(node_ids_src) - sample_size_src))]
    node_ids_dest_sel = node_ids_dest[np.random.permutation([True] * sample_size_dest + [False] * (len(node_ids_dest) - sample_size_dest))]

    if order == 1:
        return extract_1st_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == 2:
        return extract_2nd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == 3:
        return extract_3rd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == 4:
        return extract_4th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == 5:
        return extract_5th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    else:
        assert False, f'ERROR: Order-{order} data extraction not supported!'


def build(order, **kwargs):
    """Build connection probability model from data."""
    print(f'INFO: Running order-{order} model building...')

    if order == 1:
        return build_1st_order(**kwargs)
    elif order == 2:
        return build_2nd_order(**kwargs)
    elif order == 3:
        return build_3rd_order(**kwargs)
    elif order == 4:
        return build_4th_order(**kwargs)
    elif order == 5:
        return build_5th_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} model building not supported!'


def plot(order, **kwargs):
    """Visualize data vs. model."""
    print(f'INFO: Running order-{order} data/model visualization...')

    if order == 1:
        return plot_1st_order(**kwargs)
    elif order == 2:
        return plot_2nd_order(**kwargs)
    elif order == 3:
        return plot_3rd_order(**kwargs)
    elif order == 4:
        return plot_4th_order(**kwargs)
    elif order == 5:
        return plot_5th_order(**kwargs)
    else:
        assert False, f'ERROR: Order-{order} data/model visualization not supported!'


###################################################################################################
# Helper functions
###################################################################################################

def load_pos_mapping_model(pos_map_file):
    """Load a position mapping model from file."""
    if pos_map_file is None:
        pos_map = None
    else:
        assert os.path.exists(pos_map_file), 'Position mapping model file not found!'
        print(f'Loading position mapping model from {pos_map_file}')
        with open(pos_map_file, 'rb') as f:
            pos_map_dict = pickle.load(f)
        pos_map = model_building.get_model(pos_map_dict['model'], pos_map_dict['model_inputs'], pos_map_dict['model_params'])

    return pos_map


def get_neuron_positions(pos_fct, node_ids_list):
    """Get neuron positions (using position access/mapping function) [NOTE: node_ids_list should be list of node_ids lists!]."""
    if not isinstance(pos_fct, list):
        pos_fct = [pos_fct for i in node_ids_list]
    else:
        assert len(pos_fct) == len(node_ids_list), 'ERROR: "pos_fct" must be scalar or a list with same length as "node_ids_list"!'

    nrn_pos = [np.array(pos_fct[i](node_ids_list[i])) for i in range(len(node_ids_list))]

    return nrn_pos


def compute_dist_matrix(src_nrn_pos, tgt_nrn_pos):
    """Computes distance matrix between pairs of neurons."""
    dist_mat = distance_matrix(src_nrn_pos, tgt_nrn_pos)
    dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

    return dist_mat


def compute_bip_matrix(src_nrn_pos, tgt_nrn_pos):
    """Computes bipolar matrix between pairs of neurons (along z-axis; POST-synaptic neuron below (delta_z < 0) or above (delta_z > 0) PRE-synaptic neuron)."""
    bip_mat = np.sign(np.diff(np.meshgrid(src_nrn_pos[:, 2], tgt_nrn_pos[:, 2], indexing='ij'), axis=0)[0, :, :]) # Bipolar distinction based on difference in z coordinate

    return bip_mat


def compute_offset_matrices(src_nrn_pos, tgt_nrn_pos):
    """Computes dx/dy/dz offset matrices between pairs of neurons (POST minus PRE position)."""
    dx_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 0], tgt_nrn_pos[:, 0], indexing='ij'), axis=0)) # Relative difference in x coordinate
    dy_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 1], tgt_nrn_pos[:, 1], indexing='ij'), axis=0)) # Relative difference in y coordinate
    dz_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 2], tgt_nrn_pos[:, 2], indexing='ij'), axis=0)) # Relative difference in z coordinate

    return dx_mat, dy_mat, dz_mat


def compute_position_matrices(src_nrn_pos, tgt_nrn_pos):
    """Computes x/y/z position matrices (PRE neuron positions repeated over POST neuron number)."""
    x_mat, y_mat, z_mat = [np.tile(src_nrn_pos[:, i:i + 1], [1, tgt_nrn_pos.shape[0]]) for i in range(src_nrn_pos.shape[1])]

    return x_mat, y_mat, z_mat


def extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, dep_matrices, dep_bins):
    """Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins."""
    num_dep = len(dep_matrices)
    assert len(dep_bins) == num_dep, 'ERROR: Dependencies/bins mismatch!'
    assert np.all([dep_matrices[dim].shape == (len(src_node_ids), len(tgt_node_ids)) for dim in range(num_dep)]), 'ERROR: Matrix dimension mismatch!'

    # Extract adjacency
    conns = np.array(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids)))
    if len(conns) > 0:
        adj_mat = csr_matrix((np.full(conns.shape[0], True), conns.T.tolist()), shape=(max(src_node_ids) + 1, max(tgt_node_ids) + 1))
    else:
        adj_mat = csr_matrix((max(src_node_ids) + 1, max(tgt_node_ids) + 1)) # Empty matrix
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
    p_conn = np.array(count_conn / count_all)
    p_conn[np.isnan(p_conn)] = 0.0

    return p_conn, count_conn, count_all


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   1st order model (Erdos-Renyi)
###################################################################################################

def extract_1st_order(_nodes, edges, src_node_ids, tgt_node_ids, **_):
    """Extract average connection probability (1st order) from a sample of pairs of neurons."""
    p_conn, conn_count, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [], [])

    src_cell_count = len(src_node_ids)
    tgt_cell_count = len(tgt_node_ids)
    print(f'INFO: Found {conn_count} connections between {src_cell_count}x{tgt_cell_count} neurons (p = {p_conn:.3f})')

    return {'p_conn': p_conn, 'src_cell_count': src_cell_count, 'tgt_cell_count': tgt_cell_count}


def build_1st_order(p_conn, **_):
    """Build 1st order model (Erdos-Renyi, capturing average conn. prob.)."""
    p_conn_model = p_conn # Constant model

    print(f'MODEL FIT: p_conn_model()  = {p_conn_model:.3f}')

    return {'model': 'p',
            'model_inputs': [],
            'model_params': {'p': p_conn_model}}


def plot_1st_order(out_dir, p_conn, src_cell_count, tgt_cell_count, model, model_inputs, model_params, **_):  # pragma: no cover
    """Visualize data vs. model (1st order)."""
    model_str = f'f(x) = {model_params["p"]:.3f}'
    model_fct = model_building.get_model(model, model_inputs, model_params)

    # Draw figure
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(0.5, p_conn, width=1, facecolor='tab:blue', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot([-0.5, 1.5], np.ones(2) * model_fct(), '--', color='tab:red', label=f'Model: {model_str}')
    plt.text(0.5, 0.99 * p_conn, f'p = {p_conn:.3f}', color='k', ha='center', va='top')
    plt.xticks([])
    plt.ylabel('Conn. prob.')
    plt.title('Average conn. prob. (1st-order)', fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################

def extract_2nd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    pos_map = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_map is None else pos_map, [src_node_ids, tgt_node_ids])

    # Compute distance matrix
    dist_mat = compute_dist_matrix(src_nrn_pos, tgt_nrn_pos)

    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um

    p_conn_dist, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat], [dist_bins])

    return {'p_conn_dist': p_conn_dist, 'dist_bins': dist_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_2nd_order(p_conn_dist, dist_bins, **_):
    """Build 2nd order model (exponential distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    X = dist_bins[:-1][np.isfinite(p_conn_dist)] + bin_offset
    y = p_conn_dist[np.isfinite(p_conn_dist)]
    (a_opt, b_opt), _ = curve_fit(exp_model, X, y, p0=[0.0, 0.0])

    print(f'MODEL FIT: f(x) = {a_opt:.3f} * exp(-{b_opt:.3f} * x)')

    return {'model': 'a_opt * np.exp(-b_opt * np.array(d))',
            'model_inputs': ['d'],
            'model_params': {'a_opt': a_opt, 'b_opt': b_opt}}


def plot_2nd_order(out_dir, p_conn_dist, dist_bins, src_cell_count, tgt_cell_count, model, model_inputs, model_params, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (2nd order)."""
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
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Conn. prob.')
    plt.title('Data vs. model fit')
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
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=HOT, vmin=0.0)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\\Delta$x')
    plt.ylabel('$\\Delta$z')
    plt.title('2D model')
    plt.colorbar(label='Conn. prob.')

    plt.suptitle(f'Distance-dependent connection probability model (2nd order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   3rd order (bipolar distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################

def extract_3rd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, **_):
    """Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    pos_map = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_map is None else pos_map, [src_node_ids, tgt_node_ids])

    # Compute distance matrix
    dist_mat = compute_dist_matrix(src_nrn_pos, tgt_nrn_pos)

    # Compute bipolar matrix (along z-axis; post-synaptic neuron below (delta_z < 0) or above (delta_z > 0) pre-synaptic neuron)
    bip_mat = compute_bip_matrix(src_nrn_pos, tgt_nrn_pos)

    # Extract bipolar distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
    bip_bins = [np.min(bip_mat), 0, np.max(bip_mat)]

    p_conn_dist_bip, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat, bip_mat], [dist_bins, bip_bins])

    return {'p_conn_dist_bip': p_conn_dist_bip, 'dist_bins': dist_bins, 'bip_bins': bip_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_3rd_order(p_conn_dist_bip, dist_bins, _bip_bins, **_):
    """Build 3rd order model (bipolar exp. distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    X = dist_bins[:-1][np.all(np.isfinite(p_conn_dist_bip), 1)] + bin_offset
    y = p_conn_dist_bip[np.all(np.isfinite(p_conn_dist_bip), 1), :]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    (aN_opt, bN_opt), _ = curve_fit(exp_model, X, y[:, 0], p0=[0.0, 0.0])
    (aP_opt, bP_opt), _ = curve_fit(exp_model, X, y[:, 1], p0=[0.0, 0.0])
    # TODO(gevaert): if this is needed, use function
    # opt_model = lambda d, dz: np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], [aN_opt * np.exp(-bN_opt * np.array(d)), aP_opt * np.exp(-bP_opt * np.array(d)), 0.5 * (aN_opt * np.exp(-bN_opt * np.array(d)) + aP_opt * np.exp(-bP_opt * np.array(d)))])

    print(f'BIPOLAR MODEL FIT: f(x, dz) = {aN_opt:.3f} * exp(-{bN_opt:.3f} * x) if dz < 0')
    print(f'                              {aP_opt:.3f} * exp(-{bP_opt:.3f} * x) if dz > 0')
    print('                              AVERAGE OF BOTH MODELS  if dz == 0')

    return {'model': 'np.select([np.array(dz) < 0, np.array(dz) > 0, np.array(dz) == 0], [aN_opt * np.exp(-bN_opt * np.array(d)), aP_opt * np.exp(-bP_opt * np.array(d)), 0.5 * (aN_opt * np.exp(-bN_opt * np.array(d)) + aP_opt * np.exp(-bP_opt * np.array(d)))])',
            'model_inputs': ['d', 'dz'],
            'model_params': {'aN_opt': aN_opt, 'bN_opt': bN_opt, 'aP_opt': aP_opt, 'bP_opt': bP_opt}}


def plot_3rd_order(out_dir, p_conn_dist_bip, dist_bins, _bip_bins, src_cell_count, tgt_cell_count, model, model_inputs, model_params, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (3rd order)."""
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
    plt.xlabel('sign($\\Delta$z) * Distance [$\\mu$m]')
    plt.ylabel('Conn. prob.')
    plt.title('Data vs. model fit')
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
    plt.imshow(pdist, interpolation='bilinear', extent=(-plot_range, plot_range, -plot_range, plot_range), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    for r in r_markers:
        plt.gca().add_patch(plt.Circle((0, 0), r, edgecolor='w', linestyle='--', fill=False))
        plt.text(0, r, f'{r} $\\mu$m', color='w', ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$\\Delta$x')
    plt.ylabel('$\\Delta$z')
    plt.title('2D model')
    plt.colorbar(label='Conn. prob.')

    plt.suptitle(f'Bipolar distance-dependent connection probability model (3rd order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   4th order (offset-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_4th_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, **_):
    """Extract offset-dependent connection probability (4th order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    pos_map = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_map is None else pos_map, [src_node_ids, tgt_node_ids])

    # Compute dx/dy/dz offset matrices
    dx_mat, dy_mat, dz_mat = compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract offset-dependent connection probabilities
    if max_range_um is None:
        dx_range, dy_range, dz_range = zip([np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)], [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)])
    elif np.isscalar(max_range_um): # Single scalar range value to be used for all dimensions
        assert max_range_um > 0.0, 'ERROR: Maximum range must be larger than 0um!'
        dx_range = dy_range = dz_range = [-max_range_um, max_range_um]
    else: # Three values for x/y/z dimensions
        assert len(max_range_um) == 3, 'ERROR: Maximum range in x/y/z dimension expected!'
        assert np.all([r > 0.0 for r in max_range_um]), 'ERROR: Maximum range must be larger than 0um!'
        dx_range, dy_range, dz_range = zip([-r for r in max_range_um], max_range_um)

    if np.isscalar(bin_size_um): # Single scalar range value to be used for all dimensions
        assert bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!'
        bin_size_dx = bin_size_dy = bin_size_dz = bin_size_um
    else: # Three values for x/y/z dimensions
        assert len(bin_size_um) == 3, 'ERROR: Offset bin sizes in x/y/z dimension expected!'
        assert np.all([b > 0.0 for b in bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!'
        bin_size_dx, bin_size_dy, bin_size_dz = bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_offset, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dx_mat, dy_mat, dz_mat], [dx_bins, dy_bins, dz_bins])

    return {'p_conn_offset': p_conn_offset, 'dx_bins': dx_bins, 'dy_bins': dy_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_4th_order(p_conn_offset, dx_bins, dy_bins, dz_bins, model_specs=None, **_):
    """Build 4th order model (linear interpolation or random forest regression model for offset-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    dx_pos = dx_bins[:-1] + dx_bin_offset # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    model_inputs = ['dx', 'dy', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation':

        # Linear interpolation model => Removing dimensions with only single value from interpolation
        assert len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!'

        model_dict = {'model': 'np.maximum(interp_fct([pos for pos, sel in zip([dx_pos, dy_pos, dz_pos], np.array(p_conn_offset.shape) > 1) if sel], np.squeeze(p_conn_offset), np.array([pos for pos, sel in zip([np.array(dx), np.array(dy), np.array(dz)], np.array(p_conn_offset.shape) > 1) if sel]).T, method="linear", bounds_error=False, fill_value=None), 0)',
                      'model_inputs': model_inputs,
                      'model_params': {'interp_fct': scipy.interpolate.interpn, 'dx_pos': dx_pos, 'dy_pos': dy_pos, 'dz_pos': dz_pos, 'p_conn_offset': p_conn_offset}}

    elif model_specs.get('name') == 'RandomForestRegressor':

        # Random Forest Regressor model
        dxv, dyv, dzv = np.meshgrid(dx_pos, dy_pos, dz_pos, indexing='ij')
        data_pos = np.array([dxv.flatten(), dyv.flatten(), dzv.flatten()]).T
        data_val = p_conn_offset.flatten()

        offset_regr_model = RandomForestRegressor(random_state=0, **model_specs.get('kwargs', {}))
        offset_regr_model.fit(data_pos, data_val)

        model_dict = {'model': 'np.maximum(offset_regr_model.predict(np.array([np.array(dx), np.array(dy), np.array(dz)]).T), 0)',
                      'model_inputs': model_inputs,
                      'model_params': {'offset_regr_model': offset_regr_model}}

    else:
        assert False, f'ERROR: Model type "{model_specs.get("name")}" unknown!'

    print(f'OFFSET MODEL: f(dx, dy, dz) ~ {model_specs.get("name")} {model_specs.get("kwargs", {})}')

    return model_dict


def plot_4th_order(out_dir, p_conn_offset, dx_bins, dy_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, model_inputs, model_params, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (4th order)."""
    model_fct = model_building.get_model(model, model_inputs, model_params)

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    model_ovsampl = 4 # Model oversampling factor (per dimension)
    dx_pos_model = np.linspace(dx_bins[0], dx_bins[-1], (len(dx_bins) - 1) * model_ovsampl + 1)
    dy_pos_model = np.linspace(dy_bins[0], dy_bins[-1], (len(dy_bins) - 1) * model_ovsampl + 1)
    dz_pos_model = np.linspace(dz_bins[0], dz_bins[-1], (len(dz_bins) - 1) * model_ovsampl + 1)
    dxv, dyv, dzv = np.meshgrid(dx_pos_model, dy_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([dxv.flatten(), dyv.flatten(), dzv.flatten()]).T # Regular grid
    # model_pos = np.random.uniform(low=[dx_bins[0], dy_bins[0], dz_bins[0]], high=[dx_bins[-1], dy_bins[-1], dz_bins[-1]], size=[model_ovsampl**3 * len(dx_bins) * len(dy_bins) * len(dz_bins), 3]) # Random sampling
    model_val = model_fct(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2])
    model_val_xyz = model_val.reshape([len(dx_pos_model), len(dy_pos_model), len(dz_pos_model)])

    # 3D connection probability (data vs. model)
    num_p_bins = 100
    p_bins = np.linspace(0, max(np.max(p_conn_offset), np.max(model_val)), num_p_bins + 1)
    p_color_map = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(vmin=p_bins[0], vmax=p_bins[-1]))
    p_colors = p_color_map.to_rgba(np.linspace(p_bins[0], p_bins[-1], num_p_bins))

    fig = plt.figure(figsize=(16, 6), dpi=300)
    # (Data)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for pidx in range(num_p_bins):
        p_sel_idx = np.where(np.logical_and(p_conn_offset > p_bins[pidx], p_conn_offset <= p_bins[pidx + 1]))
        plt.plot(dx_bins[p_sel_idx[0]] + dx_bin_offset, dy_bins[p_sel_idx[1]] + dy_bin_offset, dz_bins[p_sel_idx[2]] + dz_bin_offset, 'o', color=p_colors[pidx, :], alpha=0.1 + 0.9 * (pidx + 1) / num_p_bins, markeredgecolor='none')
    ax.view_init(30, 60)
    ax.set_xlim((dx_bins[0], dx_bins[-1]))
    ax.set_ylim((dy_bins[0], dy_bins[-1]))
    ax.set_zlim((dz_bins[0], dz_bins[-1]))
    ax.set_xlabel('$\\Delta$x [$\\mu$m]')
    ax.set_ylabel('$\\Delta$y [$\\mu$m]')
    ax.set_zlabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(p_color_map, label='Conn. prob.')
    plt.title(f'Data: N = {src_cell_count}x{tgt_cell_count} cells')

    # (Model)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for pidx in range(num_p_bins):
        p_sel_idx = np.logical_and(model_val > p_bins[pidx], model_val <= p_bins[pidx + 1])
        plt.plot(model_pos[p_sel_idx, 0], model_pos[p_sel_idx, 1], model_pos[p_sel_idx, 2], '.', color=p_colors[pidx, :], alpha=0.1 + 0.9 * (pidx + 1) / num_p_bins, markeredgecolor='none')
    ax.view_init(30, 60)
    ax.set_xlim((dx_bins[0], dx_bins[-1]))
    ax.set_ylim((dy_bins[0], dy_bins[-1]))
    ax.set_zlim((dz_bins[0], dz_bins[-1]))
    ax.set_xlabel('$\\Delta$x [$\\mu$m]')
    ax.set_ylabel('$\\Delta$y [$\\mu$m]')
    ax.set_zlabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(p_color_map, label='Conn. prob.')
    plt.title(f'Model: {model_specs.get("name")}')

    plt.suptitle(f'Offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_3d.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)

    # Max. intensity projection (data vs. model)
    plt.figure(figsize=(12, 6), dpi=300)
    # (Data)
    plt.subplot(2, 3, 1)
    plt.imshow(np.max(p_conn_offset, 1).T, interpolation='none',
               extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    plt.subplot(2, 3, 2)
    plt.imshow(np.max(p_conn_offset, 0).T, interpolation='none', extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$y [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')
    plt.title('Data')

    plt.subplot(2, 3, 3)
    plt.imshow(np.max(p_conn_offset, 2).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$y [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    # (Model)
    plt.subplot(2, 3, 4)
    plt.imshow(np.max(model_val_xyz, 1).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    plt.subplot(2, 3, 5)
    plt.imshow(np.max(model_val_xyz, 0).T, interpolation='none', extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$y [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')
    plt.title('Model')

    plt.subplot(2, 3, 6)
    plt.imshow(np.max(model_val_xyz, 2).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$y [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    plt.suptitle(f'Offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_2d.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   5th order (position-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_5th_order(nodes, edges, src_node_ids, tgt_node_ids, position_bin_size_um=1000, offset_bin_size_um=100, offset_max_range_um=None, pos_map_file=None, **_):
    """Extract position-dependent connection probability (5th order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    pos_map = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_map is None else pos_map, [src_node_ids, tgt_node_ids])

    # Compute PRE position & POST-PRE offset matrices
    x_mat, y_mat, z_mat = compute_position_matrices(src_nrn_pos, tgt_nrn_pos)
    dx_mat, dy_mat, dz_mat = compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract position- & offset-dependent connection probabilities
    x_range, y_range, z_range = zip(np.minimum(np.nanmin(src_nrn_pos, 0), np.nanmin(tgt_nrn_pos, 0)), np.maximum(np.nanmax(src_nrn_pos, 0), np.nanmax(tgt_nrn_pos, 0)))

    if np.isscalar(position_bin_size_um): # Single scalar range value to be used for all dimensions
        assert position_bin_size_um > 0.0, 'ERROR: Position bin size must be larger than 0um!'
        bin_size_x = bin_size_y = bin_size_z = position_bin_size_um
    else: # Three values for x/y/z dimensions
        assert len(position_bin_size_um) == 3, 'ERROR: Position bin sizes in x/y/z dimension expected!'
        assert np.all([b > 0.0 for b in position_bin_size_um]), 'ERROR: Position bin size must be larger than 0um!'
        bin_size_x, bin_size_y, bin_size_z = position_bin_size_um

    num_bins_x = np.ceil((x_range[1] - x_range[0]) / bin_size_x).astype(int)
    num_bins_y = np.ceil((y_range[1] - y_range[0]) / bin_size_y).astype(int)
    num_bins_z = np.ceil((z_range[1] - z_range[0]) / bin_size_z).astype(int)

    x_bins = np.arange(0, num_bins_x + 1) * bin_size_x + x_range[0]
    y_bins = np.arange(0, num_bins_y + 1) * bin_size_y + y_range[0]
    z_bins = np.arange(0, num_bins_z + 1) * bin_size_z + z_range[0]

    if offset_max_range_um is None:
        dx_range, dy_range, dz_range = zip([np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)], [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)])
    elif np.isscalar(offset_max_range_um): # Single scalar range value to be used for all dimensions
        assert offset_max_range_um > 0.0, 'ERROR: Maximum range must be larger than 0um!'
        dx_range = dy_range = dz_range = [-offset_max_range_um, offset_max_range_um]
    else: # Three values for x/y/z dimensions
        assert len(offset_max_range_um) == 3, 'ERROR: Maximum range in x/y/z dimension expected!'
        assert np.all([r > 0.0 for r in offset_max_range_um]), 'ERROR: Maximum range must be larger than 0um!'
        dx_range, dy_range, dz_range = zip([-r for r in offset_max_range_um], offset_max_range_um)

    if np.isscalar(offset_bin_size_um): # Single scalar range value to be used for all dimensions
        assert offset_bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!'
        bin_size_dx = bin_size_dy = bin_size_dz = offset_bin_size_um
    else: # Three values for x/y/z dimensions
        assert len(offset_bin_size_um) == 3, 'ERROR: Offset bin sizes in x/y/z dimension expected!'
        assert np.all([b > 0.0 for b in offset_bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!'
        bin_size_dx, bin_size_dy, bin_size_dz = offset_bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_position, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [x_mat, y_mat, z_mat, dx_mat, dy_mat, dz_mat], [x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins])

    return {'p_conn_position': p_conn_position, 'x_bins': x_bins, 'y_bins': y_bins, 'z_bins': z_bins, 'dx_bins': dx_bins, 'dy_bins': dy_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_5th_order(p_conn_position, x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins, model_specs=None, **_):
    """Build 5th order model (linear interpolation or random forest regression model for position-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}
    x_bin_offset = 0.5 * np.diff(x_bins[:2])[0]
    y_bin_offset = 0.5 * np.diff(y_bins[:2])[0]
    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]

    x_pos = x_bins[:-1] + x_bin_offset # Positions at bin centers
    y_pos = y_bins[:-1] + y_bin_offset # Positions at bin centers
    z_pos = z_bins[:-1] + z_bin_offset # Positions at bin centers

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    dx_pos = dx_bins[:-1] + dx_bin_offset # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    model_inputs = ['x', 'y', 'z', 'dx', 'dy', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation':

        # Linear interpolation model => Removing dimensions with only single value from interpolation
        assert len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!'

        model_dict = {'model': 'np.maximum(interp_fct([pos for pos, sel in zip([x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos], np.array(p_conn_position.shape) > 1) if sel], np.squeeze(p_conn_position), np.array([pos for pos, sel in zip([np.array(x), np.array(y), np.array(z), np.array(dx), np.array(dy), np.array(dz)], np.array(p_conn_position.shape) > 1) if sel]).T, method="linear", bounds_error=False, fill_value=None), 0)',
                      'model_inputs': model_inputs,
                      'model_params': {'interp_fct': scipy.interpolate.interpn, 'x_pos': x_pos, 'y_pos': y_pos, 'z_pos': z_pos, 'dx_pos': dx_pos, 'dy_pos': dy_pos, 'dz_pos': dz_pos, 'p_conn_position': p_conn_position}}

    elif model_specs.get('name') == 'RandomForestRegressor':

        # Random Forest Regressor model
        xv, yv, zv, dxv, dyv, dzv = np.meshgrid(x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos, indexing='ij')
        data_pos = np.array([xv.flatten(), yv.flatten(), zv.flatten(), dxv.flatten(), dyv.flatten(), dzv.flatten()]).T
        data_val = p_conn_position.flatten()

        position_regr_model = RandomForestRegressor(random_state=0, **model_specs.get('kwargs', {}))
        position_regr_model.fit(data_pos, data_val)

        model_dict = {'model': 'np.maximum(position_regr_model.predict(np.array([np.array(x), np.array(y), np.array(z), np.array(dx), np.array(dy), np.array(dz)]).T), 0)',
                      'model_inputs': model_inputs,
                      'model_params': {'position_regr_model': position_regr_model}}

    else:
        assert False, f'ERROR: Model type "{model_specs.get("name")}" unknown!'

    print(f'POSITION MODEL: f(x, y, z, dx, dy, dz) ~ {model_specs.get("name")} {model_specs.get("kwargs", {})}')

    return model_dict


def plot_5th_order(out_dir, p_conn_position, x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, model_inputs, model_params, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (5th order)."""
    model_fct = model_building.get_model(model, model_inputs, model_params)

    x_bin_offset = 0.5 * np.diff(x_bins[:2])[0]
    y_bin_offset = 0.5 * np.diff(y_bins[:2])[0]
    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]

    x_pos_model = x_bins[:-1] + x_bin_offset # Positions at bin centers
    y_pos_model = y_bins[:-1] + y_bin_offset # Positions at bin centers
    z_pos_model = z_bins[:-1] + z_bin_offset # Positions at bin centers

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    model_ovsampl = 4 # Model oversampling factor (per dimension)
    dx_pos_model = np.linspace(dx_bins[0], dx_bins[-1], (len(dx_bins) - 1) * model_ovsampl + 1)
    dy_pos_model = np.linspace(dy_bins[0], dy_bins[-1], (len(dy_bins) - 1) * model_ovsampl + 1)
    dz_pos_model = np.linspace(dz_bins[0], dz_bins[-1], (len(dz_bins) - 1) * model_ovsampl + 1)
    xv, yv, zv, dxv, dyv, dzv = np.meshgrid(x_pos_model, y_pos_model, z_pos_model, dx_pos_model, dy_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([xv.flatten(), yv.flatten(), zv.flatten(), dxv.flatten(), dyv.flatten(), dzv.flatten()]).T # Regular grid
    # model_pos = np.random.uniform(low=[x_bins[0], y_bins[0], z_bins[0], dx_bins[0], dy_bins[0], dz_bins[0]], high=[x_bins[-1], y_bins[-1], z_bins[-1], dx_bins[-1], dy_bins[-1], dz_bins[-1]], size=[model_ovsampl**3 * len(x_bins) * len(y_bins) * len(z_bins), len(dx_bins) * len(dy_bins) * len(dz_bins), 3]) # Random sampling
    model_val = model_fct(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2], model_pos[:, 3], model_pos[:, 4], model_pos[:, 5])
    model_val_xyz = model_val.reshape([len(x_pos_model), len(y_pos_model), len(z_pos_model), len(dx_pos_model), len(dy_pos_model), len(dz_pos_model)])

    # 3D connection probability (data vs. model)
    num_p_bins = 100
    p_bins = np.linspace(0, max(np.max(p_conn_position), np.max(model_val)), num_p_bins + 1)
    p_color_map = plt.cm.ScalarMappable(cmap=JET, norm=plt.Normalize(vmin=p_bins[0], vmax=p_bins[-1]))
    p_colors = p_color_map.to_rgba(np.linspace(p_bins[0], p_bins[-1], num_p_bins))

    for ix in range(len(x_pos_model)):
        for iy in range(len(y_pos_model)):
            for iz in range(len(z_pos_model)):

                p_conn_sel = p_conn_position[ix, iy, iz, :, :, :]
                model_val_sel = model_val_xyz[ix, iy, iz, :, :, :]

                fig = plt.figure(figsize=(16, 6), dpi=300)
                # (Data)
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                for pidx in range(num_p_bins):
                    p_sel_idx = np.where(np.logical_and(p_conn_sel > p_bins[pidx], p_conn_sel <= p_bins[pidx + 1]))
                    plt.plot(dx_bins[p_sel_idx[0]] + dx_bin_offset, dy_bins[p_sel_idx[1]] + dy_bin_offset, dz_bins[p_sel_idx[2]] + dz_bin_offset, 'o', color=p_colors[pidx, :], alpha=0.1 + 0.9 * (pidx + 1) / num_p_bins, markeredgecolor='none')
                ax.view_init(30, 60)
                ax.set_xlim((dx_bins[0], dx_bins[-1]))
                ax.set_ylim((dy_bins[0], dy_bins[-1]))
                ax.set_zlim((dz_bins[0], dz_bins[-1]))
                ax.set_xlabel('$\\Delta$x [$\\mu$m]')
                ax.set_ylabel('$\\Delta$y [$\\mu$m]')
                ax.set_zlabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(p_color_map, label='Conn. prob.')
                plt.title(f'Data: N = {src_cell_count}x{tgt_cell_count} cells')

                # (Model)
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                for pidx in range(num_p_bins):
                    p_sel_idx = np.where(np.logical_and(model_val_sel > p_bins[pidx], model_val_sel <= p_bins[pidx + 1]))
                    plt.plot(dx_pos_model[p_sel_idx[0]].T, dy_pos_model[p_sel_idx[1]].T, dz_pos_model[p_sel_idx[2]].T, '.', color=p_colors[pidx, :], alpha=0.1 + 0.9 * (pidx + 1) / num_p_bins, markeredgecolor='none')
                ax.view_init(30, 60)
                ax.set_xlim((dx_bins[0], dx_bins[-1]))
                ax.set_ylim((dy_bins[0], dy_bins[-1]))
                ax.set_zlim((dz_bins[0], dz_bins[-1]))
                ax.set_xlabel('$\\Delta$x [$\\mu$m]')
                ax.set_ylabel('$\\Delta$y [$\\mu$m]')
                ax.set_zlabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(p_color_map, label='Conn. prob.')
                plt.title(f'Model: {model_specs.get("name")}')

                plt.suptitle(f'Position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>\nX={x_pos_model[ix]:.0f}$\\mu$m, Y={y_pos_model[iy]:.0f}$\\mu$m, Z={z_pos_model[iz]:.0f}$\\mu$m')
                plt.tight_layout()
                out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model_3d_x{ix}y{iy}z{iz}.png'))
                print(f'INFO: Saving {out_fn}...')
                plt.savefig(out_fn)

    # Max. intensity projection (data vs. model)
    for ix in range(len(x_pos_model)):
        for iy in range(len(y_pos_model)):
            for iz in range(len(z_pos_model)):

                p_conn_sel = p_conn_position[ix, iy, iz, :, :, :]
                model_val_sel = model_val_xyz[ix, iy, iz, :, :, :]

                plt.figure(figsize=(12, 6), dpi=300)
                # (Data)
                plt.subplot(2, 3, 1)
                plt.imshow(np.max(p_conn_sel, 1).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.subplot(2, 3, 2)
                plt.imshow(np.max(p_conn_sel, 0).T, interpolation='none', extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$y [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')
                plt.title('Data')

                plt.subplot(2, 3, 3)
                plt.imshow(np.max(p_conn_sel, 2).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$y [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                # (Model)
                plt.subplot(2, 3, 4)
                plt.imshow(np.max(model_val_sel, 1).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.subplot(2, 3, 5)
                plt.imshow(np.max(model_val_sel, 0).T, interpolation='none', extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$y [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')
                plt.title('Model')

                plt.subplot(2, 3, 6)
                plt.imshow(np.max(model_val_sel, 2).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]), cmap=HOT, vmin=0.0)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$y [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.suptitle(f'Position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>\nX={x_pos_model[ix]:.0f}$\\mu$m, Y={y_pos_model[iy]:.0f}$\\mu$m, Z={z_pos_model[iz]:.0f}$\\mu$m')
                plt.tight_layout()
                out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model_2d_x{ix}y{iy}z{iz}.png'))
                print(f'INFO: Saving {out_fn}...')
                plt.savefig(out_fn)
