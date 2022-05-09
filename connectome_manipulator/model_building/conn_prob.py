"""
Module for building connection probability models of different orders, consisting of three basic functions:
  -extract(...): Extracts connection probability between samples of neurons
  -build(...): Fits a connection probability model to data
  -plot(...): Visualizes extracted data vs. actual model output
"""

import itertools
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor

from connectome_manipulator import log
from connectome_manipulator.model_building import model_types
from connectome_manipulator.access_functions import get_node_ids, get_edges_population


JET = plt.cm.get_cmap('jet')
HOT = plt.cm.get_cmap('hot')


def extract(circuit, order, sel_src=None, sel_dest=None, sample_size=None, **kwargs):
    """Extract connection probability between samples of neurons."""
    log.info(f'Running order-{order} data extraction (sel_src={sel_src}, sel_dest={sel_dest}, sample_size={sample_size} neurons)...')

    # Select edge population [assuming exactly one edge population in given edges file]
    edges = get_edges_population(circuit)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    nodes = [src_nodes, tgt_nodes]

    node_ids_src = get_node_ids(src_nodes, sel_src)
    node_ids_dest = get_node_ids(tgt_nodes, sel_dest)

    if not kwargs.get('pos_map_file') is None:
        log.log_assert((src_nodes.name == tgt_nodes.name) or (src_nodes.ids().min() > tgt_nodes.ids().max()) or (src_nodes.ids().max() < tgt_nodes.ids().min()), 'ERROR: Position mapping only supported for same source/taget node population or non-overlapping id ranges!')

    if sample_size is None or sample_size <= 0:
        sample_size = np.inf # Select all nodes
    if sample_size < len(node_ids_src) or sample_size < len(node_ids_dest):
        log.warning('Sub-sampling neurons! Consider running model building with a different random sub-samples!')
    sample_size_src = min(sample_size, len(node_ids_src))
    sample_size_dest = min(sample_size, len(node_ids_dest))
    log.log_assert(sample_size_src > 0 and sample_size_dest > 0, 'ERROR: Empty nodes selection!')
    node_ids_src_sel = node_ids_src[np.random.permutation([True] * sample_size_src + [False] * (len(node_ids_src) - sample_size_src))]
    node_ids_dest_sel = node_ids_dest[np.random.permutation([True] * sample_size_dest + [False] * (len(node_ids_dest) - sample_size_dest))]

    if not isinstance(order, str):
        order = str(order)

    if order == '1':
        return extract_1st_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '2':
        return extract_2nd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '3':
        return extract_3rd_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '4':
        return extract_4th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '4R':
        return extract_4th_order_reduced(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '5':
        return extract_5th_order(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    elif order == '5R':
        return extract_5th_order_reduced(nodes, edges, node_ids_src_sel, node_ids_dest_sel, **kwargs)
    else:
        log.log_assert(False, f'ERROR: Order-{order} data extraction not supported!')


def build(order, **kwargs):
    """Build connection probability model from data."""
    log.info(f'Running order-{order} model building...')

    if not isinstance(order, str):
        order = str(order)

    if order == '1':
        return build_1st_order(**kwargs)
    elif order == '2':
        return build_2nd_order(**kwargs)
    elif order == '3':
        return build_3rd_order(**kwargs)
    elif order == '4':
        return build_4th_order(**kwargs)
    elif order.upper() == '4R':
        return build_4th_order_reduced(**kwargs)
    elif order == '5':
        return build_5th_order(**kwargs)
    elif order == '5R':
        return build_5th_order_reduced(**kwargs)
    else:
        log.log_assert(False, f'ERROR: Order-{order} model building not supported!')


def plot(order, **kwargs):
    """Visualize data vs. model."""
    log.info(f'Running order-{order} data/model visualization...')

    if not isinstance(order, str):
        order = str(order)

    if order == '1':
        return plot_1st_order(**kwargs)
    elif order == '2':
        return plot_2nd_order(**kwargs)
    elif order == '3':
        return plot_3rd_order(**kwargs)
    elif order == '4':
        return plot_4th_order(**kwargs)
    elif order == '4R':
        return plot_4th_order_reduced(**kwargs)
    elif order == '5':
        return plot_5th_order(**kwargs)
    elif order == '5R':
        return plot_5th_order_reduced(**kwargs)
    else:
        log.log_assert(False, f'ERROR: Order-{order} data/model visualization not supported!')


###################################################################################################
# Helper functions
###################################################################################################

def load_pos_mapping_model(pos_map_file):
    """Load a position mapping model from file (incl. access function)."""
    if pos_map_file is None:
        pos_map = None
        pos_acc = None
    else:
        log.log_assert(os.path.exists(pos_map_file), 'Position mapping model file not found!')
        log.info(f'Loading position mapping model from {pos_map_file}')
        pos_map = model_types.AbstractModel.model_from_file(pos_map_file)
        log.log_assert(pos_map.input_names == ['gids'], 'ERROR: Position mapping model error (must take "gids" as input)!')
        pos_acc = lambda gids: pos_map.apply(gids=gids) # Access function

    return pos_map, pos_acc


def get_neuron_positions(pos_fct, node_ids_list):
    """Get neuron positions (using position access/mapping function) [NOTE: node_ids_list should be list of node_ids lists!]."""
    if not isinstance(pos_fct, list):
        pos_fct = [pos_fct for i in node_ids_list]
    else:
        log.log_assert(len(pos_fct) == len(node_ids_list), 'ERROR: "pos_fct" must be scalar or a list with same length as "node_ids_list"!')

    nrn_pos = [np.array(pos_fct[i](node_ids_list[i])) for i in range(len(node_ids_list))]

    return nrn_pos


# NOT USED ANY MORE (model-specific implementation to be used instead for better consistency)
# def compute_dist_matrix(src_nrn_pos, tgt_nrn_pos):
#     """Computes distance matrix between pairs of neurons."""
#     dist_mat = distance_matrix(src_nrn_pos, tgt_nrn_pos)
#     dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections

#     return dist_mat


# NOT USED ANY MORE (model-specific implementation to be used instead for better consistency)
# def compute_bip_matrix(src_nrn_pos, tgt_nrn_pos):
#     """Computes bipolar matrix between pairs of neurons (along z-axis; POST-synaptic neuron below (delta_z < 0) or above (delta_z > 0) PRE-synaptic neuron)."""
#     bip_mat = np.sign(np.diff(np.meshgrid(src_nrn_pos[:, 2], tgt_nrn_pos[:, 2], indexing='ij'), axis=0)[0, :, :]) # Bipolar distinction based on difference in z coordinate

#     return bip_mat


# NOT USED ANY MORE (model-specific implementation to be used instead for better consistency)
# def compute_offset_matrices(src_nrn_pos, tgt_nrn_pos):
#     """Computes dx/dy/dz offset matrices between pairs of neurons (POST minus PRE position)."""
#     dx_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 0], tgt_nrn_pos[:, 0], indexing='ij'), axis=0)) # Relative difference in x coordinate
#     dy_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 1], tgt_nrn_pos[:, 1], indexing='ij'), axis=0)) # Relative difference in y coordinate
#     dz_mat = np.squeeze(np.diff(np.meshgrid(src_nrn_pos[:, 2], tgt_nrn_pos[:, 2], indexing='ij'), axis=0)) # Relative difference in z coordinate

#     return dx_mat, dy_mat, dz_mat


# NOT USED ANY MORE (model-specific implementation to be used instead for better consistency)
# def compute_position_matrices(src_nrn_pos, tgt_nrn_pos):
#     """Computes x/y/z position matrices (PRE neuron positions repeated over POST neuron number)."""
#     x_mat, y_mat, z_mat = [np.tile(src_nrn_pos[:, i:i + 1], [1, tgt_nrn_pos.shape[0]]) for i in range(src_nrn_pos.shape[1])]

#     return x_mat, y_mat, z_mat


def extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, dep_matrices, dep_bins, min_count_per_bin=None):
    """Extract D-dimensional conn. prob. dependent on D property matrices between source-target pairs of neurons within given range of bins."""
    num_dep = len(dep_matrices)
    log.log_assert(len(dep_bins) == num_dep, 'ERROR: Dependencies/bins mismatch!')
    log.log_assert(np.all([dep_matrices[dim].shape == (len(src_node_ids), len(tgt_node_ids)) for dim in range(num_dep)]), 'ERROR: Matrix dimension mismatch!')

    # Extract adjacency
    conns = np.array(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids)))
    if len(conns) > 0:
        adj_mat = csr_matrix((np.full(conns.shape[0], True), conns.T.tolist()), shape=(max(src_node_ids) + 1, max(tgt_node_ids) + 1))
    else:
        adj_mat = csr_matrix((max(src_node_ids) + 1, max(tgt_node_ids) + 1)) # Empty matrix
    if np.any(adj_mat.diagonal()):
        log.warning('Autaptic connection(s) found!')

    # Extract connection probability
    num_bins = [len(b) - 1 for b in dep_bins]
    bin_indices = [list(range(n)) for n in num_bins]
    count_all = np.full(num_bins, -1) # Count of all pairs of neurons for each combination of dependencies
    count_conn = np.full(num_bins, -1) # Count of connected pairs of neurons for each combination of dependencies

    log.info(f'Extracting {num_dep}-dimensional ({"x".join([str(n) for n in num_bins])}) connection probabilities...')
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
#     p_conn[np.isnan(p_conn)] = 0.0

    # Check bin counts below threshold and ignore
    if min_count_per_bin is None:
        min_count_per_bin = 0 # No threshold
    bad_bins = np.logical_and(count_all > 0, count_all < min_count_per_bin)
    if np.sum(bad_bins) > 0:
        log.warning(f'Found {np.sum(bad_bins)} of {count_all.size} ({100.0 * np.sum(bad_bins) / count_all.size:.1f}%) bins with less than th={min_count_per_bin} pairs of neurons ... IGNORING! (Consider increasing sample size and/or bin size and/or smoothing!)')
        p_conn[bad_bins] = np.nan # 0.0

    return p_conn, count_conn, count_all


def get_value_ranges(max_range, num_coords, pos_range=False):
    """Returns ranges of values for given max. ranges (strictly positive incl. zero, symmetric around zero, or arbitrary)"""
    if np.isscalar(pos_range):
        pos_range = [pos_range for i in range(num_coords)]
    else:
        if num_coords == 1: # Special case
            pos_range = [pos_range]
        log.log_assert(len(pos_range) == num_coords, f'ERROR: pos_range must have {num_coords} elements!')

    if np.isscalar(max_range):
        max_range = [max_range for i in range(num_coords)]
    else:
        if num_coords == 1: # Special case
            max_range = [max_range]
        log.log_assert(len(max_range) == num_coords, f'ERROR: max_range must have {num_coords} elements!')

    val_ranges = []
    for ridx, (r, p) in enumerate(zip(max_range, pos_range)):
        if np.isscalar(r):
            log.log_assert(r > 0.0, f'ERROR: Maximum range of coord {ridx} must be larger than 0!')
            if p: # Positive range
                val_ranges.append([0, r])
            else: # Symmetric range
                val_ranges.append([-r, r])
        else: # Arbitrary range
            log.log_assert(len(r) == 2 and r[0] < r[1], f'ERROR: Range of coord {ridx} invalid!')
            if p:
                log.log_assert(r[0] == 0, f'ERROR: Range of coord {ridx} must include 0!')
            val_ranges.append(r)

    if num_coords == 1: # Special case
        return val_ranges[0]
    else:
        return val_ranges


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   1st order model (Erdos-Renyi)
###################################################################################################

def extract_1st_order(_nodes, edges, src_node_ids, tgt_node_ids, min_count_per_bin=10, **_):
    """Extract average connection probability (1st order) from a sample of pairs of neurons."""
    p_conn, conn_count, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [], [], min_count_per_bin)

    src_cell_count = len(src_node_ids)
    tgt_cell_count = len(tgt_node_ids)
    log.info(f'Found {conn_count} connections between {src_cell_count}x{tgt_cell_count} neurons (p = {p_conn:.3f})')

    return {'p_conn': p_conn, 'src_cell_count': src_cell_count, 'tgt_cell_count': tgt_cell_count}


def build_1st_order(p_conn, **_):
    """Build 1st order model (Erdos-Renyi, capturing average conn. prob.)."""

    # Create model
    model = model_types.ConnProb1stOrderModel(p_conn=p_conn)
    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_1st_order(out_dir, p_conn, src_cell_count, tgt_cell_count, model, **_):  # pragma: no cover
    """Visualize data vs. model (1st order)."""
    model_params = model.get_param_dict()
    model_str = f'f(x) = {model_params["p_conn"]:.3f}'

    # Draw figure
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(0.5, p_conn, width=1, facecolor='tab:blue', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot([-0.5, 1.5], np.ones(2) * model.get_conn_prob(), '--', color='tab:red', label=f'Model: {model_str}')
    plt.text(0.5, 0.99 * p_conn, f'p = {p_conn:.3f}', color='k', ha='center', va='top')
    plt.xticks([])
    plt.ylabel('Conn. prob.')
    plt.title('Average conn. prob. (1st-order)', fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    plt.tight_layout()

    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   2nd order (distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################

def extract_2nd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, min_count_per_bin=10, **_):
    """Extract distance-dependent connection probability (2nd order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_acc is None else pos_acc, [src_node_ids, tgt_node_ids])

    # Compute distance matrix
    dist_mat = model_types.ConnProb2ndOrderExpModel.compute_dist_matrix(src_nrn_pos, tgt_nrn_pos)

    # Extract distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    log.log_assert(max_range_um > 0 and bin_size_um > 0, 'ERROR: Max. range and bin size must be larger than 0um!')
    num_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_bins + 1) * bin_size_um

    p_conn_dist, count_conn, count_all = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat], [dist_bins], min_count_per_bin)

    return {'p_conn_dist': p_conn_dist, 'count_conn': count_conn, 'count_all': count_all, 'dist_bins': dist_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_2nd_order(p_conn_dist, dist_bins, **_):
    """Build 2nd order model (exponential distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    X = dist_bins[:-1][np.isfinite(p_conn_dist)] + bin_offset
    y = p_conn_dist[np.isfinite(p_conn_dist)]
    try:
        (a_opt, b_opt), _ = curve_fit(exp_model, X, y, p0=[0.0, 0.0])
    except Exception as e:
        log.error(e)
        (a_opt, b_opt) = (np.nan, np.nan)

    # Create model
    model = model_types.ConnProb2ndOrderExpModel(scale=a_opt, exponent=b_opt)
    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_2nd_order(out_dir, p_conn_dist, count_conn, count_all, dist_bins, src_cell_count, tgt_cell_count, model, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (2nd order)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_params = model.get_param_dict()
    model_str = f'f(x) = {model_params["scale"]:.3f} * exp(-{model_params["exponent"]:.3f} * x)'

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    plt.plot(dist_bins[:-1] + bin_offset, p_conn_dist, '.-', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot(dist_model, model.get_conn_prob(dist_model), '--', label='Model: ' + model_str)
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
    pdist = model.get_conn_prob(vdist)
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
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)

    # Data counts
    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(dist_bins[:-1] + bin_offset, count_all, width=1.5 * bin_offset, label='All pair count')
    plt.bar(dist_bins[:-1] + bin_offset, count_conn, width=1.0 * bin_offset, label='Connection count')
    plt.gca().set_yscale('log')
    plt.grid()
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Count')
    plt.title(f'Distance-dependent connection counts (N = {src_cell_count}x{tgt_cell_count} cells)\n<Position mapping: {pos_map_file}>')
    plt.legend()
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_counts.png'))
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   3rd order (bipolar distance-dependent) => Position mapping model (flatmap) supported
###################################################################################################

def extract_3rd_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, no_dist_mapping=False, min_count_per_bin=10, **_):
    """Extract distance-dependent connection probability (3rd order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    src_nrn_pos_raw, tgt_nrn_pos_raw = get_neuron_positions([n.positions for n in nodes], [src_node_ids, tgt_node_ids]) # Raw positions w/o mapping
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    if pos_acc is None:
        src_nrn_pos = src_nrn_pos_raw
        tgt_nrn_pos = tgt_nrn_pos_raw
    else:
        src_nrn_pos, tgt_nrn_pos = get_neuron_positions(pos_acc, [src_node_ids, tgt_node_ids])

    # Compute distance matrix
    if no_dist_mapping: # Don't use position mapping for computing distances
        dist_mat = model_types.ConnProb3rdOrderExpModel.compute_dist_matrix(src_nrn_pos_raw, tgt_nrn_pos_raw)
    else: # Use position mapping for computing distances
        dist_mat = model_types.ConnProb3rdOrderExpModel.compute_dist_matrix(src_nrn_pos, tgt_nrn_pos)

    # Compute bipolar matrix (always using position mapping, if provided; along z-axis; post-synaptic neuron below (delta_z < 0) or above (delta_z > 0) pre-synaptic neuron)
    bip_mat = model_types.ConnProb3rdOrderExpModel.compute_bip_matrix(src_nrn_pos, tgt_nrn_pos)

    # Extract bipolar distance-dependent connection probabilities
    if max_range_um is None:
        max_range_um = np.nanmax(dist_mat)
    log.log_assert(max_range_um > 0 and bin_size_um > 0, 'ERROR: Max. range and bin size must be larger than 0um!')
    num_dist_bins = np.ceil(max_range_um / bin_size_um).astype(int)
    dist_bins = np.arange(0, num_dist_bins + 1) * bin_size_um
    bip_bins = [np.min(bip_mat), 0, np.max(bip_mat)]

    p_conn_dist_bip, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dist_mat, bip_mat], [dist_bins, bip_bins], min_count_per_bin)

    return {'p_conn_dist_bip': p_conn_dist_bip, 'dist_bins': dist_bins, 'bip_bins': bip_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_3rd_order(p_conn_dist_bip, dist_bins, **_):
    """Build 3rd order model (bipolar exp. distance-dependent conn. prob.)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]

    X = dist_bins[:-1][np.all(np.isfinite(p_conn_dist_bip), 1)] + bin_offset
    y = p_conn_dist_bip[np.all(np.isfinite(p_conn_dist_bip), 1), :]

    exp_model = lambda x, a, b: a * np.exp(-b * np.array(x))
    try:
        (aN_opt, bN_opt), _ = curve_fit(exp_model, X, y[:, 0], p0=[0.0, 0.0])
        (aP_opt, bP_opt), _ = curve_fit(exp_model, X, y[:, 1], p0=[0.0, 0.0])
    except Exception as e:
        log.error(e)
        (aN_opt, bN_opt) = (np.nan, np.nan)
        (aP_opt, bP_opt) = (np.nan, np.nan)

    # Create model
    model = model_types.ConnProb3rdOrderExpModel(scale_N=aN_opt, exponent_N=bN_opt, scale_P=aP_opt, exponent_P=bP_opt, bip_coord=2) # [bip_coord=2 ... bipolar along z-axis]
    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_3rd_order(out_dir, p_conn_dist_bip, dist_bins, src_cell_count, tgt_cell_count, model, pos_map_file=None, **_):  # pragma: no cover
    """Visualize data vs. model (3rd order)."""
    bin_offset = 0.5 * np.diff(dist_bins[:2])[0]
    dist_model = np.linspace(dist_bins[0], dist_bins[-1], 100)

    model_params = model.get_param_dict()
    model_strN = f'{model_params["scale_N"]:.3f} * exp(-{model_params["exponent_N"]:.3f} * x)'
    model_strP = f'{model_params["scale_P"]:.3f} * exp(-{model_params["exponent_P"]:.3f} * x)'

    plt.figure(figsize=(12, 4), dpi=300)

    # Data vs. model
    plt.subplot(1, 2, 1)
    bip_dist = np.concatenate((-dist_bins[:-1][::-1] - bin_offset, [0.0], dist_bins[:-1] + bin_offset))
    bip_data = np.concatenate((p_conn_dist_bip[::-1, 0], [np.nan], p_conn_dist_bip[:, 1]))
    plt.plot(bip_dist, bip_data, '.-', label=f'Data: N = {src_cell_count}x{tgt_cell_count} cells')
    plt.plot(-dist_model, model.get_conn_prob(dist_model, np.sign(-dist_model)), '--', label='Model: ' + model_strN)
    plt.plot(dist_model, model.get_conn_prob(dist_model, np.sign(dist_model)), '--', label='Model: ' + model_strP)
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
    pdist = model.get_conn_prob(vdist, np.sign(zv))
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
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   4th order (offset-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_4th_order(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, min_count_per_bin=10, **_):
    """Extract offset-dependent connection probability (4th order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_acc is None else pos_acc, [src_node_ids, tgt_node_ids])

    # Compute dx/dy/dz offset matrices
    dx_mat, dy_mat, dz_mat = model_types.ConnProb4thOrderLinInterpnModel.compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract offset-dependent connection probabilities
    if max_range_um is None:
        dx_range, dy_range, dz_range = zip([np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)], [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)])
    else:
        dx_range, dy_range, dz_range = get_value_ranges(max_range_um, 3, pos_range=False)

    if np.isscalar(bin_size_um): # Single scalar range value to be used for all dimensions
        log.log_assert(bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dx = bin_size_dy = bin_size_dz = bin_size_um
    else: # Three values for x/y/z dimensions
        log.log_assert(len(bin_size_um) == 3, 'ERROR: Offset bin sizes in x/y/z dimension expected!')
        log.log_assert(np.all([b > 0.0 for b in bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dx, bin_size_dy, bin_size_dz = bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_offset, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dx_mat, dy_mat, dz_mat], [dx_bins, dy_bins, dz_bins], min_count_per_bin)

    return {'p_conn_offset': p_conn_offset, 'dx_bins': dx_bins, 'dy_bins': dy_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_4th_order(p_conn_offset, dx_bins, dy_bins, dz_bins, model_specs=None, smoothing_sigma_um=None, **_):
    """Build 4th order model (linear interpolation or random forest regression model for offset-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}

    bin_sizes = [np.diff(dx_bins[:2])[0], np.diff(dy_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    dx_bin_offset = 0.5 * bin_sizes[0]
    dy_bin_offset = 0.5 * bin_sizes[1]
    dz_bin_offset = 0.5 * bin_sizes[2]

    dx_pos = dx_bins[:-1] + dx_bin_offset # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_offset = p_conn_offset.copy()
    p_conn_offset[np.isnan(p_conn_offset)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 3 # Same value for all coordinates
        else:
            log.log_assert(len(smoothing_sigma_um) == 3, 'ERROR: Smoothing sigma for 3 dimensions required!')
        log.log_assert(np.all(np.array(smoothing_sigma_um) >= 0.0), 'ERROR: Smoothing sigma must be non-negative!')
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.info(f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms')
        p_conn_offset = gaussian_filter(p_conn_offset, sigmas, mode='constant')

    model_inputs = ['dx', 'dy', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation': # Linear interpolation model => Removing dimensions with only single value from interpolation

        log.log_assert(len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!')

        # Create model
        index = pd.MultiIndex.from_product([dx_pos, dy_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_offset.flatten(), index=index, columns=['p'])
        model = model_types.ConnProb4thOrderLinInterpnModel(p_conn_table=df)

    elif model_specs.get('name') == 'RandomForestRegressor': # Random Forest Regressor model

        log.log_assert(False, 'ERROR: No model class implemented for RandomForestRegressor!')

#         dxv, dyv, dzv = np.meshgrid(dx_pos, dy_pos, dz_pos, indexing='ij')
#         data_pos = np.array([dxv.flatten(), dyv.flatten(), dzv.flatten()]).T
#         data_val = p_conn_offset.flatten()

#         offset_regr_model = RandomForestRegressor(random_state=0, **model_specs.get('kwargs', {}))
#         offset_regr_model.fit(data_pos, data_val)

#         # Create model
#         model = model_types...

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("name")}" unknown!')

    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_4th_order(out_dir, p_conn_offset, dx_bins, dy_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, pos_map_file=None, plot_model_ovsampl=3, plot_model_extsn=0, **_):  # pragma: no cover
    """Visualize data vs. model (4th order)."""

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1, 'ERROR: Model plot oversampling must be an integer factor >= 1!')
    log.log_assert(isinstance(plot_model_extsn, int) and plot_model_extsn >= 0, 'ERROR: Model plot extension must be an integer number of bins >= 0!')
    dx_bin_size_model = np.diff(dx_bins[:2])[0] / plot_model_ovsampl
    dy_bin_size_model = np.diff(dy_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dx_bins_model = np.arange(dx_bins[0] - plot_model_extsn * dx_bin_size_model * plot_model_ovsampl, dx_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dx_bin_size_model, dx_bin_size_model)
    dy_bins_model = np.arange(dy_bins[0] - plot_model_extsn * dy_bin_size_model * plot_model_ovsampl, dy_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dy_bin_size_model, dy_bin_size_model)
    dz_bins_model = np.arange(dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl, dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model, dz_bin_size_model)

    # Sample positions (at bin centers)
    dx_pos_model = dx_bins_model[:-1] + 0.5 * dx_bin_size_model
    dy_pos_model = dy_bins_model[:-1] + 0.5 * dy_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    dxv, dyv, dzv = np.meshgrid(dx_pos_model, dy_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([dxv.flatten(), dyv.flatten(), dzv.flatten()]).T # Regular grid
    # model_pos = np.random.uniform(low=[dx_bins[0], dy_bins[0], dz_bins[0]], high=[dx_bins[-1], dy_bins[-1], dz_bins[-1]], size=[model_ovsampl**3 * len(dx_bins) * len(dy_bins) * len(dz_bins), 3]) # Random sampling
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2])
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
        plt.plot(dx_bins[p_sel_idx[0]] + dx_bin_offset, dy_bins[p_sel_idx[1]] + dy_bin_offset, dz_bins[p_sel_idx[2]] + dz_bin_offset, 'o', color=p_colors[pidx, :], alpha=0.01 + 0.99 * pidx / (num_p_bins - 1), markeredgecolor='none')
#     ax.view_init(30, 60)
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
        plt.plot(model_pos[p_sel_idx, 0], model_pos[p_sel_idx, 1], model_pos[p_sel_idx, 2], '.', color=p_colors[pidx, :], alpha=0.01 + 0.99 * pidx / (num_p_bins - 1), markeredgecolor='none')
#     ax.view_init(30, 60)
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
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)

    # Max. intensity projection (data vs. model)
    plt.figure(figsize=(12, 6), dpi=300)
    # (Data)
    plt.subplot(2, 3, 1)
    plt.imshow(np.max(p_conn_offset, 1).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
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
    plt.imshow(np.max(model_val_xyz, 1).T, interpolation='none', extent=(dx_bins_model[0], dx_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    plt.subplot(2, 3, 5)
    plt.imshow(np.max(model_val_xyz, 0).T, interpolation='none', extent=(dy_bins_model[0], dy_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$y [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')
    plt.title('Model')

    plt.subplot(2, 3, 6)
    plt.imshow(np.max(model_val_xyz, 2).T, interpolation='none', extent=(dx_bins_model[0], dx_bins_model[-1], dy_bins_model[-1], dy_bins_model[0]), cmap=HOT, vmin=0.0)
    plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
    plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$x [$\\mu$m]')
    plt.ylabel('$\\Delta$y [$\\mu$m]')
    plt.colorbar(label='Max. conn. prob.')

    plt.suptitle(f'Offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_2d.png'))
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity:
#   Reduced 4th order (offset-dependent), modified from [Gal et al. 2020]
#     => Radial/axial offsets only
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_4th_order_reduced(nodes, edges, src_node_ids, tgt_node_ids, bin_size_um=100, max_range_um=None, pos_map_file=None, min_count_per_bin=10, **_):
    """Extract offset-dependent connection probability (reduced 4th order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_acc is None else pos_acc, [src_node_ids, tgt_node_ids])

    # Compute dr/dz offset matrices
    dr_mat, dz_mat = model_types.ConnProb4thOrderLinInterpnReducedModel.compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract offset-dependent connection probabilities
    if max_range_um is None:
        dr_range, dz_range = zip([0, np.nanmin(dz_mat)], [np.nanmax(dr_mat), np.nanmax(dz_mat)])
    else:
        dr_range, dz_range = get_value_ranges(max_range_um, 2, pos_range=[True, False])

    if np.isscalar(bin_size_um): # Single scalar range value to be used for all dimensions
        log.log_assert(bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dr = bin_size_dz = bin_size_um
    else: # Two values for r/z directions
        log.log_assert(len(bin_size_um) == 2, 'ERROR: Offset bin sizes in r/z directions expected!')
        log.log_assert(np.all([b > 0.0 for b in bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dr, bin_size_dz = bin_size_um

    num_bins_dr = np.ceil((dr_range[1] - dr_range[0]) / bin_size_dr).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dr_bins = np.arange(0, num_bins_dr + 1) * bin_size_dr + dr_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_offset, count_conn_offset, count_all_offset = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [dr_mat, dz_mat], [dr_bins, dz_bins], min_count_per_bin)

    return {'p_conn_offset': p_conn_offset, 'count_conn_offset': count_conn_offset, 'count_all_offset': count_all_offset, 'dr_bins': dr_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_4th_order_reduced(p_conn_offset, dr_bins, dz_bins, model_specs=None, smoothing_sigma_um=None, **_):
    """Build reduced 4th order model (linear interpolation or random forest regression model for offset-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}

    bin_sizes = [np.diff(dr_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    dr_bin_offset = 0.5 * bin_sizes[0]
    dz_bin_offset = 0.5 * bin_sizes[1]

    dr_pos = dr_bins[:-1] + dr_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_offset = p_conn_offset.copy()
    p_conn_offset[np.isnan(p_conn_offset)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 2 # Same value for all coordinates
        else:
            log.log_assert(len(smoothing_sigma_um) == 2, 'ERROR: Smoothing sigma for 2 dimensions required!')
        log.log_assert(np.all(np.array(smoothing_sigma_um) >= 0.0), 'ERROR: Smoothing sigma must be non-negative!')
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.info(f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms')
        p_reflect = np.vstack([p_conn_offset[::-1, :], p_conn_offset]) # Mirror along radial axis at dr==0, to avoid edge effect
        p_reflect = gaussian_filter(p_reflect, sigmas, mode='constant')
        p_conn_offset = p_reflect[p_conn_offset.shape[0]:, :] # Cut original part of the data

    model_inputs = ['dr', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation': # Linear interpolation model => Removing dimensions with only single value from interpolation

        log.log_assert(len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!')

        # Create model
        index = pd.MultiIndex.from_product([dr_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_offset.flatten(), index=index, columns=['p'])
        model = model_types.ConnProb4thOrderLinInterpnReducedModel(p_conn_table=df)

    elif model_specs.get('name') == 'RandomForestRegressor': # Random Forest Regressor model
        log.log_assert(False, 'ERROR: No model class implemented for RandomForestRegressor!')

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("name")}" unknown!')

    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_4th_order_reduced(out_dir, p_conn_offset, dr_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, pos_map_file=None, plot_model_ovsampl=3, plot_model_extsn=0, **_):  # pragma: no cover
    """Visualize data vs. model (4th order reduced)."""

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1, 'ERROR: Model plot oversampling must be an integer factor >= 1!')
    log.log_assert(isinstance(plot_model_extsn, int) and plot_model_extsn >= 0, 'ERROR: Model plot extension must be an integer number of bins >= 0!')
    dr_bin_size_model = np.diff(dr_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dr_bins_model = np.arange(dr_bins[0], dr_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dr_bin_size_model, dr_bin_size_model)
    dz_bins_model = np.arange(dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl, dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model, dz_bin_size_model)

    # Sample positions (at bin centers)
    dr_pos_model = dr_bins_model[:-1] + 0.5 * dr_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    drv, dzv = np.meshgrid(dr_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([drv.flatten(), dzv.flatten()]).T # Regular grid
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1])
    model_val = model_val.reshape([len(dr_pos_model), len(dz_pos_model)])

    # Connection probability (data vs. model)
    fig = plt.figure(figsize=(12, 4), dpi=300)
    # (Data)
    log.log_assert(dr_bins[0] == 0, 'ERROR: Radial bin range error!')
    plt.subplot(1, 2, 1)
    plt.imshow(np.hstack([p_conn_offset.T[:, ::-1], p_conn_offset.T]), interpolation='nearest', extent=(-dr_bins[-1], dr_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0)
    plt.plot(np.zeros(2), plt.ylim(), color='lightgrey', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$r [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Conn. prob.')
    plt.title(f'Data: N = {src_cell_count}x{tgt_cell_count} cells')

    # (Model)
    plt.subplot(1, 2, 2)
    plt.imshow(np.hstack([model_val.T[:, ::-1], model_val.T]), interpolation='nearest', extent=(-dr_bins_model[-1], dr_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0)
    plt.plot(np.zeros(2), plt.ylim(), color='lightgrey', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('$\\Delta$r [$\\mu$m]')
    plt.ylabel('$\\Delta$z [$\\mu$m]')
    plt.colorbar(label='Conn. prob.')
    plt.title(f'Model: {model_specs.get("name")}')

    plt.suptitle(f'Reduced offset-dependent connection probability model (4th order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity from [Gal et al. 2020]:
#   5th order (position-dependent)
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_5th_order(nodes, edges, src_node_ids, tgt_node_ids, position_bin_size_um=1000, position_max_range_um=None, offset_bin_size_um=100, offset_max_range_um=None, pos_map_file=None, min_count_per_bin=10, **_):
    """Extract position-dependent connection probability (5th order) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_acc is None else pos_acc, [src_node_ids, tgt_node_ids])

    # Compute PRE position & POST-PRE offset matrices
    x_mat, y_mat, z_mat = model_types.ConnProb5thOrderLinInterpnModel.compute_position_matrices(src_nrn_pos, tgt_nrn_pos)
    dx_mat, dy_mat, dz_mat = model_types.ConnProb5thOrderLinInterpnModel.compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract position- & offset-dependent connection probabilities
    if position_max_range_um is None:
        x_range, y_range, z_range = zip(np.minimum(np.nanmin(src_nrn_pos, 0), np.nanmin(tgt_nrn_pos, 0)), np.maximum(np.nanmax(src_nrn_pos, 0), np.nanmax(tgt_nrn_pos, 0)))
    else:
        x_range, y_range, z_range = get_value_ranges(position_max_range_um, 3, pos_range=False)
    
    if np.isscalar(position_bin_size_um): # Single scalar range value to be used for all dimensions
        log.log_assert(position_bin_size_um > 0.0, 'ERROR: Position bin size must be larger than 0um!')
        bin_size_x = bin_size_y = bin_size_z = position_bin_size_um
    else: # Three values for x/y/z dimensions
        log.log_assert(len(position_bin_size_um) == 3, 'ERROR: Position bin sizes in x/y/z dimension expected!')
        log.log_assert(np.all([b > 0.0 for b in position_bin_size_um]), 'ERROR: Position bin size must be larger than 0um!')
        bin_size_x, bin_size_y, bin_size_z = position_bin_size_um

    num_bins_x = np.ceil((x_range[1] - x_range[0]) / bin_size_x).astype(int)
    num_bins_y = np.ceil((y_range[1] - y_range[0]) / bin_size_y).astype(int)
    num_bins_z = np.ceil((z_range[1] - z_range[0]) / bin_size_z).astype(int)

    x_bins = np.arange(0, num_bins_x + 1) * bin_size_x + x_range[0]
    y_bins = np.arange(0, num_bins_y + 1) * bin_size_y + y_range[0]
    z_bins = np.arange(0, num_bins_z + 1) * bin_size_z + z_range[0]

    if offset_max_range_um is None:
        dx_range, dy_range, dz_range = zip([np.nanmin(dx_mat), np.nanmin(dy_mat), np.nanmin(dz_mat)], [np.nanmax(dx_mat), np.nanmax(dy_mat), np.nanmax(dz_mat)])
    else:
        dx_range, dy_range, dz_range = get_value_ranges(offset_max_range_um, 3, pos_range=False)

    if np.isscalar(offset_bin_size_um): # Single scalar range value to be used for all dimensions
        log.log_assert(offset_bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dx = bin_size_dy = bin_size_dz = offset_bin_size_um
    else: # Three values for x/y/z dimensions
        log.log_assert(len(offset_bin_size_um) == 3, 'ERROR: Offset bin sizes in x/y/z dimension expected!')
        log.log_assert(np.all([b > 0.0 for b in offset_bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dx, bin_size_dy, bin_size_dz = offset_bin_size_um

    num_bins_dx = np.ceil((dx_range[1] - dx_range[0]) / bin_size_dx).astype(int)
    num_bins_dy = np.ceil((dy_range[1] - dy_range[0]) / bin_size_dy).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dx_bins = np.arange(0, num_bins_dx + 1) * bin_size_dx + dx_range[0]
    dy_bins = np.arange(0, num_bins_dy + 1) * bin_size_dy + dy_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_position, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [x_mat, y_mat, z_mat, dx_mat, dy_mat, dz_mat], [x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins], min_count_per_bin)

    return {'p_conn_position': p_conn_position, 'x_bins': x_bins, 'y_bins': y_bins, 'z_bins': z_bins, 'dx_bins': dx_bins, 'dy_bins': dy_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_5th_order(p_conn_position, x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins, model_specs=None, smoothing_sigma_um=None, **_):
    """Build 5th order model (linear interpolation or random forest regression model for position-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}

    bin_sizes = [ np.diff(x_bins[:2])[0],  np.diff(y_bins[:2])[0],  np.diff(z_bins[:2])[0],
                 np.diff(dx_bins[:2])[0], np.diff(dy_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    x_bin_offset = 0.5 * bin_sizes[0]
    y_bin_offset = 0.5 * bin_sizes[1]
    z_bin_offset = 0.5 * bin_sizes[2]

    x_pos = x_bins[:-1] + x_bin_offset # Positions at bin centers
    y_pos = y_bins[:-1] + y_bin_offset # Positions at bin centers
    z_pos = z_bins[:-1] + z_bin_offset # Positions at bin centers

    dx_bin_offset = 0.5 * bin_sizes[3]
    dy_bin_offset = 0.5 * bin_sizes[4]
    dz_bin_offset = 0.5 * bin_sizes[5]

    dx_pos = dx_bins[:-1] + dx_bin_offset # Positions at bin centers
    dy_pos = dy_bins[:-1] + dy_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_position = p_conn_position.copy()
    p_conn_position[np.isnan(p_conn_position)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 6 # Same value for all coordinates
        else:
            log.log_assert(len(smoothing_sigma_um) == 6, 'ERROR: Smoothing sigma for 6 dimensions required!')
        log.log_assert(np.all(np.array(smoothing_sigma_um) >= 0.0), 'ERROR: Smoothing sigma must be non-negative!')
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.info(f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms')
        p_conn_position = gaussian_filter(p_conn_position, sigmas, mode='constant')

    model_inputs = ['x', 'y', 'z', 'dx', 'dy', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation': # Linear interpolation model => Removing dimensions with only single value from interpolation

        log.log_assert(len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!')

        # Create model
        index = pd.MultiIndex.from_product([x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_position.flatten(), index=index, columns=['p'])
        model = model_types.ConnProb5thOrderLinInterpnModel(p_conn_table=df)

    elif model_specs.get('name') == 'RandomForestRegressor': # Random Forest Regressor model

        log.log_assert(False, 'ERROR: No model class implemented for RandomForestRegressor!')

#         xv, yv, zv, dxv, dyv, dzv = np.meshgrid(x_pos, y_pos, z_pos, dx_pos, dy_pos, dz_pos, indexing='ij')
#         data_pos = np.array([xv.flatten(), yv.flatten(), zv.flatten(), dxv.flatten(), dyv.flatten(), dzv.flatten()]).T
#         data_val = p_conn_position.flatten()

#         position_regr_model = RandomForestRegressor(random_state=0, **model_specs.get('kwargs', {}))
#         position_regr_model.fit(data_pos, data_val)

#         # Create model
#         model = model_types...

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("name")}" unknown!')

    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_5th_order(out_dir, p_conn_position, x_bins, y_bins, z_bins, dx_bins, dy_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, pos_map_file=None, plot_model_ovsampl=3, plot_model_extsn=0, **_):  # pragma: no cover
    """Visualize data vs. model (5th order)."""

    x_bin_offset = 0.5 * np.diff(x_bins[:2])[0]
    y_bin_offset = 0.5 * np.diff(y_bins[:2])[0]
    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]

    x_pos_model = x_bins[:-1] + x_bin_offset # Positions at bin centers
    y_pos_model = y_bins[:-1] + y_bin_offset # Positions at bin centers
    z_pos_model = z_bins[:-1] + z_bin_offset # Positions at bin centers

    dx_bin_offset = 0.5 * np.diff(dx_bins[:2])[0]
    dy_bin_offset = 0.5 * np.diff(dy_bins[:2])[0]
    dz_bin_offset = 0.5 * np.diff(dz_bins[:2])[0]

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1, 'ERROR: Model plot oversampling must be an integer factor >= 1!')
    log.log_assert(isinstance(plot_model_extsn, int) and plot_model_extsn >= 0, 'ERROR: Model plot extension must be an integer number of bins >= 0!')
    dx_bin_size_model = np.diff(dx_bins[:2])[0] / plot_model_ovsampl
    dy_bin_size_model = np.diff(dy_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dx_bins_model = np.arange(dx_bins[0] - plot_model_extsn * dx_bin_size_model * plot_model_ovsampl, dx_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dx_bin_size_model, dx_bin_size_model)
    dy_bins_model = np.arange(dy_bins[0] - plot_model_extsn * dy_bin_size_model * plot_model_ovsampl, dy_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dy_bin_size_model, dy_bin_size_model)
    dz_bins_model = np.arange(dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl, dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model, dz_bin_size_model)

    # Sample positions (at bin centers)
    dx_pos_model = dx_bins_model[:-1] + 0.5 * dx_bin_size_model
    dy_pos_model = dy_bins_model[:-1] + 0.5 * dy_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    xv, yv, zv, dxv, dyv, dzv = np.meshgrid(x_pos_model, y_pos_model, z_pos_model, dx_pos_model, dy_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([xv.flatten(), yv.flatten(), zv.flatten(), dxv.flatten(), dyv.flatten(), dzv.flatten()]).T # Regular grid
    # model_pos = np.random.uniform(low=[x_bins[0], y_bins[0], z_bins[0], dx_bins[0], dy_bins[0], dz_bins[0]], high=[x_bins[-1], y_bins[-1], z_bins[-1], dx_bins[-1], dy_bins[-1], dz_bins[-1]], size=[model_ovsampl**3 * len(x_bins) * len(y_bins) * len(z_bins), len(dx_bins) * len(dy_bins) * len(dz_bins), 3]) # Random sampling
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2], model_pos[:, 3], model_pos[:, 4], model_pos[:, 5])
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
                    plt.plot(dx_bins[p_sel_idx[0]] + dx_bin_offset, dy_bins[p_sel_idx[1]] + dy_bin_offset, dz_bins[p_sel_idx[2]] + dz_bin_offset, 'o', color=p_colors[pidx, :], alpha=0.01 + 0.99 * pidx / (num_p_bins - 1), markeredgecolor='none')
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
                    plt.plot(dx_pos_model[p_sel_idx[0]].T, dy_pos_model[p_sel_idx[1]].T, dz_pos_model[p_sel_idx[2]].T, '.', color=p_colors[pidx, :], alpha=0.01 + 0.99 * pidx / (num_p_bins - 1), markeredgecolor='none')
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
                log.info(f'Saving {out_fn}...')
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
                plt.imshow(np.max(p_conn_sel, 1).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(p_conn_sel, 1)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.subplot(2, 3, 2)
                plt.imshow(np.max(p_conn_sel, 0).T, interpolation='none', extent=(dy_bins[0], dy_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(p_conn_sel, 0)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$y [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')
                plt.title('Data')

                plt.subplot(2, 3, 3)
                plt.imshow(np.max(p_conn_sel, 2).T, interpolation='none', extent=(dx_bins[0], dx_bins[-1], dy_bins[-1], dy_bins[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(p_conn_sel, 2)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$y [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                # (Model)
                plt.subplot(2, 3, 4)
                plt.imshow(np.max(model_val_sel, 1).T, interpolation='none', extent=(dx_bins_model[0], dx_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(model_val_sel, 1)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.subplot(2, 3, 5)
                plt.imshow(np.max(model_val_sel, 0).T, interpolation='none', extent=(dy_bins_model[0], dy_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(model_val_sel, 0)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$y [$\\mu$m]')
                plt.ylabel('$\\Delta$z [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')
                plt.title('Model')

                plt.subplot(2, 3, 6)
                plt.imshow(np.max(model_val_sel, 2).T, interpolation='none', extent=(dx_bins_model[0], dx_bins_model[-1], dy_bins_model[-1], dy_bins_model[0]), cmap=HOT, vmin=0.0, vmax=0.1 if np.max(np.max(model_val_sel, 2)) == 0.0 else None)
                plt.plot(plt.xlim(), np.zeros(2), 'w', linewidth=0.5)
                plt.plot(np.zeros(2), plt.ylim(), 'w', linewidth=0.5)
                plt.gca().invert_yaxis()
                plt.xlabel('$\\Delta$x [$\\mu$m]')
                plt.ylabel('$\\Delta$y [$\\mu$m]')
                plt.colorbar(label='Max. conn. prob.')

                plt.suptitle(f'Position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>\nX={x_pos_model[ix]:.0f}$\\mu$m, Y={y_pos_model[iy]:.0f}$\\mu$m, Z={z_pos_model[iz]:.0f}$\\mu$m')
                plt.tight_layout()
                out_fn = os.path.abspath(os.path.join(out_dir, f'data_vs_model_2d_x{ix}y{iy}z{iz}.png'))
                log.info(f'Saving {out_fn}...')
                plt.savefig(out_fn)


###################################################################################################
# Generative models for circuit connectivity:
#   Reduced 5th order (position-dependent), modified from [Gal et al. 2020]
#     => Axial position only
#     => Radial/axial offsets only
#     => Position mapping model (flatmap) supported
#     => model_specs with 'name' (e.g., 'LinearInterpolation', 'RandomForestRegressor')
#                    and optionally, 'kwargs' may be provided
###################################################################################################

def extract_5th_order_reduced(nodes, edges, src_node_ids, tgt_node_ids, position_bin_size_um=1000, position_max_range_um=None, offset_bin_size_um=100, offset_max_range_um=None, pos_map_file=None, plot_model_ovsampl=3, plot_model_extsn=0, min_count_per_bin=10, **_):
    """Extract position-dependent connection probability (5th order reduced) from a sample of pairs of neurons."""
    # Get neuron positions (incl. position mapping, if provided)
    _, pos_acc = load_pos_mapping_model(pos_map_file)
    src_nrn_pos, tgt_nrn_pos = get_neuron_positions([n.positions for n in nodes] if pos_acc is None else pos_acc, [src_node_ids, tgt_node_ids])

    # Compute PRE position & POST-PRE offset matrices
    z_mat = model_types.ConnProb5thOrderLinInterpnReducedModel.compute_position_matrix(src_nrn_pos, tgt_nrn_pos)
    dr_mat, dz_mat = model_types.ConnProb5thOrderLinInterpnReducedModel.compute_offset_matrices(src_nrn_pos, tgt_nrn_pos)

    # Extract position- & offset-dependent connection probabilities
    if position_max_range_um is None:
        z_range = [np.minimum(np.nanmin(src_nrn_pos[:, 2]), np.nanmin(tgt_nrn_pos[:, 2])), np.maximum(np.nanmax(src_nrn_pos[:, 2]), np.nanmax(tgt_nrn_pos[:, 2]))]
    else:
        z_range = get_value_ranges(position_max_range_um, 1, pos_range=False)

    log.log_assert(np.isscalar(position_bin_size_um) and position_bin_size_um > 0.0, 'ERROR: Position bin size must be a scalar larger than 0um!')
    bin_size_z = position_bin_size_um
    num_bins_z = np.ceil((z_range[1] - z_range[0]) / bin_size_z).astype(int)
    z_bins = np.arange(0, num_bins_z + 1) * bin_size_z + z_range[0]

    if offset_max_range_um is None:
        dr_range, dz_range = zip([np.nanmin(dr_mat), np.nanmin(dz_mat)], [np.nanmax(dr_mat), np.nanmax(dz_mat)])
    else:
        dr_range, dz_range = get_value_ranges(offset_max_range_um, 2, pos_range=[True, False])

    if np.isscalar(offset_bin_size_um): # Single scalar range value to be used for all dimensions
        log.log_assert(offset_bin_size_um > 0.0, 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dr = bin_size_dz = offset_bin_size_um
    else: # Two values for r/z dimensions
        log.log_assert(len(offset_bin_size_um) == 2, 'ERROR: Offset bin sizes in r/z directions expected!')
        log.log_assert(np.all([b > 0.0 for b in offset_bin_size_um]), 'ERROR: Offset bin size must be larger than 0um!')
        bin_size_dr, bin_size_dz = offset_bin_size_um

    num_bins_dr = np.ceil((dr_range[1] - dr_range[0]) / bin_size_dr).astype(int)
    num_bins_dz = np.ceil((dz_range[1] - dz_range[0]) / bin_size_dz).astype(int)

    dr_bins = np.arange(0, num_bins_dr + 1) * bin_size_dr + dr_range[0]
    dz_bins = np.arange(0, num_bins_dz + 1) * bin_size_dz + dz_range[0]

    p_conn_position, _, _ = extract_dependent_p_conn(src_node_ids, tgt_node_ids, edges, [z_mat, dr_mat, dz_mat], [z_bins, dr_bins, dz_bins], min_count_per_bin)

    return {'p_conn_position': p_conn_position, 'z_bins': z_bins, 'dr_bins': dr_bins, 'dz_bins': dz_bins, 'src_cell_count': len(src_node_ids), 'tgt_cell_count': len(tgt_node_ids)}


def build_5th_order_reduced(p_conn_position, z_bins, dr_bins, dz_bins, model_specs=None, smoothing_sigma_um=None, **_):
    """Build reduced 5th order model (linear interpolation or random forest regression model for position-dependent conn. prob.)."""
    if model_specs is None:
        model_specs = {'name': 'LinearInterpolation'}

    bin_sizes = [np.diff(z_bins[:2])[0], np.diff(dr_bins[:2])[0], np.diff(dz_bins[:2])[0]]

    z_bin_offset = 0.5 * bin_sizes[0]
    z_pos = z_bins[:-1] + z_bin_offset # Positions at bin centers

    dr_bin_offset = 0.5 * bin_sizes[1]
    dz_bin_offset = 0.5 * bin_sizes[2]

    dr_pos = dr_bins[:-1] + dr_bin_offset # Positions at bin centers
    dz_pos = dz_bins[:-1] + dz_bin_offset # Positions at bin centers

    # Set NaNs to zero (for smoothing/interpolation)
    p_conn_position = p_conn_position.copy()
    p_conn_position[np.isnan(p_conn_position)] = 0.0

    # Apply Gaussian smoothing filter to data points (optional)
    if smoothing_sigma_um is not None:
        if not isinstance(smoothing_sigma_um, list):
            smoothing_sigma_um = [smoothing_sigma_um] * 3 # Same value for all coordinates
        else:
            log.log_assert(len(smoothing_sigma_um) == 3, 'ERROR: Smoothing sigma for 3 dimensions required!')
        log.log_assert(np.all(np.array(smoothing_sigma_um) >= 0.0), 'ERROR: Smoothing sigma must be non-negative!')
        sigmas = [sig / b for sig, b in zip(smoothing_sigma_um, bin_sizes)]
        log.info(f'Applying data smoothing with sigma {"/".join([str(sig) for sig in smoothing_sigma_um])}ms')
        p_conn_position = gaussian_filter(p_conn_position, sigmas, mode='constant')

    model_inputs = ['z', 'dr', 'dz'] # Must be the same for all interpolation types!
    if model_specs.get('name') == 'LinearInterpolation': # Linear interpolation model => Removing dimensions with only single value from interpolation

        log.log_assert(len(model_specs.get('kwargs', {})) == 0, f'ERROR: No parameters expected for "{model_specs.get("name")}" model!')

        # Create model
        index = pd.MultiIndex.from_product([z_pos, dr_pos, dz_pos], names=model_inputs)
        df = pd.DataFrame(p_conn_position.flatten(), index=index, columns=['p'])
        model = model_types.ConnProb5thOrderLinInterpnReducedModel(p_conn_table=df)

    elif model_specs.get('name') == 'RandomForestRegressor': # Random Forest Regressor model
        log.log_assert(False, 'ERROR: No model class implemented for RandomForestRegressor!')

    else:
        log.log_assert(False, f'ERROR: Model type "{model_specs.get("name")}" unknown!')

    log.info('Model description:\n' + model.get_model_str())

    return model


def plot_5th_order_reduced(out_dir, p_conn_position, z_bins, dr_bins, dz_bins, src_cell_count, tgt_cell_count, model_specs, model, pos_map_file=None, plot_model_ovsampl=3, plot_model_extsn=0, **_):  # pragma: no cover
    """Visualize data vs. model (5th order reduced)."""

    z_bin_offset = 0.5 * np.diff(z_bins[:2])[0]
    z_pos_model = z_bins[:-1] + z_bin_offset # Positions at bin centers

    # Oversampled bins for model plotting, incl. plot extension (#bins of original size) in each direction
    log.log_assert(isinstance(plot_model_ovsampl, int) and plot_model_ovsampl >= 1, 'ERROR: Model plot oversampling must be an integer factor >= 1!')
    log.log_assert(isinstance(plot_model_extsn, int) and plot_model_extsn >= 0, 'ERROR: Model plot extension must be an integer number of bins >= 0!')
    dr_bin_size_model = np.diff(dr_bins[:2])[0] / plot_model_ovsampl
    dz_bin_size_model = np.diff(dz_bins[:2])[0] / plot_model_ovsampl
    dr_bins_model = np.arange(dr_bins[0], dr_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dr_bin_size_model, dr_bin_size_model)
    dz_bins_model = np.arange(dz_bins[0] - plot_model_extsn * dz_bin_size_model * plot_model_ovsampl, dz_bins[-1] + (1 + plot_model_extsn * plot_model_ovsampl) * dz_bin_size_model, dz_bin_size_model)

    # Sample positions (at bin centers)
    dr_pos_model = dr_bins_model[:-1] + 0.5 * dr_bin_size_model
    dz_pos_model = dz_bins_model[:-1] + 0.5 * dz_bin_size_model

    # Model probability at sample positions
    zv, drv, dzv = np.meshgrid(z_pos_model, dr_pos_model, dz_pos_model, indexing='ij')
    model_pos = np.array([zv.flatten(), drv.flatten(), dzv.flatten()]).T # Regular grid
    model_val = model.get_conn_prob(model_pos[:, 0], model_pos[:, 1], model_pos[:, 2])
    model_val = model_val.reshape([len(z_pos_model), len(dr_pos_model), len(dz_pos_model)])

    # Connection probability (data vs. model)
    p_max = np.max(p_conn_position)
    p_max_model = np.max(model_val)
    fig = plt.figure(figsize=(12, 4 * len(z_pos_model)), dpi=300)
    for zidx, zval in enumerate(z_pos_model):
        # (Data)
        log.log_assert(dr_bins[0] == 0, 'ERROR: Radial bin range error!')
        plt.subplot(len(z_pos_model), 2, zidx * 2 + 1)
        plt.imshow(np.hstack([np.squeeze(p_conn_position[zidx, ::-1, :]).T, np.squeeze(p_conn_position[zidx, :, :]).T]), interpolation='nearest', extent=(-dr_bins[-1], dr_bins[-1], dz_bins[-1], dz_bins[0]), cmap=HOT, vmin=0.0, vmax=p_max)
        plt.plot(np.zeros(2), plt.ylim(), color='lightgrey', linewidth=0.5)
        plt.text(np.min(plt.xlim()), np.max(plt.ylim()), f'z={zval}um', color='lightgrey', ha='left', va='top')
        plt.gca().invert_yaxis()
        plt.xlabel('$\\Delta$r [$\\mu$m]')
        plt.ylabel('$\\Delta$z [$\\mu$m]')
        plt.colorbar(label='Conn. prob.')
        if zidx == 0:
            plt.title(f'Data: N = {src_cell_count}x{tgt_cell_count} cells')

        # (Model)
        plt.subplot(len(z_pos_model), 2, zidx * 2 + 2)
        plt.imshow(np.hstack([np.squeeze(model_val[zidx, ::-1, :]).T, np.squeeze(model_val[zidx, :, :]).T]), interpolation='nearest', extent=(-dr_bins_model[-1], dr_bins_model[-1], dz_bins_model[-1], dz_bins_model[0]), cmap=HOT, vmin=0.0, vmax=p_max_model)
        plt.plot(np.zeros(2), plt.ylim(), color='lightgrey', linewidth=0.5)
        plt.text(np.min(plt.xlim()), np.max(plt.ylim()), f'z={zval}um', color='lightgrey', ha='left', va='top')
        plt.gca().invert_yaxis()
        plt.xlabel('$\\Delta$r [$\\mu$m]')
        plt.ylabel('$\\Delta$z [$\\mu$m]')
        plt.colorbar(label='Conn. prob.')
        if zidx == 0:
            plt.title(f'Model: {model_specs.get("name")}')

    plt.suptitle(f'Reduced position-dependent connection probability model (5th order)\n<Position mapping: {pos_map_file}>')
    plt.tight_layout()
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model.png'))
    log.info(f'Saving {out_fn}...')
    plt.savefig(out_fn)
