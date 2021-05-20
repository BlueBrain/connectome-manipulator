# Structural comparison function
#
# Two functions need to be defined 
# (1) compute(circuit, ...):
#     - The first parameter is always: circuit
#     - Other parameters may be added (optional)
#     - Returns a dict with different results (each containing data, name, unit) and common information needed for plotting
# (2) plot(res_dict, common_dict, fig_title=None, vmin=None, vmax=None):
#     - The first two parameters are always: res_dict...one selected results dictionary (in case of more than one) returned by compute()
#                                            common_dict...dictionary with common properties/results returned by compute()
#     -fig_title, vmin, vmax: optional parameters to control parameters across subplots
# Comment: For performance reasons, different related results can be computed in one computation run and returned/saved together.
#          They can then be plotted separately one at a time by specifying which of them to plot.

import progressbar
import numpy as np
import matplotlib.pyplot as plt

""" Extract adjacency and count matrices """
def compute(circuit, nrn_filter=None, **_):
    
    # Select edge population [assuming exactly one edge population in given edges file]
    assert len(circuit.edges.population_names) == 1, 'ERROR: Only a single edge population per file supported!'
    edges = circuit.edges[circuit.edges.population_names[0]]
    
    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    
    src_node_ids = src_nodes.ids(nrn_filter)
    tgt_node_ids = tgt_nodes.ids(nrn_filter)
    
    # Map source/target node ids to continuous range of indices for plotting
    src_gid_min = min(src_nodes.ids())
    src_gid_max = max(src_nodes.ids())
    tgt_gid_min = min(tgt_nodes.ids())
    tgt_gid_max = max(tgt_nodes.ids())
    
    src_plot_ids = np.full(src_gid_max - src_gid_min + 1, -1).astype(int)
    src_gid_offset = src_gid_min
    src_plot_ids[src_node_ids - src_gid_offset] = np.arange(len(src_node_ids))
    def src_gid_to_idx(gids):
        return src_plot_ids[gids - src_gid_offset]
    
    tgt_plot_ids = np.full(tgt_gid_max - tgt_gid_min + 1, -1).astype(int)
    tgt_gid_offset = tgt_gid_min
    tgt_plot_ids[tgt_node_ids - tgt_gid_offset] = np.arange(len(tgt_node_ids))
    def tgt_gid_to_idx(gids):
        return tgt_plot_ids[gids - tgt_gid_offset]
    
    print(f'INFO: Creating {len(src_node_ids)}x{len(tgt_node_ids)} adjacency matrix (nrn_filter={nrn_filter})', flush=True)
    
    count_matrix = np.zeros((len(src_node_ids), len(tgt_node_ids)), dtype=int)
#     pbar = progressbar.ProgressBar()
#     for post_idx in pbar(range(len(tgt_node_ids))):
#         post_gid = tgt_node_ids[post_idx]
#         conns = np.array(list(edges.iter_connections(target=post_gid, return_edge_count=True)))
#         if len(conns) > 0:
#             idx = src_gid_to_idx(conns[:, 0])
#             count_matrix[idx[idx >= 0], tgt_gid_to_idx(post_gid)] = conns[idx >= 0, 2]
    
    conns = np.array(list(edges.iter_connections(source=src_node_ids, target=tgt_node_ids, return_edge_count=True)))
    count_matrix[src_gid_to_idx(conns[:, 0]), tgt_gid_to_idx(conns[:, 1])] = conns[:, 2]
    
    adj_matrix = count_matrix > 0
    
    return {'adj': {'data': adj_matrix, 'name': 'Adjacency', 'unit': None},
            'adj_cnt': {'data': count_matrix, 'name': 'Adjacency count', 'unit': 'Synapse count'},
            'common': {'src_gids': src_node_ids, 'tgt_gids': tgt_node_ids}}


""" Plot adjacency matrix [NOT using imshow causing display errors] """
def plot(res_dict, common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, **_):
    
    if isdiff: # Difference plot
        assert -vmin == vmax, 'ERROR: Symmetric plot range required!'
        cmap = 'PiYG' # Symmetric (diverging) colormap
    else: # Regular plot
        assert vmin == 0, 'ERROR: Plot range including 0 required!'
        cmap = 'hot_r' # Regular colormap [color at 0 should be white (not actually drawn), to match figure background!]
    
    conns = np.array(np.where(res_dict['data'])).T
    col_idx = res_dict['data'][conns[:, 0], conns[:, 1]]        
    plt.scatter(conns[:, 1], conns[:, 0], marker=',', s=0.1, edgecolors='none', alpha=0.5, c=col_idx, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if not res_dict['data'].dtype == bool:
        cb = plt.colorbar()
        cb.set_label(res_dict['unit'])
    
    if fig_title is None:
        plt.title(res_dict['name'])
    else:
        plt.title(fig_title)
    
    plt.xlabel(f'Postsynaptic neurons')
    plt.ylabel(f'Presynaptic neurons')
    
    plt.axis('image')
    plt.xlim((-0.5, res_dict['data'].shape[1] - 0.5))
    plt.ylim((-0.5, res_dict['data'].shape[0] - 0.5))
    plt.gca().invert_yaxis()
