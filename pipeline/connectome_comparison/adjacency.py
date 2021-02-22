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
    
    all_node_ids = circuit.nodes['All'].ids()
    node_ids = circuit.nodes['All'].ids(nrn_filter)

    # Map node ids to continuous range of indices for plotting
    gid_min = min(all_node_ids)
    gid_max = max(all_node_ids)
    
    plot_ids = np.full(gid_max - gid_min + 1, -1).astype(int)
    gid_offset = gid_min
    plot_ids[node_ids - gid_offset] = np.arange(len(node_ids))
    def gid_to_idx(gids):
        return plot_ids[gids - gid_offset]
    
    print(f'INFO: Creating adjacency matrix (nrn_filter={nrn_filter})', flush=True)
    
    count_matrix = np.zeros([len(node_ids)] * 2).astype(int)
    pbar = progressbar.ProgressBar()
    for pre_idx in pbar(range(len(node_ids))):
        
        pre_gid = node_ids[pre_idx]
        conns = np.array(list(circuit.edges['default'].iter_connections(pre_gid, return_edge_count=True)))
        if len(conns) > 0:
            gid_to_idx(conns[:, 1])
            idx = gid_to_idx(conns[:, 1])
            count_matrix[gid_to_idx(pre_gid), idx[idx >= 0]] = conns[idx >= 0, 2] # Filter selected gids here [faster than selecting post GIDs within iter_connections]
    
    adj_matrix = count_matrix > 0
    
    return {'adj': {'data': adj_matrix, 'name': 'Adjacency', 'unit': None},
            'adj_cnt': {'data': count_matrix, 'name': 'Adjacency count', 'unit': 'Synapse count'},
            'common': {'gids': node_ids}}


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
