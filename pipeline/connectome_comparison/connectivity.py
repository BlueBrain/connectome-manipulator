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
from matplotlib import colors

""" Compute connectivity grouped by given property """
def compute(circuit, group_by=None, nrn_filter=None, **_):
    
    if group_by is None:
        group_values = ['Overall']
        if isinstance(nrn_filter, dict):
            group_sel = [nrn_filter]
        else:
            group_sel = ['All']
    else:
        group_values = sorted(circuit.nodes['All'].property_values(group_by))
        group_sel = [{group_by: group_values[idx]} for idx in range(len(group_values))]
        if isinstance(nrn_filter, dict):
            print(f'INFO: Applying neuron filter {nrn_filter}', flush=True)
            assert group_by not in nrn_filter.keys(), 'ERROR: Group/filter selection mismatch!'
            for idx in range(len(group_sel)):
                group_sel[idx].update(nrn_filter)
    
    print(f'INFO: Computing connectivity (group_by={group_by}, nrn_filter={nrn_filter}, N={len(group_values)})', flush=True)
    
    syn_table = np.zeros((len(group_sel), len(group_sel)))
    p_table = np.zeros((len(group_sel), len(group_sel)))
    pbar = progressbar.ProgressBar()
    for idx_pre in pbar(range(len(group_sel))):
        sel_pre = group_sel[idx_pre]
        for idx_post, type_post in enumerate(group_sel):
            sel_post = group_sel[idx_post]
            it_conn = circuit.edges['default'].iter_connections(sel_pre, sel_post, return_edge_count=True)
            conns = np.array(list(it_conn))
            
            if conns.size > 0:
                scounts = conns[:, 2] # Synapse counts per connection
                ccount = len(scounts) # Connection count
                pre_count = len(circuit.nodes['All'].ids(sel_pre))
                post_count = len(circuit.nodes['All'].ids(sel_post))
                
                syn_table[idx_pre, idx_post] = np.mean(scounts)
                p_table[idx_pre, idx_post] = 100.0 * ccount / (pre_count * post_count)
    
    syn_table_name = 'Synapses per connection'
    syn_table_unit = 'Mean #syn/conn'
    p_table_name = 'Connection probability'
    p_table_unit = 'Conn. prob. (%)'
    
    return {'nsyn_conn': {'data': syn_table, 'name': syn_table_name, 'unit': syn_table_unit},
            'conn_prob': {'data': p_table, 'name': p_table_name, 'unit': p_table_unit},
            'common': {'group_values': group_values}}


""" Connectivity (matrix) plotting """
def plot(res_dict, common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, group_by=None, **_):
    
    if isdiff: # Difference plot
        assert -vmin == vmax, 'ERROR: Symmetric plot range required!'
        cmap = 'PiYG' # Symmetric (diverging) colormap
    else: # Regular plot
        cmap = 'jet' # Regular colormap
    
    plt.imshow(res_dict['data'], interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        
    if fig_title is None:
        plt.title(res_dict['name'])
    else:
        plt.title(fig_title)
    
    if group_by:
        plt.xlabel(f'Postsynaptic {group_by}')
        plt.ylabel(f'Presynaptic {group_by}')
    
    if len(common_dict['group_values']) > 0:
        if max([len(str(grp)) for grp in common_dict['group_values']]) > 1:
            rot_x = 90
        else:
            rot_x = 0
        font_size = max(13 - len(res_dict['data']) / 6, 1) # Font scaling
        
        plt.xticks(range(len(common_dict['group_values'])), common_dict['group_values'], rotation=rot_x, fontsize=font_size)
        plt.yticks(range(len(common_dict['group_values'])), common_dict['group_values'], rotation=0, fontsize=font_size)
    
    cb = plt.colorbar()
    cb.set_label(res_dict['unit'])
