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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

""" Compute mean/std/... values of all synapse properties grouped by given cell property """
def compute(circuit, fct='np.mean', group_by=None, sel_src=None, sel_dest=None, per_conn=False, **_):
    
    # Select edge population [assuming exactly one edge population in given edges file]
    assert len(circuit.edges.population_names) == 1, 'ERROR: Only a single edge population per file supported!'
    edges = circuit.edges[circuit.edges.population_names[0]]
    
    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target
    
    if group_by is None:
        group_values = ['Overall']
        src_group_sel = [sel_src]
        tgt_group_sel = [sel_dest]
    else:
        if sel_src is None:
            sel_src = {}
        else:
            assert isinstance(sel_src, dict), 'ERROR: Source node selection must be a dict or empty!' # Otherwise, it cannot be merged with group selection
        if sel_dest is None:
            sel_dest = {}
        else:
            assert isinstance(sel_dest, dict), 'ERROR: Target node selection must be a dict or empty!' # Otherwise, it cannot be merged with pathway selection
        src_group_values = sorted(src_nodes.property_values(group_by))
        src_group_sel = [{group_by: src_group_values[idx], **sel_src} for idx in range(len(src_group_values))]
        tgt_group_values = sorted(tgt_nodes.property_values(group_by))
        tgt_group_sel = [{group_by: tgt_group_values[idx], **sel_dest} for idx in range(len(tgt_group_values))]
    
    print(f'INFO: Extracting synapse properties (group_by={group_by}, sel_src={sel_src}, sel_dest={sel_dest}, N={len(src_group_values)}x{len(tgt_group_values)} groups, per_conn={per_conn})', flush=True)
    
    edge_props = sorted(edges.property_names)
    print(f'INFO: Available synapse properties: \n{edge_props}', flush=True)
    
    prop_fct = eval(fct)
    prop_tables = np.full((len(src_group_sel), len(tgt_group_sel), len(edge_props)), np.nan)
    pbar = progressbar.ProgressBar()
    for idx_pre in pbar(range(len(src_group_sel))):
        sel_pre = src_group_sel[idx_pre]
        for idx_post in range(len(tgt_group_sel)):
            sel_post = tgt_group_sel[idx_post]
            e_sel = edges.pathway_edges(sel_pre, sel_post, edge_props)
            if e_sel.size > 0:
                if per_conn: # Apply prop_fct to average value per connection
                    conn, conn_idx = np.unique(e_sel[['@source_node', '@target_node']], axis=0, return_inverse=True)
                    c_sel = pd.DataFrame(index=range(conn.shape[0]), columns=edge_props)
                    for cidx in range(conn.shape[0]):
                        c_sel.loc[cidx, :] = np.mean(e_sel[conn_idx == cidx], axis=0)
                    prop_tables[idx_pre, idx_post, :] = prop_fct(c_sel.to_numpy(), axis=0)
                else:
                    prop_tables[idx_pre, idx_post, :] = prop_fct(e_sel.to_numpy(), axis=0)
    
    fname = prop_fct.__name__[0].upper() + prop_fct.__name__[1:]
    cname = ' (per conn)' if per_conn else ''
    res_dict = {edge_props[idx]: {'data': prop_tables[:, :, idx], 'name': f'"{edge_props[idx]}" property', 'unit': f'{fname} {edge_props[idx]}{cname}'} for idx in range(len(edge_props))}
    res_dict['common'] = {'src_group_values': src_group_values, 'tgt_group_values': tgt_group_values}
    
    return res_dict


""" Connectivity (matrix) plotting """
def plot(res_dict, common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, group_by=None, **_):
    
    if isdiff: # Difference plot
        assert -vmin == vmax, 'ERROR: Symmetric plot range required!'
        cmap = 'PiYG' # Symmetric (diverging) colormap
    else: # Regular plot
        cmap = 'hot_r' # Regular colormap
    
    plt.imshow(res_dict['data'], interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if fig_title is None:
        plt.title(res_dict['name'])
    else:
        plt.title(fig_title)
    
    if group_by:
        plt.xlabel(f'Postsynaptic {group_by}')
        plt.ylabel(f'Presynaptic {group_by}')
    
    if len(common_dict['src_group_values']) > 0:
        font_size = max(13 - len(common_dict['src_group_values']) / 6, 1) # Font scaling
        plt.yticks(range(len(common_dict['src_group_values'])), common_dict['src_group_values'], rotation=0, fontsize=font_size)
    
    if len(common_dict['tgt_group_values']) > 0:
        if max([len(str(grp)) for grp in common_dict['tgt_group_values']]) > 1:
            rot_x = 90
        else:
            rot_x = 0
        font_size = max(13 - len(common_dict['tgt_group_values']) / 6, 1) # Font scaling
        plt.xticks(range(len(common_dict['tgt_group_values'])), common_dict['tgt_group_values'], rotation=rot_x, fontsize=font_size)
    
    cb = plt.colorbar()
    cb.set_label(res_dict['unit'])
