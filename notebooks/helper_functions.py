# Helper functions

import os.path
import numpy
import pandas
import matplotlib.pyplot as plt

""" Combined plotting of all topological parameters in one figure with num_rows rows """
""" [Modified from plot_comparison(...) in topological_comparator/bin/compare_topo_db.py] """
def topocomp_plot_comparison_combined(db_dict, param_dict, groupby, out_dir, num_rows=1, show_fig=False):
    
    cname1, cname2 = list(db_dict.keys())
    out_dir = os.path.join(out_dir, cname1 + "_vs_" + cname2)
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
    
    num_cols = numpy.ceil(len(param_dict) / num_rows).astype(int)
    figsize_x = 3 * num_cols
    figsize_y = num_rows * (4 + len(full_db[groupby].drop_duplicates()) / 3)
    plt.figure(figsize=(figsize_x, figsize_y))
    for param_idx, (param_name, param_spec) in enumerate(param_dict.items()):
        try:
            ax = plt.subplot(num_rows, num_cols, param_idx + 1)
            plot_frame = full_db[['Circuit', groupby]].copy()
            plot_frame[param_name] = get_column_from_database(full_db, param_spec["column"],
                                                              index=param_spec.get("index", None),
                                                              function=param_spec.get("function", None))
            sns.violinplot(y=groupby, x=param_name, hue='Circuit', split=True, data=plot_frame, ax=ax, orient='h')
            if param_idx % num_cols != 0:
                ax.set_ylabel(None)
            ax.legend(loc='lower right', fontsize=6, bbox_to_anchor=(1.0, 1.0))
        except:
            print("Error occured when trying to compare {0}".format(param_name))
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, "topo_comp-all_params" + (("-per_" + groupby) if groupby else "") + ".pdf"))
    print("Saving {0}...".format(out_fn))
    plt.gcf().savefig(out_fn)
    if show_fig:
        plt.show()
    else:
        plt.close("all")