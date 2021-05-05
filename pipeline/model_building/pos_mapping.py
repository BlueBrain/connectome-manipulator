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
import pandas as pd
import matplotlib.pyplot as plt
from voxcell.nexus.voxelbrain import Atlas
from scipy.interpolate import griddata
from scipy.spatial import distance_matrix

""" Extract position mapping from atlas space to flat space (2 files required: 1. xy mapping, 2. z (=depth) mapping) """
def extract(circuit, flatmap_path, xy_file, z_file, xy_scale=None, z_scale=None, **_):
    
    # Get neuron positions
    nodes = circuit.nodes['All']
    nrn_pos = nodes.positions()
    nrn_ids = nrn_pos.index.to_numpy()
    nrn_pos = nrn_pos.to_numpy()
    nrn_lay = nodes.get(properties='layer')
    
    # Load flatmap
    flatmap_atlas = Atlas.open(flatmap_path)
    flatmap = flatmap_atlas.load_data(xy_file)
    depths = flatmap_atlas.load_data(z_file)
    assert flatmap.raw.shape[:3] == depths.raw.shape, 'ERROR: Flatmap and depths map sizes inconsistent!'
    assert np.all(flatmap.voxel_dimensions == depths.voxel_dimensions), 'ERROR: Flatmap and depths map voxels inconsistent!'
    assert np.all(flatmap.bbox == depths.bbox), 'ERROR: Flatmap and depths map bounding boxes inconsistent!'
    
    if xy_scale is None: # x/y scaling from a.u. to um
        xy_scale = [flatmap.voxel_dimensions[0], flatmap.voxel_dimensions[1]] # Default: Assume same pixel size in flat space as voxel size in atlas
    else:
        assert np.array(xy_scale).size == 2 and np.all(np.isfinite(xy_scale)) and np.all(np.array(xy_scale) != 0), 'ERROR: XY scale error!'
    
    if z_scale is None: # z scaling from a.u. to um
        z_scale = 1.0 # Default: Assume depth values are already scaled to um
    else:
        assert np.array(z_scale).size == 1 and np.isfinite(z_scale) and z_scale != 0, 'ERROR: Z scale error!'
    
    print(f'INFO: Loaded x/y flatmap ("{xy_file}"; scale={np.round(xy_scale, decimals=2)}) and z (depth) map ("{z_file}"; scale={np.round(z_scale, decimals=2)}) from "{flatmap_path}"')
    
    # Convert cell positions to flat space [Assume: missing values set to -1]
    flat_x = flatmap.lookup(nrn_pos)[:, 0].astype(float)
    flat_x[flat_x != -1] = flat_x[flat_x != -1] * xy_scale[0]
    flat_y = flatmap.lookup(nrn_pos)[:, 1].astype(float)
    flat_y[flat_y != -1] = flat_y[flat_y != -1] * xy_scale[1]
    flat_z = depths.lookup(nrn_pos).astype(float)
    flat_z[flat_z != -1] = flat_z[flat_z != -1] * z_scale
    
    # Determine map indices/positions
    map_indices = flatmap.positions_to_indices(nrn_pos, keep_fraction=True) # Keep fractions within voxels => fraction x.5 corresponds to voxel center
    map_pos = np.floor(map_indices) + 0.5 # Voxel values assumed to correspond to voxel center
    
    # Linear interpolation, if possible
    flat_x_intpl = griddata(map_pos[flat_x != -1], flat_x[flat_x != -1], map_indices)
    flat_y_intpl = griddata(map_pos[flat_y != -1], flat_y[flat_y != -1], map_indices)
    flat_z_intpl = griddata(map_pos[flat_z != -1], flat_z[flat_z != -1], map_indices)
    
    # Nearest-neighbor interpolation, otherwise
    flat_x_intpl[np.isnan(flat_x_intpl)] = griddata(map_pos[flat_x != -1], flat_x[flat_x != -1], map_indices[np.isnan(flat_x_intpl)], method='nearest')
    flat_y_intpl[np.isnan(flat_y_intpl)] = griddata(map_pos[flat_y != -1], flat_y[flat_y != -1], map_indices[np.isnan(flat_y_intpl)], method='nearest')
    flat_z_intpl[np.isnan(flat_z_intpl)] = griddata(map_pos[flat_z != -1], flat_z[flat_z != -1], map_indices[np.isnan(flat_z_intpl)], method='nearest')
    
    flat_pos = np.vstack((flat_x_intpl, flat_y_intpl, flat_z_intpl)).T
    
    return {'nrn_ids': nrn_ids, 'nrn_lay': nrn_lay, 'nrn_pos': nrn_pos, 'flat_pos': flat_pos}


""" Build flat space position mapping model """
def build(nrn_ids, flat_pos, **_):
        
    flat_pos_model = pd.DataFrame(flat_pos, index=nrn_ids, columns=['x', 'y', 'z'])
    assert np.all(np.isfinite(flat_pos_model)), 'ERROR: Flatmap interpolation error!'
    
    print(f'POSITION MODEL: flat space, {len(nrn_ids)} cells, x/y/z dimensions {flat_pos_model.x.max()-flat_pos_model.x.min():.1f} x {flat_pos_model.y.max()-flat_pos_model.y.min():.1f} x {flat_pos_model.z.max()-flat_pos_model.z.min():.1f}um')
    
    return {'model': 'flat_pos_model.loc[gids].to_numpy() if not gids is None else flat_pos_model.index.to_numpy()',
            'model_inputs': ['gids'],
            'model_params': {'flat_pos_model': flat_pos_model}}


""" Visualize data vs. model """
def plot(out_dir, nrn_ids, nrn_lay, nrn_pos, model, model_inputs, model_params, **_):
    
    model_fct = model_building.get_model(model, model_inputs, model_params)
    nrn_pos_model = model_fct(nrn_ids)
    
    # 3D cell positions in atlas vs. flat space
    num_layers = len(np.unique(nrn_lay))
    lay_colors = plt.cm.jet(np.linspace(0, 1, num_layers))
    views = [[90, 0], [0, 0]]
    pos_list = [nrn_pos, nrn_pos_model]
    lbl_list = ['Atlas space (data)', 'Flat space (model)']
    fig = plt.figure(figsize=(10, 3 * len(views)), dpi=300)
    plt.gcf().patch.set_facecolor('w')
    for vidx, v in enumerate(views):
        for pidx, (pos, lbl) in enumerate(zip(pos_list, lbl_list)):
            ax = fig.add_subplot(len(views), len(pos_list), vidx * len(pos_list) + pidx + 1, projection='3d')
            for lidx in range(num_layers):
                pos_sel = pos[nrn_lay == lidx + 1, :]
                plt.plot(pos_sel[:, 0], pos_sel[:, 1], pos_sel[:, 2], '.', color=lay_colors[lidx, :], markersize=1.0, alpha=0.5, label=f'L{lidx + 1}')
            ax.view_init(*v)
            ax.set_xlabel('x [$\mu$m]')
            ax.set_ylabel('y [$\mu$m]')
            ax.set_zlabel('z [$\mu$m]')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
            if vidx == 0:
                plt.title(lbl + f'\n[N={len(nrn_ids)}cells]')
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_positions.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    # Cell distances in atlas vs. flat space
    dist_mat_data = distance_matrix(nrn_pos, nrn_pos)
    dist_mat_model = distance_matrix(nrn_pos_model, nrn_pos_model)
    
    triu_idx = np.triu_indices(len(nrn_ids), 1)
    dist_val_data = dist_mat_data[triu_idx]
    dist_val_model = dist_mat_model[triu_idx]

    dist_max = max(max(dist_val_data), max(dist_val_model))
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(dist_val_data, dist_val_model, 'b.', alpha=0.1, markersize=1.0, markeredgecolor='none')
    plt.plot([0, dist_max], [0, dist_max], 'k--')
    plt.xlim((0, dist_max))
    plt.ylim((0, dist_max))
    plt.grid(True)
    plt.xlabel('Distance in atlas space (data) [$\mu$m]')
    plt.ylabel('Distance in flat space (model) [$\mu$m]')
    plt.title(f'Cell distances in atlas vs. flat space [N={len(nrn_ids)}cells]')
    plt.tight_layout()
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_distances.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    # Nearest neighbors in atlas vs. flat space
    NN_mat_data = np.argsort(dist_mat_data, axis=1)
    NN_mat_model = np.argsort(dist_mat_model, axis=1)
    
    num_NN_list = list(range(1, 30, 1))
    NN_match = np.full(len(num_NN_list), np.nan)
    
    print('Computing nearest neighbors in atlas vs. flat space...', flush=True)
    pbar = progressbar.ProgressBar()
    for nidx in pbar(range(len(num_NN_list))):
        num_NN = num_NN_list[nidx]
        NN_match[nidx] = np.mean([len(np.intersect1d(NN_mat_data[i, 1 : 1 + num_NN], NN_mat_model[i, 1 : 1 + num_NN])) / num_NN for i in range(len(nrn_ids))])
    
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(num_NN_list, NN_match, '.-')
    plt.grid(True)
    plt.ylim((0, 1))
    plt.xlabel('#Nearest neighbors')
    plt.ylabel('Mean match')
    plt.title(f'Nearest neighbors in atlas vs. flat space [N={len(nrn_ids)}cells]')
    
    out_fn = os.path.abspath(os.path.join(out_dir, 'data_vs_model_neighbors.png'))
    print(f'INFO: Saving {out_fn}...')
    plt.savefig(out_fn)
    
    return

