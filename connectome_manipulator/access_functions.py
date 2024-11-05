# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Collection of function for flexible nodes/edges access, to be used by model building and manipulation operations"""

import numpy as np
import pandas as pd

import libsonata
from bluepysnap.sonata_constants import Node, DYNAMICS_PREFIX
from bluepysnap.utils import add_dynamic_prefix
from bluepysnap.utils import euler2mat, quaternion2mat
from sklearn.model_selection import KFold

from connectome_manipulator import log


def property_names(nodes):
    """Get all property names for a population"""
    population = nodes.to_libsonata
    return set(population.attribute_names) | set(
        add_dynamic_prefix(population.dynamics_attribute_names)
    )


def _ids_to_ranges(ids):
    """Convert a list of IDs to libsonata.Selection style ranges"""
    return libsonata.Selection(ids).ranges


def get_nodes(nodes, selection=None):
    """Get a pandas table of all nodes and their properties, optionally narrowed by a selection"""
    population = nodes.to_libsonata
    categoricals = population.enumeration_names

    if selection is None:
        selection = nodes.select_all()
    result = pd.DataFrame(index=selection.flatten())

    for attr in sorted(population.attribute_names):
        if attr in categoricals:
            enumeration = np.asarray(population.get_enumeration(attr, selection))
            values = np.asarray(population.enumeration_values(attr))
            # if the size of `values` is large enough compared to `enumeration`, not using
            # categorical reduces the memory usage.
            if values.shape[0] < 0.5 * enumeration.shape[0]:
                result[attr] = pd.Categorical.from_codes(enumeration, categories=values)
            else:
                result[attr] = values[enumeration]
        else:
            result[attr] = population.get_attribute(attr, selection)
    for attr in sorted(add_dynamic_prefix(population.dynamics_attribute_names)):
        result[attr] = population.get_dynamics_attribute(attr.split(DYNAMICS_PREFIX)[1], selection)
    return result


def orientations(nodes, node_sel=None):
    """Node orientation(s) as a list of numpy arrays.

    Args:
        nodes: the node set for which we want to get the rotation matrices
        node_sel: (optional) a libsonata Selection to narrow our selection

    Returns:
        numpy.ndarray:
            A list of 3x3 rotation matrices for the given node set and selection.
    """
    # need to keep this quaternion ordering for quaternion2mat (expects w, x, y , z)
    props = np.array(
        [Node.ORIENTATION_W, Node.ORIENTATION_X, Node.ORIENTATION_Y, Node.ORIENTATION_Z]
    )
    props_mask = np.isin(props, list(property_names(nodes)))
    orientation_count = np.count_nonzero(props_mask)
    if orientation_count == 4:
        trans = quaternion2mat
    elif orientation_count in [1, 2, 3]:
        raise ValueError(
            "Missing orientation fields. Should be 4 quaternions or euler angles or nothing"
        )
    else:
        # need to keep this rotation_angle ordering for euler2mat (expects z, y, x)
        props = np.array(
            [
                Node.ROTATION_ANGLE_Z,
                Node.ROTATION_ANGLE_Y,
                Node.ROTATION_ANGLE_X,
            ]
        )
        props_mask = np.isin(props, list(property_names(nodes)))
        trans = euler2mat
    result = get_nodes(nodes, node_sel)
    if props[props_mask].size:
        result = result[props[props_mask]]

    def _get_values(prop):
        """Retrieve prop from the result Dataframe/Series."""
        if isinstance(result, pd.Series):
            return [result.get(prop, 0)]
        return result.get(prop, np.zeros((result.shape[0],)))

    args = [_get_values(prop) for prop in props]
    return trans(*args)


def get_enumeration_list(pop, column):
    """Takes a node population and column name and returns a list to values."""
    raw_pop = pop.to_libsonata
    if column in raw_pop.enumeration_names:
        return raw_pop.enumeration_values(column)
    return sorted(np.unique(raw_pop.get_attribute(column, raw_pop.select_all())))


def get_enumeration_map(pop, column):
    """Takes a node population and column name and returns a dictionary that maps values to indices."""
    raw_pop = pop.to_libsonata
    if column in raw_pop.enumeration_names:
        return {key: idx for idx, key in enumerate(raw_pop.enumeration_values(column))}
    return {
        key: idx
        for idx, key in enumerate(
            sorted(np.unique(raw_pop.get_attribute(column, raw_pop.select_all())))
        )
    }


def get_attribute(pop, column, ids):
    """Get the attribute values for `column` from population `pop` for node IDs `ids`."""
    raw_pop = pop.to_libsonata
    return raw_pop.get_attribute(column, libsonata.Selection(ids))


def get_enumeration(pop, column, ids=None):
    """Get the raw enumeration values for `column` from population `pop` for node IDs `ids`."""
    raw_pop = pop.to_libsonata
    if ids is None:
        ids = raw_pop.select_all()
    else:
        ids = libsonata.Selection(ids)
    if column in raw_pop.enumeration_names:
        return raw_pop.get_enumeration(column, ids)
    mapping = get_enumeration_map(pop, column)
    return np.array([mapping[v] for v in raw_pop.get_attribute(column, ids)])


def get_node_ids(nodes, sel_spec, split_ids=None):
    """Returns list of selected node IDs of given nodes population.

    nodes ... NodePopulation
    sel_spec ... Node selection specifier, as accepted by nodes.ids(group=sel_spec).
                 In addition, if sel_spec is a dict, 'node_set': '<node_set_name>'
                 can be specified in combination with other selection properties.
    split_ids ... Node IDs to filter the selection by, either as an array or a
                  libsonata.Selection
    """
    pop = nodes.to_libsonata
    enumeration_names = pop.enumeration_names
    if split_ids is None:
        sel_ids = pop.select_all()
    else:
        sel_ids = libsonata.Selection(split_ids)
    if isinstance(sel_spec, dict):
        sel_group = sel_spec.copy()
        node_set = sel_group.pop("node_set", None)

        selection = None
        for sel_k, sel_v in sel_group.items():
            if sel_k in enumeration_names:
                if isinstance(sel_v, list):  # Merge multiple selections
                    sel_idx = [pop.enumeration_values(sel_k).index(_v) for _v in sel_v]
                    sel_prop = np.isin(pop.get_enumeration(sel_k, sel_ids), sel_idx)
                else:  # Single selection
                    sel_idx = pop.enumeration_values(sel_k).index(sel_v)
                    sel_prop = pop.get_enumeration(sel_k, sel_ids) == sel_idx
            else:
                if isinstance(sel_v, list):  # Merge multiple selections
                    sel_prop = np.isin(pop.get_attribute(sel_k, sel_ids), sel_v)
                else:  # Single selection
                    sel_prop = pop.get_attribute(sel_k, sel_ids) == sel_v
            if selection is None:
                selection = sel_prop
            else:
                selection &= sel_prop
        # selection is not of all nodes (starting with node id 0), but a generic subset specified by sel_ids
        if len(sel_ids.ranges) == 1:
            # First turn selection array into an index array then
            # because we filtered contiguous ids with a simple offset, just shift by the first node id
            gids = np.nonzero(selection)[0] + sel_ids.ranges[0][0]
        else:
            # for more complex cases, fully resolve the node-preselection
            gids = sel_ids.flatten().astype(np.int64)[selection]

        if node_set is not None:
            log.log_assert(isinstance(node_set, str), "Node set must be a string!")
            if selection is None:  # Nothing else selected
                gids = nodes.ids(node_set)
                if split_ids is not None:
                    gids = np.intersect1d(gids, sel_ids.flatten().astype(np.int64))
            else:  # Otherwise, intersect with selection
                gids = np.intersect1d(gids, nodes.ids(node_set))
    else:
        gids = nodes.ids(sel_spec)
        if split_ids is not None:
            gids = np.intersect1d(gids, sel_ids.flatten().astype(np.int64))

    return gids


def get_nodes_population(circuit, popul_name=None):
    """Select default nodes population. Optionally, the population name can be specified."""
    log.log_assert(len(circuit.nodes.population_names) > 0, "No node population found!")
    if popul_name is None:
        if len(circuit.nodes.population_names) == 1:
            popul_name = circuit.nodes.population_names[0]  # Select the only existing population
        else:
            popul_name = "All"  # Use default name
            log.warning(
                f'Multiple node populations found - Trying to load "{popul_name}" population!'
            )
    log.log_assert(
        popul_name in circuit.nodes.population_names,
        f'Population "{popul_name}" not found in nodes file!',
    )
    nodes = circuit.nodes[popul_name]

    return nodes


def get_edges_population(circuit, popul_name=None, return_popul_name=False):
    """Select default edge population. Optionally, the population name can be specified."""
    #     log.log_assert(len(circuit.edges.population_names) == 1, 'Only a single edge population per file supported!')
    #     edges = circuit.edges[circuit.edges.population_names[0]]
    log.log_assert(len(circuit.edges.population_names) > 0, "No edge population found!")
    if popul_name is None:
        if len(circuit.edges.population_names) == 1:
            popul_name = circuit.edges.population_names[0]  # Select the only existing population
        else:  # Try to use one of default names
            popul_name = "default__default__chemical"
            if popul_name not in circuit.edges.population_names:
                popul_name = "default"
            log.warning(
                f'Multiple edges populations found - Trying to load "{popul_name}" population!'
            )
    log.log_assert(
        popul_name in circuit.edges.population_names,
        f'Population "{popul_name}" not found in edges file!',
    )
    edges = circuit.edges[popul_name]

    if return_popul_name:
        return edges, popul_name
    else:
        return edges


def get_node_positions(nodes, node_ids, vox_map=None):
    """Return x/y/z positions of list of nodes, optionally mapped using VoxelData map."""
    _pop = nodes.to_libsonata
    _sel = libsonata.Selection(node_ids)
    raw_pos = np.column_stack(
        (
            _pop.get_attribute("x", _sel),
            _pop.get_attribute("y", _sel),
            _pop.get_attribute("z", _sel),
        )
    )
    if vox_map:  # Apply voxel map
        pos = vox_map.lookup(raw_pos)
        log.log_assert(
            not np.any(pos == vox_map.OUT_OF_BOUNDS),
            "Out of bounds error in mapped positions from voxel data!",
        )
    else:  # No voxel mapping
        pos = raw_pos
    return raw_pos, pos


def get_connections(edges, pre_ids, post_ids, with_nsyn=False):
    """Returns connections between given src/tgt node IDs, optionally incl. #synapses per connection."""
    it_conns = edges.iter_connections(pre_ids, post_ids, return_edge_count=with_nsyn)
    conns = np.array(
        [([_c.id for _c in _conn[:2]] + list(_conn[2:])) for _conn in it_conns]
    )  # Resolve src/tgt IDs
    return conns


def get_cv_data(data_list, cv_dict=None):
    """Returns training/testing data items of specified cross-validation fold."""
    if cv_dict is None:  # No CV
        return data_list
    else:
        log.log_assert(
            isinstance(cv_dict, dict)
            and "n_folds" in cv_dict
            and "fold_idx" in cv_dict
            and "training_set" in cv_dict,
            'ERROR: Cross-validation "cv_dict" must be a dict containing "n_folds", "fold_idx", and "training_set" keys!',
        )
        log.log_assert(
            cv_dict["n_folds"] > 1 and 0 <= cv_dict["fold_idx"] < cv_dict["n_folds"],
            "ERROR: Cross-validation index error!",
        )

        kf = KFold(n_splits=cv_dict["n_folds"], shuffle=True)
        cv_data_list = []
        for _data in data_list:
            sel_idx = [
                _train_index if cv_dict["training_set"] else _test_index
                for (_train_index, _test_index) in kf.split(_data)
            ][cv_dict["fold_idx"]]
            cv_data_list.append(_data[sel_idx])

        # CV TESTING: Writing CV data splits into data log #
        # dlog_name = f"CVData{cv_dict.get('n_folds', 0)}-{cv_dict.get('fold_idx', -1) + 1}-Train{cv_dict.get('training_set')}"
        # cv_data_dict = {f"cv_data{_i}": _data for _i, _data in enumerate(cv_data_list)}
        # log.data(dlog_name, **cv_data_dict)
        ##############

        return cv_data_list
