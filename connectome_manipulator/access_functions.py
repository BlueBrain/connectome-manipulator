"""Collection of function for flexible nodes/edges access, to be used by model building and manipulation operations"""

import numpy as np

import libsonata

from connectome_manipulator import log


def get_enumeration_map(pop, column):
    """Takes a node population and column name and returns a dictionary that maps values to indices."""
    raw_pop = pop._population  # pylint: disable=protected-access
    if column in raw_pop.enumeration_names:
        return {key: idx for idx, key in enumerate(raw_pop.enumeration_values(column))}
    return {
        key: idx
        for idx, key in enumerate(
            sorted(np.unique(raw_pop.get_attribute(column, raw_pop.select_all())))
        )
    }


def get_enumeration(pop, column, ids):
    """Get the raw enumeration values for `column` from population `pop` for node IDs `ids`."""
    raw_pop = pop._population  # pylint: disable=protected-access
    if column in raw_pop.enumeration_names:
        return raw_pop.get_enumeration(column, libsonata.Selection(ids))
    mapping = get_enumeration_map(pop, column)
    return np.array([mapping[v] for v in raw_pop.get_attribute(column, libsonata.Selection(ids))])


def get_node_ids(nodes, sel_spec, split_ids=None):
    """Returns list of selected node IDs of given nodes population.

    nodes ... NodePopulation
    sel_spec ... Node selection specifier, as accepted by nodes.ids(group=sel_spec).
                 In addition, if sel_spec is a dict, 'node_set': '<node_set_name>'
                 can be specified in combination with other selection properties.
    split_ids ... Node IDs to filter the selection by, either as an array or a
                  libsonata.Selection
    """
    # pylint: disable=protected-access
    pop = nodes._population
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
                sel_idx = pop.enumeration_values(sel_k).index(sel_v)
                sel_prop = pop.get_enumeration(sel_k, sel_ids) == sel_idx
            else:
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
            gids = np.intersect1d(gids, nodes.ids(node_set))
    else:
        gids = nodes.ids(sel_spec)
        if split_ids is not None:
            gids = np.intersect1d(gids, sel_ids.flatten().astype(np.int64))

    return gids


def get_nodes_population(circuit, popul_name=None):
    """Select default edge population. Optionally, the population name can be specified."""
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
        f'Population "{popul_name}" not found in edges file!',
    )
    nodes = circuit.nodes[popul_name]

    return nodes


def get_edges_population(circuit, popul_name=None):
    """Select default edge population. Optionally, the population name can be specified."""
    #     log.log_assert(len(circuit.edges.population_names) == 1, 'Only a single edge population per file supported!')
    #     edges = circuit.edges[circuit.edges.population_names[0]]
    log.log_assert(len(circuit.edges.population_names) > 0, "No edge population found!")
    if popul_name is None:
        if len(circuit.edges.population_names) == 1:
            popul_name = circuit.edges.population_names[0]  # Select the only existing population
        else:
            popul_name = "default"  # Use default name
            log.warning(
                f'Multiple edges populations found - Trying to load "{popul_name}" population!'
            )
    log.log_assert(
        popul_name in circuit.edges.population_names,
        f'Population "{popul_name}" not found in edges file!',
    )
    edges = circuit.edges[popul_name]

    return edges
