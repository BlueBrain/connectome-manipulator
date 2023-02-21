"""Collection of function for flexible nodes/edges access, to be used by model building and manipulation operations"""

import numpy as np
from connectome_manipulator import log


def get_node_ids(nodes, sel_spec):
    """Returns list of selected node IDs of given nodes population.

    nodes ... NodePopulation
    sel_spec ... Node selection specifier, as accepted by nodes.ids(group=sel_spec).
                 In addition, if sel_spec is a dict, 'node_set': '<node_set_name>'
                 can be specified in combination with other selection properties.
    """
    if isinstance(sel_spec, dict):
        sel_group = sel_spec.copy()
        node_set = sel_group.pop("node_set", None)

        gids = nodes.ids(sel_group)

        if node_set is not None:
            log.log_assert(isinstance(node_set, str), "Node set must be a string!")
            gids = np.intersect1d(gids, nodes.ids(node_set))
    else:
        gids = nodes.ids(sel_spec)

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
