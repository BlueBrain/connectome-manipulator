"""Manipulation name: no_manipulation

Description: This is just a dummy manipulation function performing no manipulation at all.
This function is intended as a control condition to run the manipulation pipeline
without actually manipulating the connectome.
"""

from connectome_manipulator import log


def apply(edges_table, _nodes, _aux_dict):
    """No manipulation (control condition)."""
    log.info("Nothing to do")

    return edges_table
