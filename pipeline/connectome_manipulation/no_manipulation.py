# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

import logging

""" No manipulation (control condition) """
def apply(edges_table, nodes, aux_dict):
    
    logging.info('Nothing to do')
    
    return edges_table
