# Python script to apply connectome manipulations to SONATA circuit

# Initialization

""" Global imports """
import sys
import os.path

""" Paths """
sys.path.insert(0, '../pipeline/')

""" Local imports """
from connectome_manipulation import connectome_manipulation


# Connectome manipulation - Configuration
manip_config = {}

""" Circuit """
manip_config['circuit_path'] = '/gpfs/bbp.cscs.ch/project/proj83/home/pokorny/SSCx/projections'
manip_config['circuit_config'] = 'VPM_config.json' # SONATA (.json) format; path rel. to 'circuit_path'

""" Random seed """
manip_config['seed'] = 3210
manip_config['N_split_nodes'] = 20

""" Manipulation (name + sequence of manipulation functions with arguments) """
manip_config['manip'] = {'name': 'ScaleGsynX2', 'fcts': [{'source': 'syn_prop_alteration', 'kwargs': {'syn_prop': 'conductance', 'new_value': {'mode': 'scale', 'factor': 2}, 'amount_pct': 100.0}}]}

# Connectome manipulation - Apply manipulation
connectome_manipulation.main(manip_config, do_profiling=True)
