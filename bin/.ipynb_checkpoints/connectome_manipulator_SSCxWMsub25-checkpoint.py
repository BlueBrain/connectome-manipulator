# Python script to apply connectome manipulations to SONATA circuit

# Initialization

""" Global imports """
import sys
import os.path

""" Paths """
sys.path.insert(0, '../pipeline/')

""" Local imports """
from connectome_manipulation import connectome_manipulation
from connectome_comparison import structural_comparator


# Connectome manipulation - Configuration
manip_config = {}

""" Circuit (base path) """
manip_config['circuit_path'] = f'/gpfs/bbp.cscs.ch/data/scratch/proj83/home/pokorny/sscx-white-matter'
circuit_name = os.path.split(manip_config['circuit_path'])[1]

""" Random seed """
manip_config['seed'] = 3210
manip_config['N_split_nodes'] = 20

""" Manipulation (name + sequence of manipulation functions with arguments) """
manip_config['manip'] = {'name': 'SubsampleSyn25pct', 'fcts': [{'source': 'syn_subsampling', 'kwargs': {'keep_pct': 25.0}}]}

# Connectome manipulation - Apply manipulation
connectome_manipulation.main(manip_config, do_profiling=True)
