"""init"""

from connectome_manipulator.connectome_manipulation.manipulation.base import (
    Manipulation,
    MorphologyCachingManipulation,
)

from . import (
    conn_wiring,
    conn_rewiring,
    conn_extraction,
    conn_removal,
    null_manipulation,
    syn_subsampling,
    syn_removal,
    syn_prop_alteration,
)
