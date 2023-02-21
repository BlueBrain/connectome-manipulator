import numpy as np
import pandas as pd
import connectome_manipulator.connectome_manipulation.no_manipulation as test_module


def test_apply():
    datalen = 10000
    edges_table = pd.DataFrame(np.random.random((datalen, 3)))

    res = test_module.apply(edges_table, None, None)
    assert res.equals(edges_table)
