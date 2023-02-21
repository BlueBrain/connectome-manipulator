import numpy as np
import pandas as pd
import connectome_manipulator.connectome_manipulation.syn_subsampling as test_module


def test_apply():
    # NOTE by herttuai on 13/10/2021:
    # If I am not completely mistaken syn_subsampling could be as simple as
    # def apply(edges_table, nodes, _aux_dict, keep_pct=100,0):
    #     return syn_removal(edges_table, nodes, _aux_dict, amount_pct=100-keep_pct)
    # NOTE by chr-pok on 24/01/2022:
    # In principle yes, but syn_subsampling should run a bit faster since it does
    # not treat any cell selections or other special cases

    datalen = 10000
    edges_table = pd.DataFrame(np.random.random((datalen, 3)))

    for pct in [10.1, 50.99999, 0.666, 99.9999]:
        res = test_module.apply(edges_table, None, None, keep_pct=pct)
        assert res.shape[0] == np.round(pct * datalen / 100)
        assert np.all(
            [np.any(np.all(res.iloc[i] == edges_table, 1)) for i in range(res.shape[0])]
        )  # Check is all rows are contained
