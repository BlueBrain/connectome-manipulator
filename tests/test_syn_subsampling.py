import numpy as np
import connectome_manipulator.connectome_manipulation.syn_subsampling as test_module


def test_apply():
    # NOTE by herttuai on 13/10/2021:
    # If I am not completely mistaken syn_subsampling could be as simple as
    # def apply(edges_table, nodes, _aux_dict, keep_pct=100,0):
    #     return syn_removal(edges_table, nodes, _aux_dict, amount_pct=100-keep_pct)

    datalen = 10000
    edges_table = np.random.random((datalen, 3))

    for pct in [10.1, 50.99999, 0.666, 99.9999]:
        res = test_module.apply(edges_table, None, None, keep_pct=pct)
        assert len(res) == np.round(pct * datalen / 100)
        assert np.all(np.isin(res, edges_table))
