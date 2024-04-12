# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from bluepysnap import Circuit
from utils import TEST_DATA_DIR

import connectome_manipulator.model_building.conn_props as test_module


def test_extract():
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    src_mtypes = np.unique(nodes[0].get(src_ids, properties="mtype"))
    tgt_mtypes = np.unique(nodes[1].get(tgt_ids, properties="mtype"))
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    tab_mtypes_src = (
        nodes[0].get(edges_table["@source_node"].to_numpy(), properties="mtype").to_numpy()
    )
    tab_mtypes_tgt = (
        nodes[1].get(edges_table["@target_node"].to_numpy(), properties="mtype").to_numpy()
    )

    # Extract selected synapse/connection properties per pathway (= reference data)
    sel_prop = "conductance"
    nsyn_data = {
        k: np.full((len(src_mtypes), len(tgt_mtypes)), np.nan)
        for k in ["mean", "std", "min", "max"]
    }
    prop_data = {
        k: np.full((len(src_mtypes), len(tgt_mtypes), 1), np.nan)
        for k in ["mean", "std", "shared_within", "min", "max"]
    }
    for i1, mt1 in enumerate(src_mtypes):
        for i2, mt2 in enumerate(tgt_mtypes):
            tab_sel = edges_table.loc[np.logical_and(tab_mtypes_src == mt1, tab_mtypes_tgt == mt2)]
            if tab_sel.size == 0:
                continue
            _, syn_conn_idx, nsyn_conn = np.unique(
                tab_sel[["@source_node", "@target_node"]],
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            nsyn_data["mean"][i1, i2] = np.mean(nsyn_conn)
            nsyn_data["std"][i1, i2] = np.std(nsyn_conn)
            nsyn_data["min"][i1, i2] = np.min(nsyn_conn)
            nsyn_data["max"][i1, i2] = np.max(nsyn_conn)

            # Check if property value shared within connection
            if len(nsyn_conn) > 1 and len(np.unique(tab_sel[sel_prop])) == 1:
                # In case of a constant distribution, assume no sharing
                is_shared = False
            else:
                is_shared = True
                for cidx in range(len(nsyn_conn)):
                    if len(np.unique(tab_sel.loc[syn_conn_idx == cidx, sel_prop])) > 1:
                        # Found different property values within same connection
                        is_shared = False
                        break

            # Get property values
            prop_values = []
            for cidx in range(len(nsyn_conn)):
                if is_shared:
                    # Shared within connection, so take only first value
                    prop_values.append(tab_sel.loc[syn_conn_idx == cidx, sel_prop].iloc[0])
                else:
                    # Different values within connection, so take all values
                    prop_values.append(tab_sel.loc[syn_conn_idx == cidx, sel_prop].to_numpy())
            prop_values = np.hstack(prop_values)

            # Compute statistics
            prop_data["mean"][i1, i2] = np.mean(prop_values)
            prop_data["std"][i1, i2] = np.std(prop_values)
            prop_data["shared_within"][i1, i2] = is_shared
            prop_data["min"][i1, i2] = np.min(prop_values)
            prop_data["max"][i1, i2] = np.max(prop_values)

    # Check extraction (w/o histograms)
    res = test_module.extract(
        c,
        min_sample_size_per_group=None,
        max_sample_size_per_group=None,
        hist_bins=51,
        sel_props=[sel_prop],
        sel_src=None,
        sel_dest=None,
    )
    assert_array_equal(res["m_types"][0], src_mtypes)
    assert_array_equal(res["m_types"][1], tgt_mtypes)
    assert_array_equal(res["syn_props"], [sel_prop])
    for k in nsyn_data.keys():
        assert_array_equal(res["syns_per_conn_data"][k], nsyn_data[k])
    for k in prop_data.keys():
        assert_array_almost_equal(res["conn_prop_data"][k], prop_data[k])

    # Check extraction with Inf min sample size => Everything should be NaN
    res = test_module.extract(
        c,
        min_sample_size_per_group=np.inf,
        max_sample_size_per_group=None,
        hist_bins=50,
        sel_props=[sel_prop],
        sel_src=None,
        sel_dest=None,
    )
    assert_array_equal(res["m_types"][0], src_mtypes)
    assert_array_equal(res["m_types"][1], tgt_mtypes)
    assert_array_equal(res["syn_props"], [sel_prop])
    for k in nsyn_data.keys():
        assert_array_equal(res["syns_per_conn_data"][k], np.full_like(nsyn_data[k], np.nan))
    for k in prop_data.keys():
        assert_array_equal(res["conn_prop_data"][k], np.full_like(prop_data[k], np.nan))


def test_build():
    m_types = [["L4_MC", "L4_PC", "L5_PC"]] * 2
    m_type_class = [["INH", "EXC", "EXC"]] * 2
    m_type_layer = [[4, 4, 5]] * 2

    props = ["conductance", "u_syn"]
    src_mtypes = m_types[0]
    tgt_mtypes = m_types[1]
    np.random.seed(0)
    nsyn_data = {
        k: np.random.randint(low=1, high=10, size=(len(src_mtypes), len(tgt_mtypes)))
        for k in ["mean", "std", "min", "max", "norm_loc", "norm_scale"]
    }
    prop_data = {
        k: np.random.rand(len(src_mtypes), len(tgt_mtypes), len(props))
        for k in ["mean", "std", "shared_within", "min", "max", "norm_loc", "norm_scale"]
    }
    prop_data["shared_within"] = np.round(prop_data["shared_within"])

    # Check model building (default settings)
    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={},
        data_types={},
        data_bounds={},
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr["type"] == "normal"
                assert model_distr["mean"] == prop_data["mean"][sidx, tidx, pidx]
                assert model_distr["std"] == prop_data["std"][sidx, tidx, pidx]
                assert model_distr["shared_within"] == prop_data["shared_within"][sidx, tidx, pidx]
            model_distr = res.get_distr_props(
                prop_name="n_syn_per_conn", src_type=s_mt, tgt_type=t_mt
            )
            assert model_distr["mean"] == nsyn_data["mean"][sidx, tidx]
            assert model_distr["std"] == nsyn_data["std"][sidx, tidx]

    with pytest.raises(KeyError):
        res.get_distr_props(prop_name="WRONG_NAME", src_type=s_mt, tgt_type=t_mt)
    with pytest.raises(KeyError):
        res.get_distr_props(prop_name="n_syn_per_conn", src_type="WRONG_TYPE", tgt_type=t_mt)
    with pytest.raises(KeyError):
        res.get_distr_props(prop_name="n_syn_per_conn", src_type=s_mt, tgt_type="WRONG_TYPE")

    # Check distribution types (truncnorm)
    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={pp: "truncnorm" for pp in props},
        data_types={},
        data_bounds={},
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr["type"] == "truncnorm"
                assert model_distr["norm_loc"] == prop_data["norm_loc"][sidx, tidx, pidx]
                assert model_distr["norm_scale"] == prop_data["norm_scale"][sidx, tidx, pidx]
                assert model_distr["min"] == prop_data["min"][sidx, tidx, pidx]
                assert model_distr["max"] == prop_data["max"][sidx, tidx, pidx]
                assert model_distr["shared_within"] == prop_data["shared_within"][sidx, tidx, pidx]

    with pytest.raises(AssertionError, match='Distribution type "WRONG_TYPE" not supported!'):
        test_module.build(
            nsyn_data,
            prop_data,
            m_types,
            m_type_class,
            m_type_layer,
            props,
            distr_types={pp: "WRONG_TYPE" for pp in props},
            data_types={},
            data_bounds={},
        )
    with pytest.raises(AssertionError, match="ERROR: Not all required attribute"):
        test_module.build(
            nsyn_data,
            prop_data,
            m_types,
            m_type_class,
            m_type_layer,
            props,
            distr_types={pp: "discrete" for pp in props},
            data_types={},
            data_bounds={},
        )

    # Check data types
    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={},
        data_types={pp: "int" for pp in props},
        data_bounds={},
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr = res.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr["dtype"] == "int"
                assert isinstance(res.draw(prop_name=pp, src_type=s_mt, tgt_type=t_mt)[0], np.int64)

    # Check data bounds
    test_bound = 10.0
    with pytest.raises(AssertionError, match="Data bounds error!"):
        test_module.build(
            nsyn_data,
            prop_data,
            m_types,
            m_type_class,
            m_type_layer,
            props,
            distr_types={},
            data_types={},
            data_bounds={pp: [2 * test_bound, test_bound] for pp in props},
        )

    res_l = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={},
        data_types={},
        data_bounds={pp: [test_bound, np.inf] for pp in props},
    )
    res_u = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={},
        data_types={},
        data_bounds={pp: [-np.inf, -test_bound] for pp in props},
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            for pidx, pp in enumerate(props):
                model_distr_l = res_l.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                model_distr_u = res_u.get_distr_props(prop_name=pp, src_type=s_mt, tgt_type=t_mt)
                assert model_distr_l["lower_bound"] == test_bound
                assert model_distr_u["upper_bound"] == -test_bound
                assert res_l.draw(prop_name=pp, src_type=s_mt, tgt_type=t_mt)[0] == test_bound
                assert res_u.draw(prop_name=pp, src_type=s_mt, tgt_type=t_mt)[0] == -test_bound

    # Check synapse generation (w/o randomization)
    n_syn = 5.0
    p_val = 2.0
    prop_data = {
        k: np.full((len(src_mtypes), len(tgt_mtypes), len(props)), p_val)
        for k in ["mean", "min", "max"]
    }
    prop_data.update({k: np.zeros((len(src_mtypes), len(tgt_mtypes), len(props))) for k in ["std"]})
    prop_data.update(
        {k: np.ones((len(src_mtypes), len(tgt_mtypes), len(props))) for k in ["shared_within"]}
    )
    nsyn_data = {
        k: np.full((len(src_mtypes), len(tgt_mtypes)), n_syn) for k in ["mean", "min", "max"]
    }
    nsyn_data.update({"std": np.zeros((len(src_mtypes), len(tgt_mtypes)))})
    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        props,
        distr_types={},
        data_types={},
        data_bounds={},
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            syn = res.apply(src_type=s_mt, tgt_type=t_mt)
            assert syn.shape[0] == n_syn
            assert syn.shape[1] == len(props)
            assert np.all(syn == p_val)

    # Check interpolation
    m_types = [["L4_PC:A", "L4_PC:B", "L5_PC", "L5_MC"]] * 2
    m_type_class = [["EXC", "EXC", "EXC", "INH"]] * 2
    m_type_layer = [[4, 4, 5, 5]] * 2

    prop = "conductance"
    src_mtypes = m_types[0]
    tgt_mtypes = m_types[1]
    np.random.seed(0)
    nsyn_data = {
        k: np.random.randint(low=1, high=10, size=(len(src_mtypes), len(tgt_mtypes))).astype(float)
        for k in ["mean", "std", "min", "max", "norm_loc", "norm_scale"]
    }
    prop_data = {
        k: np.random.rand(len(src_mtypes), len(tgt_mtypes), 1)
        for k in ["mean", "std", "shared_within", "min", "max", "norm_loc", "norm_scale"]
    }
    prop_data["shared_within"] = np.round(prop_data["shared_within"])

    nsyn_data.update(
        {
            k: np.full((len(src_mtypes), len(tgt_mtypes)), np.nan, dtype=object)
            for k in ["val", "cnt", "p"]
        }
    )
    prop_data.update(
        {
            k: np.full((len(src_mtypes), len(tgt_mtypes), 1), np.nan, dtype=object)
            for k in ["val", "cnt", "p"]
        }
    )
    for sidx, s_mt in enumerate(src_mtypes):
        for tidx, t_mt in enumerate(tgt_mtypes):
            fake_data = np.random.randint(20, size=10)
            val, cnt = np.unique(fake_data, return_counts=True)
            nsyn_data["val"][sidx, tidx] = val
            nsyn_data["cnt"][sidx, tidx] = cnt
            nsyn_data["p"][sidx, tidx] = cnt / np.sum(cnt)

            fake_data = np.random.randint(20, size=10)
            val, cnt = np.unique(fake_data, return_counts=True)
            prop_data["val"][sidx, tidx, 0] = val
            prop_data["cnt"][sidx, tidx, 0] = cnt
            prop_data["p"][sidx, tidx, 0] = cnt / np.sum(cnt)

    test_src = "L4_PC:A"
    test_tgt = "L4_PC:A"
    test_src_idx = np.where(np.array(src_mtypes) == test_src)[0][0]
    test_tgt_idx = np.where(np.array(tgt_mtypes) == test_tgt)[0][0]

    def comb_discrete(vals, cnts):
        """Combines two discrete distributions. [Assuming no NaNs in vals/cnts!]"""
        comb = np.hstack(
            [np.hstack([np.repeat(_v, _c) for _v, _c in zip(v, c)]) for v, c in zip(vals, cnts)]
        )
        new_val, new_cnt = np.unique(comb, return_counts=True)
        new_p = new_cnt / np.sum(new_cnt)
        return new_val, new_cnt, new_p

    ## No interpolation
    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert model_distr["norm_loc"] == prop_data["norm_loc"][test_src_idx, test_tgt_idx, 0]
    assert model_distr["norm_scale"] == prop_data["norm_scale"][test_src_idx, test_tgt_idx, 0]
    assert model_distr["shared_within"] == prop_data["shared_within"][test_src_idx, test_tgt_idx, 0]
    assert model_distr["min"] == prop_data["min"][test_src_idx, test_tgt_idx, 0]
    assert model_distr["max"] == prop_data["max"][test_src_idx, test_tgt_idx, 0]

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert np.array_equal(model_distr["val"], prop_data["val"][test_src_idx, test_tgt_idx, 0])
    assert np.array_equal(model_distr["p"], prop_data["p"][test_src_idx, test_tgt_idx, 0])
    assert model_distr["shared_within"] == prop_data["shared_within"][test_src_idx, test_tgt_idx, 0]

    ## Level0 interpolation (source m-type & target layer/synapse class value)
    prop_data["mean"][
        test_src_idx, test_tgt_idx, 0
    ] = np.nan  # NOTE: Based on 'mean', the interpolation strategy (level) will be determined
    nsyn_data["mean"][test_src_idx, test_tgt_idx] = np.nan
    tgt_sel = np.logical_and(
        np.array(m_type_layer[1]) == m_type_layer[1][test_tgt_idx],
        np.array(m_type_class[1]) == m_type_class[1][test_tgt_idx],
    )

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert model_distr["norm_loc"] == np.nanmean(prop_data["norm_loc"][test_src_idx, tgt_sel, 0])
    assert model_distr["norm_scale"] == np.nanmean(
        prop_data["norm_scale"][test_src_idx, tgt_sel, 0]
    )
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][test_src_idx, tgt_sel, 0]).astype(bool)
    )  # Majority vote
    assert model_distr["min"] == np.nanmin(prop_data["min"][test_src_idx, tgt_sel, 0])
    assert model_distr["max"] == np.nanmax(prop_data["max"][test_src_idx, tgt_sel, 0])

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    val, _, p = comb_discrete(
        prop_data["val"][test_src_idx, tgt_sel, 0], prop_data["cnt"][test_src_idx, tgt_sel, 0]
    )
    assert np.array_equal(model_distr["val"], val)
    assert np.array_equal(model_distr["p"], p)
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][test_src_idx, tgt_sel, 0]).astype(bool)
    )  # Majority vote

    ## Level1 interpolation (source m-type & target synapse class value)
    prop_data["mean"][
        test_src_idx, tgt_sel, 0
    ] = np.nan  # NOTE: Based on 'mean', the interpolation strategy (level) will be determined
    nsyn_data["mean"][test_src_idx, tgt_sel] = np.nan
    tgt_sel = np.array(m_type_class[1]) == m_type_class[1][test_tgt_idx]

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert model_distr["norm_loc"] == np.nanmean(prop_data["norm_loc"][test_src_idx, tgt_sel, 0])
    assert model_distr["norm_scale"] == np.nanmean(
        prop_data["norm_scale"][test_src_idx, tgt_sel, 0]
    )
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][test_src_idx, tgt_sel, 0]).astype(bool)
    )  # Majority vote
    assert model_distr["min"] == np.nanmin(prop_data["min"][test_src_idx, tgt_sel, 0])
    assert model_distr["max"] == np.nanmax(prop_data["max"][test_src_idx, tgt_sel, 0])

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    val, _, p = comb_discrete(
        prop_data["val"][test_src_idx, tgt_sel, 0], prop_data["cnt"][test_src_idx, tgt_sel, 0]
    )
    assert np.array_equal(model_distr["val"], val)
    assert np.array_equal(model_distr["p"], p)
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][test_src_idx, tgt_sel, 0]).astype(bool)
    )  # Majority vote

    ## Level2 interpolation (source & target per layer/synapse class value)
    prop_data["mean"][
        test_src_idx, tgt_sel, 0
    ] = np.nan  # NOTE: Based on 'mean', the interpolation strategy (level) will be determined
    nsyn_data["mean"][test_src_idx, tgt_sel] = np.nan
    src_sel = np.logical_and(
        np.array(m_type_layer[0]) == m_type_layer[0][test_src_idx],
        np.array(m_type_class[0]) == m_type_class[0][test_src_idx],
    )
    tgt_sel = np.logical_and(
        np.array(m_type_layer[1]) == m_type_layer[1][test_tgt_idx],
        np.array(m_type_class[1]) == m_type_class[1][test_tgt_idx],
    )

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert model_distr["norm_loc"] == np.nanmean(prop_data["norm_loc"][src_sel, :, 0][:, tgt_sel])
    assert model_distr["norm_scale"] == np.nanmean(
        prop_data["norm_scale"][src_sel, :, 0][:, tgt_sel]
    )
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][src_sel, :, 0][:, tgt_sel]).astype(bool)
    )  # Majority vote
    assert model_distr["min"] == np.nanmin(prop_data["min"][src_sel, :, 0][:, tgt_sel])
    assert model_distr["max"] == np.nanmax(prop_data["max"][src_sel, :, 0][:, tgt_sel])

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    val, _, p = comb_discrete(
        prop_data["val"][src_sel, :, 0][:, tgt_sel], prop_data["cnt"][src_sel, :, 0][:, tgt_sel]
    )
    assert np.array_equal(model_distr["val"], val)
    assert np.array_equal(model_distr["p"], p)
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][src_sel, :, 0][:, tgt_sel]).astype(bool)
    )  # Majority vote

    ## Level3 interpolation (source & target per synapse class value)
    for tidx in np.where(tgt_sel)[0]:
        prop_data["mean"][
            src_sel, tidx, 0
        ] = np.nan  # NOTE: Based on 'mean', the interpolation strategy (level) will be determined
        nsyn_data["mean"][src_sel, tidx] = np.nan
    src_sel = np.array(m_type_class[0]) == m_type_class[0][test_src_idx]
    tgt_sel = np.array(m_type_class[1]) == m_type_class[1][test_tgt_idx]

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert model_distr["norm_loc"] == np.nanmean(prop_data["norm_loc"][src_sel, :, 0][:, tgt_sel])
    assert model_distr["norm_scale"] == np.nanmean(
        prop_data["norm_scale"][src_sel, :, 0][:, tgt_sel]
    )
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][src_sel, :, 0][:, tgt_sel]).astype(bool)
    )  # Majority vote
    assert model_distr["min"] == np.nanmin(prop_data["min"][src_sel, :, 0][:, tgt_sel])
    assert model_distr["max"] == np.nanmax(prop_data["max"][src_sel, :, 0][:, tgt_sel])

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    val, _, p = comb_discrete(
        prop_data["val"][src_sel, :, 0][:, tgt_sel], prop_data["cnt"][src_sel, :, 0][:, tgt_sel]
    )
    assert np.array_equal(model_distr["val"], val)
    assert np.array_equal(model_distr["p"], p)
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"][src_sel, :, 0][:, tgt_sel]).astype(bool)
    )  # Majority vote

    ## Level4 interpolation (overall value)
    for tidx in np.where(tgt_sel)[0]:
        prop_data["mean"][
            src_sel, tidx, 0
        ] = np.nan  # NOTE: Based on 'mean', the interpolation strategy (level) will be determined
        nsyn_data["mean"][src_sel, tidx] = np.nan

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "truncnorm"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    assert np.isclose(model_distr["norm_loc"], np.nanmean(prop_data["norm_loc"]))
    assert np.isclose(model_distr["norm_scale"], np.nanmean(prop_data["norm_scale"]))
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"]).astype(bool)
    )  # Majority vote
    assert model_distr["min"] == np.nanmin(prop_data["min"])
    assert model_distr["max"] == np.nanmax(prop_data["max"])

    res = test_module.build(
        nsyn_data,
        prop_data,
        m_types,
        m_type_class,
        m_type_layer,
        [prop],
        distr_types={prop: "discrete"},
        data_types={},
        data_bounds={},
    )
    model_distr = res.get_distr_props(prop_name=prop, src_type=test_src, tgt_type=test_tgt)
    val, _, p = comb_discrete(np.squeeze(prop_data["val"]), np.squeeze(prop_data["cnt"]))
    assert np.array_equal(model_distr["val"], val)
    assert np.array_equal(model_distr["p"], p)
    assert model_distr["shared_within"] == np.round(
        np.nanmean(prop_data["shared_within"]).astype(bool)
    )  # Majority vote
