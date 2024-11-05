# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os

import morphio
import neurom as nm
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from bluepysnap import Circuit
from bluepysnap.morph import MorphHelper

import pytest
import re
from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.model_building import model_types
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator import log


# SONATA section type mapping (as in MorphIO): 1 = soma, 2 = axon, 3 = basal, 4 = apical
SEC_SOMA = 1
SEC_TYPE_MAP = {nm.AXON: 2, nm.BASAL_DENDRITE: 3, nm.APICAL_DENDRITE: 4}


@pytest.fixture
def manipulation():
    m = Manipulation.get("conn_rewiring")
    return m


def test_apply(manipulation):
    log.setup_logging()  # To have data logging in a defined state

    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    edges_table_empty = edges_table.loc[[]].copy()

    delay_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__DistDepDelay.json"
    )  # Deterministic delay model w/o variation
    delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
    props_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__ConnProps.json"
    )  # Deterministic connection properties model w/o variation
    props_model = model_types.AbstractModel.model_from_file(props_model_file)
    props_model_file2 = os.path.join(
        TEST_DATA_DIR, f"model_config__ConnPropsNoNSynConn.json"
    )  # Deterministic connection properties model w/o variation, w/o #syn/conn
    props_model2 = model_types.AbstractModel.model_from_file(props_model_file2)

    assert not np.all(
        np.diff(edges_table["@target_node"]) >= 0
    ), "ERROR: Edges table assumed to be not sorted!"

    with pytest.raises(
        AssertionError, match=re.escape("Edges table must be ordered by @target_node!")
    ):
        with EdgeWriter(None, existing_edges=edges_table.copy()) as writer:
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class="EXC",
                prob_model_spec=None,
                delay_model_spec={"file": delay_model_file},
            )
            res = writer.to_pandas()

    # Use sorted edges table from now on
    edges_table = edges_table.sort_values(["@target_node", "@source_node"]).reset_index(drop=True)

    def check_delay(res, nodes, syn_class, delay_model):
        """Check delays (from PRE neuron (soma) to POST synapse position)"""
        for idx in res[
            np.isin(res["@source_node"], nodes[0].ids({"synapse_class": syn_class}))
        ].index:
            delay_offset = delay_model.get_param_dict()["delay_mean_coeff_a"]
            delay_scale = delay_model.get_param_dict()["delay_mean_coeff_b"]
            src_pos = nodes[0].positions(res.loc[idx]["@source_node"]).to_numpy()
            syn_pos = res.loc[idx][
                ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
            ].to_numpy()
            dist = np.sqrt(np.sum((src_pos - syn_pos) ** 2))
            delay = delay_scale * dist + delay_offset
            assert np.all(np.isclose(res.loc[idx]["delay"], delay)), "ERROR: Delay mismatch!"

    def check_nsyn(ref, res):
        """Check number of synapses per connection"""
        nsyn_per_conn1 = np.unique(
            ref[["@source_node", "@target_node"]], axis=0, return_counts=True
        )[1]
        nsyn_per_conn2 = np.unique(
            res[["@source_node", "@target_node"]], axis=0, return_counts=True
        )[1]
        assert np.array_equal(
            np.sort(nsyn_per_conn1), np.sort(nsyn_per_conn2)
        ), "ERROR: Synapses per connection mismatch!"

    def check_indegree(ref, res, nodes, check_not_equal=False):
        """Check indegree"""
        indeg1 = [
            len(np.unique(ref["@source_node"][ref["@target_node"] == tid]))
            for tid in nodes[1].ids()
        ]
        indeg2 = [
            len(np.unique(res["@source_node"][res["@target_node"] == tid]))
            for tid in nodes[1].ids()
        ]
        if check_not_equal:
            assert not np.array_equal(indeg1, indeg2), "ERROR: Indegree should be different!"
        else:
            assert np.array_equal(indeg1, indeg2), "ERROR: Indegree mismatch!"

    def check_unchanged(ref, res, nodes, syn_class):
        """Check that non-target connections unchanged"""
        unch_tab1 = ref[~np.isin(ref["@source_node"], nodes[0].ids({"synapse_class": syn_class}))]
        unch_tab2 = res[~np.isin(res["@source_node"], nodes[0].ids({"synapse_class": syn_class}))]
        assert np.all(
            [
                np.sum(np.all(unch_tab1.iloc[idx] == unch_tab2, 1)) == 1
                for idx in range(unch_tab1.shape[0])
            ]
        ), f"ERROR: Non-{syn_class} connections changed!"

    def check_all_removed(ref, res, nodes, syn_class):
        """Check if all (EXC or INH) connections are removed"""
        assert (
            ref[~np.isin(ref["@source_node"], nodes[0].ids({"synapse_class": syn_class}))][
                res.columns
            ]
            .reset_index(drop=True)
            .equals(res)
        ), f"ERROR: Removed {syn_class} connections mismatch!"

    def check_sampling(ref, res, col_sel):
        """Check if synapse properties (incl. #syn/conn) sampled from existing values"""
        nsyn_per_conn1 = np.unique(
            ref[["@source_node", "@target_node"]], axis=0, return_counts=True
        )[1]
        nsyn_per_conn2 = np.unique(
            res[["@source_node", "@target_node"]], axis=0, return_counts=True
        )[1]
        assert np.all(
            np.isin(nsyn_per_conn2, nsyn_per_conn1)
        ), "ERROR: Synapses per connection sampling error!"  # Check sampling (#syn/conn)
        assert np.all(
            [np.all(np.isin(np.unique(res[col]), np.unique(ref[col]))) for col in col_sel]
        ), "ERROR: Synapse properties sampling error!"  # Check sampling (w/o #syn/conn)

    def check_zero(res, cols_unused, nodes, syn_class):
        """Check if unused properties of all rewired (EXC or INH) connections are set to zero"""
        src_sel = nodes[0].ids({"synapse_class": syn_class})
        assert np.all(
            res[np.isin(res["@source_node"], src_sel)][cols_unused] == 0
        ), "ERROR: Unused properties not set to zero!"

    def check_all_to_all(ref, res, nodes, syn_class, props_model=None, reuse_pos=True):
        """Check if all-to-all connectivity (incl. #syn/conn, if props_model provided)"""
        src_sel = nodes[0].ids({"synapse_class": syn_class})
        if reuse_pos:
            # When reusing positions, consider only target node with existing synapses; for all others, no rewiring is possible!!
            tgt_sel = np.unique(ref["@target_node"])
        else:
            # Otherwise, consider all target nodes
            tgt_sel = nodes[1].ids()
        cnt_mat = np.zeros((len(src_sel), len(tgt_sel)), dtype=int)
        if props_model is not None:
            cnt_mat_model = np.zeros((len(src_sel), len(tgt_sel)), dtype=int)
        eq_mat = np.zeros((len(src_sel), len(tgt_sel)), dtype=bool)
        for sidx, s in enumerate(src_sel):
            for tidx, t in enumerate(tgt_sel):
                cnt_mat[sidx, tidx] = np.sum(
                    np.logical_and(res["@source_node"] == s, res["@target_node"] == t)
                )
                if props_model is not None:
                    cnt_mat_model[sidx, tidx] = props_model.draw(
                        prop_name="n_syn_per_conn",
                        src_type=nodes[0].get(s, properties="mtype"),
                        tgt_type=nodes[1].get(t, properties="mtype"),
                    )[0]
                eq_mat[sidx, tidx] = s == t
        assert np.sum(cnt_mat[eq_mat]) == 0, "ERROR: Autapses found!"
        assert np.all(cnt_mat[~eq_mat] > 0), "ERROR: All-to-all connectivity expected!"
        if props_model is not None:
            assert np.all(
                cnt_mat[~eq_mat] == cnt_mat_model[~eq_mat]
            ), "ERROR: Number of synapses per connection not consistent with model!"

    def check_randomization(ref, res, props_model):
        """Check model-based randomization of synaptic properties (w/o #syn/conn)"""
        for i in range(res.shape[0]):
            r = res.iloc[i]
            if np.any(np.all(r == ref, 1)):
                continue  # Original synapse, i.e., not rewired
            src_type = nodes[0].get(int(r["@source_node"]), properties="mtype")
            tgt_type = nodes[1].get(int(r["@target_node"]), properties="mtype")
            for p in np.setdiff1d(props_model.get_prop_names(), "n_syn_per_conn"):
                assert (
                    r[p] == props_model.draw(prop_name=p, src_type=src_type, tgt_type=tgt_type)[0]
                )  # Assuming constant model!!

    def get_adj(edges_table, src_ids, tgt_ids):
        """Extract adjacency matrix from edges table"""
        conns = np.unique(edges_table[["@source_node", "@target_node"]], axis=0)
        adj_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=bool)
        for _s, _t in conns:
            adj_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = True
        return adj_mat

    def get_syn(edges_table, src_ids, tgt_ids):
        """Extract synaptome matrix from edges table"""
        conns, counts = np.unique(
            edges_table[["@source_node", "@target_node"]], axis=0, return_counts=True
        )
        syn_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=int)
        for (_s, _t), _c in zip(conns, counts):
            syn_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = _c
        return syn_mat

    def check_adj(res, edges_table, nodes, syn_class, compare_to="ref"):
        """Check adj. matrices (excl. tgt nodes w/o any original synapses; excl. autapses)"""
        _src = nodes[0].ids({"synapse_class": syn_class})
        _tgt = np.unique(edges_table["@target_node"])
        res_adj = get_adj(res, _src, _tgt)
        ref_adj = get_adj(edges_table, _src, _tgt)
        for _sidx, _s in enumerate(_src):
            for _tidx, _t in enumerate(_tgt):
                if _s == _t:  # Skip autapses
                    continue
                if compare_to == "ref":  # Compare to reference
                    assert res_adj[_sidx, _tidx] == ref_adj[_sidx, _tidx]
                elif compare_to == "inv_ref":  # Compare to inverse reference
                    assert res_adj[_sidx, _tidx] == ~ref_adj[_sidx, _tidx]
                else:  # Expect boolean value to compare to
                    assert isinstance(compare_to, bool)
                    assert res_adj[_sidx, _tidx] == compare_to

    for syn_class in ["EXC", "INH"]:
        # Case 1: Rewire connectivity with conn. prob. p=1.0 but no target selection => Nothing should be changed
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 0.0

        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=True,
            reuse_conns=True,
            gen_method=None,
            props_model_spec=None,
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert res.equals(
            edges_table[res.columns].reset_index(drop=True)
        ), "ERROR: Edges table has changed!"

        # Case 2: Rewire connectivity with conn. prob. p=0.0 (no connectivity)
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb0p0.json")
        pct = 100.0

        ## (a) Keeping indegree => NOT POSSIBLE with p=0.0
        with pytest.raises(
            AssertionError,
            match=re.escape("Keeping indegree not possible since connection probability zero!"),
        ):
            with EdgeWriter(None, existing_edges=edges_table.copy()) as writer:
                manipulation(nodes, writer).apply(
                    tgt_ids,
                    syn_class=syn_class,
                    prob_model_spec={"file": prob_model_file},
                    delay_model_spec={"file": delay_model_file},
                    sel_src=None,
                    sel_dest=None,
                    amount_pct=pct,
                    keep_indegree=True,
                    reuse_conns=True,
                    gen_method=None,
                    props_model_spec=None,
                    pos_map_file=None,
                )

        ## (b) Not keeping indegree => All selected (EXC or INH) connections should be removed
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            gen_method="sample",
            props_model_spec=None,
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_removed(edges_table, res, nodes, syn_class)

        # Case 3: Rewire connectivity with conn. prob. p=1.0 (full connectivity, w/o autapses)
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 100.0

        ## (a) Keeping indegree & reusing connections
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=True,
            reuse_conns=True,
            gen_method=None,
            props_model_spec=None,
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert np.array_equal(edges_table.shape, res.shape), "ERROR: Number of synapses mismatch!"

        col_sel = np.setdiff1d(edges_table.columns, ["@source_node", "@target_node", "delay"])
        assert edges_table[col_sel].equals(res[col_sel]), "ERROR: Synapse properties mismatch!"

        check_nsyn(edges_table, res)  # Check reuse_conns option
        check_indegree(edges_table, res, nodes)  # Check keep_indegree option
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged

        ## (b) Keeping indegree & w/o reusing connections
        writer = EdgeWriter(None, existing_edges=edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=True,
            reuse_conns=False,
            gen_method="sample",
            props_model_spec=None,
            pos_map_file=None,
        )
        res = writer.to_pandas()

        # Selected columns excluding unused efferent/spine_length properties, which are not duplicated (syn_pos_mode="reuse" by default) and set to zero
        eff_props = [_col for _col in edges_table.columns if "efferent_" in _col]
        cols_unused_reuse = eff_props + ["spine_length"]
        col_sel = np.setdiff1d(
            edges_table.columns, ["@source_node", "@target_node", "delay"] + cols_unused_reuse
        )

        check_indegree(edges_table, res, nodes)  # Check keep_indegree option
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check sampling method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns

        ## (c) W/o keeping indegree & w/o reusing connections ("sample" method)
        with EdgeWriter(None, existing_edges=edges_table.copy()) as writer:
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                gen_method="sample",
                props_model_spec=None,
                pos_map_file=None,
            )
            res = writer.to_pandas()

        check_all_to_all(edges_table, res, nodes, syn_class)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check sampling method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns

        ## (d) W/o keeping indegree & w/o reusing connections ("sample" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            writer = EdgeWriter(None, edges_table_split)
            manipulation(nodes, writer, i_split, len(split_ids_list)).apply(
                split_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                gen_method="sample",
                props_model_spec=None,
                pos_map_file=None,
            )
            res_list.append(writer.to_pandas())
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(edges_table, res, nodes, syn_class)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check sampling method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns

        ## (e) W/o keeping indegree & w/o reusing connections ("randomize" method)
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()

        check_all_to_all(
            edges_table, res, nodes, syn_class, props_model
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check randomization method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns

        ## (f) W/o keeping indegree & w/o reusing connections ("randomize" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            writer = EdgeWriter(None, edges_table_split)
            manipulation(nodes, writer, i_split, len(split_ids_list)).apply(
                split_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                gen_method="randomize",
                props_model_spec={"file": props_model_file},
                pos_map_file=None,
            )
            res_list.append(writer.to_pandas())
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(
            edges_table, res, nodes, syn_class, props_model
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check randomization method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns

        # Case 4: Keeping indegree & add/delete only => NOT POSSIBLE!
        pct = 100.0
        for _mode in ["add_only", "delete_only"]:
            with pytest.raises(
                AssertionError,
                match=re.escape(f'"keep_indegree" not supported for rewire mode "{_mode}"!'),
            ):
                with EdgeWriter(None, existing_edges=edges_table.copy()) as writer:
                    manipulation(nodes, writer).apply(
                        tgt_ids,
                        syn_class=syn_class,
                        prob_model_spec={"file": prob_model_file},
                        delay_model_spec={"file": delay_model_file},
                        sel_src=None,
                        sel_dest=None,
                        amount_pct=pct,
                        keep_indegree=True,
                        rewire_mode=_mode,
                        reuse_conns=False,
                        gen_method=None,
                        props_model_spec=None,
                        pos_map_file=None,
                    )

        # Case 5: Deterministic rewiring (+ keep_conns) with empty adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMatEmpty.json")

        ## (a) Default rewire mode => All (EXC or INH) connections should be removed
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode=None,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_removed(edges_table, res, nodes, syn_class)

        ## (b) Rewire mode "add_only" => All connections should be unchanged
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="add_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert edges_table.equals(res), "ERROR: Edges table mismatch!"

        ## (c) Rewire mode "delete_only" => All (EXC or INH) connections should be removed
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="delete_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_removed(edges_table, res, nodes, syn_class)

        # Case 6: Deterministic rewiring (+ keep_conns) with full adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMatFull.json")

        ## (a) Default rewire mode => All-to-all (EXC or INH) connections should exist
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode=None,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_to_all(edges_table, res, nodes, syn_class)
        for _sclass in ["EXC", "INH"]:
            # With "keep_conns", all existing synapses should be unchanged
            check_unchanged(edges_table, res, nodes, _sclass)

        ## (b) Rewire mode "add_only" => All-to-all (EXC or INH) connections should exist
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="add_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_to_all(edges_table, res, nodes, syn_class)
        for _sclass in ["EXC", "INH"]:
            # With "keep_conns", all existing synapses should be unchanged
            check_unchanged(edges_table, res, nodes, _sclass)

        ## (c) Rewire mode "delete_only" => All connections should be unchanged
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="delete_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert edges_table.equals(res), "ERROR: Edges table mismatch!"

        # Case 7: Deterministic rewiring (+ keep_conns) with actual adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMat.json")

        ## All rewire modes => All connections should be unchanged
        for _mode in [None, "add_only", "delete_only"]:
            writer = EdgeWriter(None, edges_table.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                keep_conns=True,
                rewire_mode=_mode,
                gen_method="randomize",
                props_model_spec={"file": props_model_file},
                pos_map_file=None,
            )
            res = writer.to_pandas()
            assert edges_table.equals(res), "ERROR: Edges table mismatch!"

        # Case 8: Deterministic rewiring (+ keep_conns) with actual inverted adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMatInv.json")

        ## (a) Default rewire mode => Connectivity should be inverted (EXC or INH)
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode=None,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_adj(res, edges_table, nodes, syn_class, compare_to="inv_ref")

        ## (b) Rewire mode "add_only" => All-to-all (EXC or INH) connections should exist
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="add_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_to_all(edges_table, res, nodes, syn_class)
        for _sclass in ["EXC", "INH"]:
            # With "keep_conns", all existing synapses should be unchanged
            check_unchanged(edges_table, res, nodes, _sclass)

        ## (c) Rewire mode "delete_only" => All (EXC or INH) connections should be removed
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=True,
            rewire_mode="delete_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_removed(edges_table, res, nodes, syn_class)

        # Case 9: Deterministic rewiring (w/o keep/reuse_conns) with full adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMatFull.json")

        ## (a) Default rewire mode => All-to-all (EXC or INH) connections should exist
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            rewire_mode=None,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_to_all(edges_table, res, nodes, syn_class)
        check_adj(res, edges_table, nodes, syn_class, compare_to=True)

        ## (b) Rewire mode "add_only" => All-to-all (EXC or INH) connections should exist
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            rewire_mode="add_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        check_all_to_all(edges_table, res, nodes, syn_class)
        check_adj(res, edges_table, nodes, syn_class, compare_to=True)

        ## (c) Rewire mode "delete_only" => Connectivity should be unchanged, but connections re-drawn
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            rewire_mode="delete_only",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
        check_adj(res, edges_table, nodes, syn_class, compare_to="ref")

        # Case 10: Deterministic rewiring (w/o keep/reuse_conns) with actual adj. matrix
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMat.json")

        ## All rewire modes => Connectivity should be unchanged, but connections re-drawn
        for _mode in [None, "add_only", "delete_only"]:
            writer = EdgeWriter(None, edges_table.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                keep_conns=False,
                rewire_mode=_mode,
                gen_method="randomize",
                props_model_spec={"file": props_model_file},
                pos_map_file=None,
            )
            res = writer.to_pandas()
            assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
            check_adj(res, edges_table, nodes, syn_class, compare_to="ref")

        # Case 11: Synapse placement using morphologies (w/o reusing positions)
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 100.0

        tgt_morph = MorphHelper(
            nodes[1].config.get("morphologies_dir"),
            nodes[1],
            nodes[1].config.get("alternate_morphologies"),
        )
        get_tgt_morph = lambda node_id: tgt_morph.get(
            node_id, transform=True, extension="swc"
        )  # Access function (incl. transformation!), using specified format (swc/h5/...)

        morphio.set_maximum_warnings(1)  # Suppress repeated "Warning: zero diameter in file"

        eff_props = [_col for _col in edges_table.columns if "efferent_" in _col]
        seg_props = [_col for _col in edges_table.columns if "_segment" in _col]
        surf_props = [_col for _col in edges_table.columns if "_surface" in _col]
        cols_unused_all = (
            eff_props + seg_props + surf_props + ["spine_length"]
        )  # W/o reusing positions, more columns will be unused
        aff_props = [_col for _col in edges_table.columns if "afferent_" in _col]
        col_sel = np.setdiff1d(
            edges_table.columns,
            ["@source_node", "@target_node", "delay"] + cols_unused_all + aff_props,
        )

        def check_syn_pos(res, nodes, syn_class):
            """Check synapse position/type consistency"""
            for i in range(res.shape[0]):
                syn_cl = (
                    nodes[0]
                    .get(res.iloc[i]["@source_node"], properties="synapse_class")
                    .to_numpy()[0]
                )
                if syn_cl != syn_class:
                    continue

                # Check synapse position/type consistency
                syn_pos = res.iloc[i][
                    ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
                ]
                sec_id, sec_pos, sec_type = res.iloc[i][
                    ["afferent_section_id", "afferent_section_pos", "afferent_section_type"]
                ]
                if sec_id == 0:  # Soma section
                    assert sec_pos == 0.0 and sec_type == SEC_SOMA, "ERROR: Soma section error!"
                    assert np.all(
                        np.isclose(
                            syn_pos.to_numpy(),
                            nodes[1].positions(res.iloc[i]["@target_node"]).to_numpy(),
                        )
                    ), "ERROR: Soma position error!"
                else:
                    morph = get_tgt_morph(int(res.iloc[i]["@target_node"]))
                    sec_id = int(
                        sec_id - 1
                    )  # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
                    assert (
                        sec_type == SEC_TYPE_MAP[morph.section(sec_id).type]
                    ), "ERROR: Section type mismatch!"
                    assert np.all(
                        np.isclose(
                            nm.morphmath.path_fraction_point(morph.section(sec_id).points, sec_pos),
                            syn_pos,
                        )
                    ), "ERROR: Section position error!"

        ## (a) W/o keeping indegree & w/o reusing connections ("sample" method)
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            syn_pos_mode="random",
            gen_method="sample",
            props_model_spec=None,
            pos_map_file=None,
        )
        res = writer.to_pandas()

        check_all_to_all(
            edges_table, res, nodes, syn_class, reuse_pos=False
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check sampling method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        check_syn_pos(res, nodes, syn_class)  # Check synapse positions

        ## (b) W/o keeping indegree & w/o reusing connections ("sample" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            writer = EdgeWriter(None, edges_table_split)
            manipulation(nodes, writer, i_split, len(split_ids_list)).apply(
                split_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                syn_pos_mode="random",
                gen_method="sample",
                props_model_spec=None,
                pos_map_file=None,
            )
            res_list.append(writer.to_pandas())
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(
            edges_table, res, nodes, syn_class, reuse_pos=False
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check sampling method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        check_syn_pos(res, nodes, syn_class)  # Check synapse positions

        ## (c) W/o keeping indegree & w/o reusing connections ("randomize" method)
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            syn_pos_mode="random",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()

        check_all_to_all(
            edges_table, res, nodes, syn_class, props_model, reuse_pos=False
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check randomization method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        check_syn_pos(res, nodes, syn_class)  # Check synapse positions

        ## (d) W/o keeping indegree & w/o reusing connections ("randomize" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            writer = EdgeWriter(None, edges_table_split)
            manipulation(nodes, writer, i_split, len(split_ids_list)).apply(
                split_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                syn_pos_mode="random",
                gen_method="randomize",
                props_model_spec={"file": props_model_file},
                pos_map_file=None,
            )
            res_list.append(writer.to_pandas())
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(
            edges_table, res, nodes, syn_class, props_model, reuse_pos=False
        )  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check randomization method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        check_syn_pos(res, nodes, syn_class)  # Check synapse positions

        # Case 12: Wiring an empty connectome from scratch
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 100.0

        ## (a) Reusing positions => Not possible, so output must be empty
        writer = EdgeWriter(None, edges_table_empty.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            syn_pos_mode="reuse",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert edges_table_empty.equals(res), "ERROR: Edges not empty!"

        ## (b) Keeping indegree => Nothing to wire, so output must be empty
        writer = EdgeWriter(None, edges_table_empty.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=True,
            reuse_conns=False,
            syn_pos_mode="random",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()
        assert edges_table_empty.equals(res), "ERROR: Edges not empty!"

        ## (c) Sampling method => ERROR since no synapses to sample values from
        with pytest.raises(
            AssertionError,
            match=re.escape(
                f"No synapses to sample connection property values for target neuron {tgt_ids[0]} from!"
            ),
        ):
            writer = EdgeWriter(None, edges_table_empty.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                sel_src=None,
                sel_dest=None,
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                syn_pos_mode="random",
                gen_method="sample",
                props_model_spec=None,
                pos_map_file=None,
            )

        ## (d) Wiring from scratch => All-to-all connectivity
        writer = EdgeWriter(None, edges_table_empty.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            syn_pos_mode="random",
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )
        res = writer.to_pandas()

        check_all_to_all(
            edges_table, res, nodes, syn_class, props_model, reuse_pos=False
        )  # Check all-to-all connectivity
        check_randomization(edges_table, res, props_model)  # Check randomization method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        check_syn_pos(res, nodes, syn_class)  # Check synapse positions

        # Case 14: Deterministic rewiring with actual adj. matrix and synaptome provided
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMat.json")
        syn_model_file = os.path.join(TEST_DATA_DIR, "model_config__SynMatRnd.json")
        syn_model = model_types.AbstractModel.init_model({"file": syn_model_file})

        ## (a) Provide synaptome model, #syn/conn from ConnPropsModel ignored
        #      => Adjacency should be unchanged, but #syn/conn from synaptome
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            gen_method="randomize",
            props_model_spec={"file": props_model_file},
            nsynconn_model_spec={"file": syn_model_file},
        )
        res = writer.to_pandas()
        assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
        check_adj(res, edges_table, nodes, syn_class, compare_to="ref")
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns
        src_ids_sel = nodes[0].ids({"synapse_class": syn_class})
        res_syn = get_syn(res, src_ids_sel, tgt_ids)  # synaptome
        assert_array_equal(res_syn, syn_model.apply(src_nid=src_ids_sel, tgt_nid=tgt_ids))

        ## (b) Provide synaptome model & ConnPropsModel w/o #syn/conn
        #      => Adjacency should be unchanged, but #syn/conn from synaptome
        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            gen_method="randomize",
            props_model_spec={"file": props_model_file2},
            nsynconn_model_spec={"file": syn_model_file},
        )
        res = writer.to_pandas()
        assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
        check_adj(res, edges_table, nodes, syn_class, compare_to="ref")
        check_zero(res, cols_unused_reuse, nodes, syn_class)  # Check unused columns
        src_ids_sel = nodes[0].ids({"synapse_class": syn_class})
        res_syn = get_syn(res, src_ids_sel, tgt_ids)  # synaptome
        assert_array_equal(res_syn, syn_model.apply(src_nid=src_ids_sel, tgt_nid=tgt_ids))

        ## (c) Provide ConnPropsModel w/o #syn/conn, but no synaptome => Assertion error
        with pytest.raises(
            AssertionError,
            match=re.escape("#Syn/conn model required when using a properties model w/o nsynconn!"),
        ):
            writer = EdgeWriter(None, edges_table.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                keep_conns=False,
                gen_method="randomize",
                props_model_spec={"file": props_model_file2},
            )

        ## (d) Provide empty synaptome model => Assertion error
        with pytest.raises(AssertionError, match=re.escape('"n_syn" must be at least 1!')):
            syn_model_file = os.path.join(TEST_DATA_DIR, "model_config__SynMatEmpty.json")
            writer = EdgeWriter(None, edges_table.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                keep_conns=False,
                gen_method="randomize",
                props_model_spec={"file": props_model_file},
                nsynconn_model_spec={"file": syn_model_file},
            )

        # Case 15: Deterministic rewiring with actual adj. matrix and new synapse positions externally provided
        pct = 100.0
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__AdjMat.json")
        syn_pos_file = os.path.join(TEST_DATA_DIR, "model_config__SynPosTable.json")
        syn_pos = model_types.AbstractModel.init_model({"file": syn_pos_file})

        def check_pos(syn_pos, res, nodes, syn_class, ref_count=None):
            """Check synapse positions loaded from external table."""
            src_ids_sel = nodes[0].ids({"synapse_class": syn_class})
            res_conns, res_counts = np.unique(
                res.loc[
                    np.isin(res["@source_node"], src_ids_sel), ["@source_node", "@target_node"]
                ],
                axis=0,
                return_counts=True,
            )
            if ref_count is not None:
                assert np.all(res_counts == ref_count), "ERROR: Ref synapse count mismatch!"
            aff_props = [f"afferent_section_{_p}" for _p in ["id", "pos", "type"]] + [
                f"afferent_center_{_p}" for _p in ["x", "y", "z"]
            ]
            ref_pos = pd.concat(
                [
                    syn_pos.apply(
                        src_nid=_conn[0], tgt_nid=_conn[1], num_sel=ref_count, prop_names=aff_props
                    )
                    for _conn in res_conns
                ]
            )
            res_pos = pd.concat(
                [
                    res.loc[
                        np.logical_and(
                            res["@source_node"] == _conn[0], res["@target_node"] == _conn[1]
                        ),
                        aff_props,
                    ]
                    for _conn in res_conns
                ]
            )
            assert_array_equal(res_pos.to_numpy(), ref_pos.to_numpy())

        # (a) Generate 10 synapses per connection => Assertion error since not enough positions provided
        syn_model_file = os.path.join(TEST_DATA_DIR, "model_config__SynMatTen.json")

        with pytest.raises(ValueError):
            writer = EdgeWriter(None, edges_table.copy())
            manipulation(nodes, writer).apply(
                tgt_ids,
                syn_class=syn_class,
                prob_model_spec={"file": prob_model_file},
                delay_model_spec={"file": delay_model_file},
                amount_pct=pct,
                keep_indegree=False,
                reuse_conns=False,
                keep_conns=False,
                syn_pos_mode="external",
                gen_method="randomize",
                props_model_spec={"file": props_model_file2},
                nsynconn_model_spec={"file": syn_model_file},
                syn_pos_model_spec={"file": syn_pos_file},
            )

        # (b) Only generate 1 synapse per connection => First pos from table must be loaded
        syn_model_file = os.path.join(TEST_DATA_DIR, "model_config__SynMatOne.json")
        syn_model = model_types.AbstractModel.init_model({"file": syn_model_file})

        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            syn_pos_mode="external",
            gen_method="randomize",
            props_model_spec={"file": props_model_file2},
            nsynconn_model_spec={"file": syn_model_file},
            syn_pos_model_spec={"file": syn_pos_file},
        )
        res = writer.to_pandas()
        assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
        check_adj(res, edges_table, nodes, syn_class, compare_to="ref")
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        src_ids_sel = nodes[0].ids({"synapse_class": syn_class})
        res_syn = get_syn(res, src_ids_sel, tgt_ids)  # synaptome
        assert res_syn.max() == 1
        assert_array_equal(res_syn, syn_model.apply(src_nid=src_ids_sel, tgt_nid=tgt_ids))
        check_pos(syn_pos, res, nodes, syn_class, ref_count=1)

        # (c) Use exact synaptome => Full pos table must be loaded
        syn_model_file = os.path.join(TEST_DATA_DIR, "model_config__SynMat.json")
        syn_model = model_types.AbstractModel.init_model({"file": syn_model_file})

        writer = EdgeWriter(None, edges_table.copy())
        manipulation(nodes, writer).apply(
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            keep_conns=False,
            syn_pos_mode="external",
            gen_method="randomize",
            props_model_spec={"file": props_model_file2},
            nsynconn_model_spec={"file": syn_model_file},
            syn_pos_model_spec={"file": syn_pos_file},
        )
        res = writer.to_pandas()
        assert not edges_table.equals(res), "ERROR: Edges tables should not be equal!"
        check_adj(res, edges_table, nodes, syn_class, compare_to="ref")
        check_zero(res, cols_unused_all, nodes, syn_class)  # Check unused columns
        src_ids_sel = nodes[0].ids({"synapse_class": syn_class})
        res_syn = get_syn(res, src_ids_sel, tgt_ids)  # synaptome
        assert_array_equal(res_syn, syn_model.apply(src_nid=src_ids_sel, tgt_nid=tgt_ids))
        check_pos(syn_pos, res, nodes, syn_class)
