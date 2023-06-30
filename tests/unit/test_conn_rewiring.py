import os

import numpy as np
import pandas as pd

from bluepysnap import Circuit

import pytest
import re
from utils import TEST_DATA_DIR
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation
from connectome_manipulator.model_building import model_types


@pytest.fixture
def manipulation():
    Manipulation.destroy_instances()
    m = Manipulation.get("conn_rewiring")
    return m


def test_apply(manipulation):
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)

    delay_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__DistDepDelay.json"
    )  # Deterministic delay model w/o variation
    delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
    props_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__ConnProps.json"
    )  # Deterministic connection properties model w/o variation
    props_model = model_types.AbstractModel.model_from_file(props_model_file)

    assert not np.all(
        np.diff(edges_table["@target_node"]) >= 0
    ), "ERROR: Edges table assumed to be not sorted!"

    with pytest.raises(
        AssertionError, match=re.escape("Edges table must be ordered by @target_node!")
    ):
        res = manipulation(nodes).apply(
            edges_table.copy(),
            tgt_ids,
            syn_class="EXC",
            prob_model_spec=None,
            delay_model_spec={"file": delay_model_file},
        )

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
        ), "ERROR: Synapses per connection sampling error!"  # Check duplicate_sample (#syn/conn)
        assert np.all(
            [np.all(np.isin(np.unique(res[col]), np.unique(ref[col]))) for col in col_sel]
        ), "ERROR: Synapse properties sampling error!"  # Check duplicate_sample (w/o #syn/conn)

    def check_all_to_all(ref, res, nodes, props_model=None):
        """Check if all-to-all connectivity (incl. #syn/conn, if props_model provided)"""
        src_sel = nodes[0].ids({"synapse_class": syn_class})
        tgt_sel = np.unique(
            ref["@target_node"]
        )  # Consider only target node with existing synapses; for all others, no rewiring is possible!!
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

    for syn_class in ["EXC", "INH"]:
        # Case 1: Rewire connectivity with conn. prob. p=1.0 but no target selection => Nothing should be changed
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 0.0

        res = manipulation(nodes).apply(
            edges_table.copy(),
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
        assert res.equals(edges_table.reset_index(drop=True)), "ERROR: Edges table has changed!"

        # Case 2: Rewire connectivity with conn. prob. p=0.0 (no connectivity)
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb0p0.json")
        pct = 100.0

        ## (a) Keeping indegree => NOT POSSIBLE with p=0.0
        with pytest.raises(
            AssertionError,
            match=re.escape("Keeping indegree not possible since connection probability zero!"),
        ):
            res = manipulation(nodes).apply(
                edges_table.copy(),
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
        res = manipulation(nodes).apply(
            edges_table.copy(),
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            gen_method="duplicate_sample",
            props_model_spec=None,
            pos_map_file=None,
        )
        assert (
            edges_table[
                ~np.isin(edges_table["@source_node"], nodes[0].ids({"synapse_class": syn_class}))
            ]
            .reset_index(drop=True)
            .equals(res)
        ), "ERROR: Results table mismatch!"

        # Case 3: Rewire connectivity with conn. prob. p=1.0 (full connectivity, w/o autapses)
        prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
        pct = 100.0

        ## (a) Keeping indegree & reusing connections
        res = manipulation(nodes).apply(
            edges_table.copy(),
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
        assert np.array_equal(edges_table.shape, res.shape), "ERROR: Number of synapses mismatch!"

        col_sel = np.setdiff1d(edges_table.columns, ["@source_node", "@target_node", "delay"])
        assert edges_table[col_sel].equals(res[col_sel]), "ERROR: Synapse properties mismatch!"

        check_nsyn(edges_table, res)  # Check reuse_conns option
        check_indegree(edges_table, res, nodes)  # Check keep_indegree option
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged

        ## (b) Keeping indegree & w/o reusing connections
        res = manipulation(nodes).apply(
            edges_table.copy(),
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=True,
            reuse_conns=False,
            gen_method="duplicate_sample",
            props_model_spec=None,
            pos_map_file=None,
        )

        check_indegree(edges_table, res, nodes)  # Check keep_indegree option
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check duplicate_sample method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays

        ## (c) W/o keeping indegree & w/o reusing connections ("duplicate_sample" method)
        res = manipulation(nodes).apply(
            edges_table.copy(),
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            gen_method="duplicate_sample",
            props_model_spec=None,
            pos_map_file=None,
        )

        check_all_to_all(edges_table, res, nodes)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check duplicate_sample method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays

        ## (d) W/o keeping indegree & w/o reusing connections ("duplicate_sample" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            res_list.append(
                manipulation(nodes, i_split, len(split_ids_list)).apply(
                    edges_table_split,
                    split_ids,
                    syn_class=syn_class,
                    prob_model_spec={"file": prob_model_file},
                    delay_model_spec={"file": delay_model_file},
                    sel_src=None,
                    sel_dest=None,
                    amount_pct=pct,
                    keep_indegree=False,
                    reuse_conns=False,
                    gen_method="duplicate_sample",
                    props_model_spec=None,
                    pos_map_file=None,
                )
            )
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(edges_table, res, nodes)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_sampling(edges_table, res, col_sel)  # Check duplicate_sample method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays

        ## (e) W/o keeping indegree & w/o reusing connections ("duplicate_randomize" method)
        res = manipulation(nodes).apply(
            edges_table.copy(),
            tgt_ids,
            syn_class=syn_class,
            prob_model_spec={"file": prob_model_file},
            delay_model_spec={"file": delay_model_file},
            sel_src=None,
            sel_dest=None,
            amount_pct=pct,
            keep_indegree=False,
            reuse_conns=False,
            gen_method="duplicate_randomize",
            props_model_spec={"file": props_model_file},
            pos_map_file=None,
        )

        check_all_to_all(edges_table, res, nodes, props_model)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check duplicate_randomize method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays

        ## (f) W/o keeping indegree & w/o reusing connections ("duplicate_randomize" method & block-based processing)
        split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
        res_list = []
        for i_split, split_ids in enumerate(split_ids_list):
            edges_table_split = edges_table[np.isin(edges_table["@target_node"], split_ids)].copy()
            res_list.append(
                manipulation(nodes, i_split, len(split_ids_list)).apply(
                    edges_table_split,
                    split_ids,
                    syn_class=syn_class,
                    prob_model_spec={"file": prob_model_file},
                    delay_model_spec={"file": delay_model_file},
                    sel_src=None,
                    sel_dest=None,
                    amount_pct=pct,
                    keep_indegree=False,
                    reuse_conns=False,
                    gen_method="duplicate_randomize",
                    props_model_spec={"file": props_model_file},
                    pos_map_file=None,
                )
            )
        res = pd.concat(res_list, ignore_index=True)

        check_all_to_all(edges_table, res, nodes, props_model)  # Check all-to-all connectivity
        check_indegree(
            edges_table, res, nodes, check_not_equal=True
        )  # Check if keep_indegree changed
        check_unchanged(
            edges_table, res, nodes, syn_class
        )  # Check that non-selected connections unchanged
        check_randomization(edges_table, res, props_model)  # Check duplicate_randomize method
        check_delay(res, nodes, syn_class, delay_model)  # Check synaptic delays
