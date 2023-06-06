import os

import numpy as np
import pandas as pd
import pytest

from bluepysnap.morph import MorphHelper
from bluepysnap import Circuit
from libsonata import Selection
import neurom as nm

from utils import TEST_DATA_DIR
from connectome_manipulator.model_building import model_types
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


@pytest.fixture
def manipulation():
    Manipulation.destroy_instances()
    m = Manipulation.get("conn_wiring")
    return m


def test_apply(manipulation):
    c = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    edges = c.edges[c.edges.population_names[0]]
    nodes = [edges.source, edges.target]
    src_mtypes = nodes[0].property_values("mtype")
    tgt_mtypes = nodes[1].property_values("mtype")
    src_ids = nodes[0].ids()
    tgt_ids = nodes[1].ids()
    edges_table = edges.afferent_edges(tgt_ids, properties=edges.property_names)
    edges_table_empty = edges_table.loc[[]].copy()

    required_properties = [
        "@source_node",
        "@target_node",
        "afferent_center_x",
        "afferent_center_y",
        "afferent_center_z",
        "afferent_section_pos",
        "afferent_section_id",
        "afferent_section_type",
        "syn_type_id",
    ]

    morph_dir = nodes[1].config["morphologies_dir"]
    tgt_morph = MorphHelper(
        morph_dir,
        nodes[1],
        {
            "h5v1": os.path.join(morph_dir, "h5v1"),
            "neurolucida-asc": os.path.join(morph_dir, "ascii"),
        },
    )
    get_tgt_morph = lambda node_id: tgt_morph.get(
        node_id, transform=True, extension="swc"
    )  # Access function (incl. transformation!), using specified format (swc/h5/...)

    id_selection = Selection(tgt_ids).ranges
    aux_dict = {"split_ids": tgt_ids, "id_selection": id_selection}
    n_syn_conn = 2
    nsynconn_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__NSynPerConn{n_syn_conn}.json"
    )  # Model with exactly <n_syn_conn> syn/conn (constant) fo all pathways
    nsynconn_model = model_types.AbstractModel.model_from_file(nsynconn_model_file)
    delay_model_file = os.path.join(
        TEST_DATA_DIR, f"model_config__DistDepDelay.json"
    )  # Deterministic delay model w/o variation
    delay_model = model_types.AbstractModel.model_from_file(delay_model_file)
    pct = 100.0

    # Case 1: Check connectivity with conn. prob. p=0.0 (no connectivity)
    prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb0p0.json")
    prob_model = model_types.AbstractModel.model_from_file(prob_model_file)

    ## (a) Empty edges table
    res = manipulation(nodes).apply(
        edges_table_empty,
        tgt_ids,
        aux_dict,
        amount_pct=pct,
        prob_model_spec={"file": prob_model_file},
        nsynconn_model_spec={"file": nsynconn_model_file},
    )
    assert res.equals(
        edges_table_empty
    ), "ERROR: Existing edges table changed!"  # Check if unchanged

    ## (b) Edges already existing
    res = manipulation(nodes).apply(
        edges_table,
        tgt_ids,
        aux_dict,
        amount_pct=pct,
        prob_model_spec={"file": prob_model_file},
        nsynconn_model_spec={"file": nsynconn_model_file},
    )
    assert res.equals(edges_table), "ERROR: Existing edges table changed!"  # Check if unchanged

    ## (c) Standalone wiring per pathway
    pathway_nodes = nodes[0]
    assert nodes[0] is nodes[1]
    pathway_models = []
    for pre_mt in src_mtypes:
        for post_mt in tgt_mtypes:
            pathway_models.append(
                {
                    "pre": pre_mt,
                    "post": post_mt,
                    "prob_model": prob_model,
                    "nsynconn_model": nsynconn_model,
                    "delay_model": None,
                }
            )
    res = manipulation.connectome_wiring_per_pathway(pathway_nodes, pathway_models, seed=0)
    assert res.size == 0, "ERROR: Connectome should be empty!"
    assert np.all(np.isin(required_properties, res.columns)), "ERROR: Synapse properties missing!"

    # Case 2: Check connectivity with conn. prob. p=1.0 (full connectivity, w/o autapses)
    prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb1p0.json")
    prob_model = model_types.AbstractModel.model_from_file(prob_model_file)

    ## (a) Empty edges table
    res = manipulation(nodes).apply(
        edges_table_empty,
        tgt_ids,
        aux_dict,
        amount_pct=pct,
        prob_model_spec={"file": prob_model_file},
        nsynconn_model_spec={"file": nsynconn_model_file},
    )
    assert (
        res.shape[0]
        == (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * n_syn_conn
    ), "ERROR: Wrong number of synapses!"  # Check #synapses
    assert np.all(np.isin(required_properties, res.keys())), "ERROR: Synapse properties missing!"
    assert np.all(
        np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)[1]
        == n_syn_conn
    ), "ERROR: Wrong #syn/conn!"  # Check #synapses/connection

    for i in range(res.shape[0]):
        # Check synapse class consistency
        syn_cl = nodes[0].get(res.iloc[i]["@source_node"], properties="synapse_class").to_numpy()[0]
        if syn_cl == "EXC":
            assert res.iloc[i]["syn_type_id"] >= 100, "ERROR: Wrong EXC type ID!"
        elif syn_cl == "INH":
            assert res.iloc[i]["syn_type_id"] < 100, "ERROR: Wrong INH type ID!"
        else:
            assert False, "Synapse class unknown!"

        # Check synapse position consistency
        syn_pos = res.iloc[i][["afferent_center_x", "afferent_center_y", "afferent_center_z"]]
        sec_id, sec_pos, sec_type = res.iloc[i][
            ["afferent_section_id", "afferent_section_pos", "afferent_section_type"]
        ]
        if sec_id == 0:  # Soma section
            assert sec_pos == 0.0 and sec_type == 0, "ERROR: Soma section error!"
            assert np.all(
                np.isclose(
                    syn_pos.to_numpy(), nodes[1].positions(res.iloc[i]["@target_node"]).to_numpy()
                )
            ), "ERROR: Soma position error!"
        else:
            morph = get_tgt_morph(int(res.iloc[i]["@target_node"]))
            sec_id = int(
                sec_id - 1
            )  # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
            assert sec_type == int(morph.section(sec_id).type), "ERROR: Section type mismatch!"
            assert np.all(
                np.isclose(
                    nm.morphmath.path_fraction_point(morph.section(sec_id).points, sec_pos), syn_pos
                )
            ), "ERROR: Section position error!"

    ## (b) Edges already existing
    res = manipulation(nodes).apply(
        edges_table,
        tgt_ids,
        aux_dict,
        amount_pct=pct,
        prob_model_spec={"file": prob_model_file},
        nsynconn_model_spec={"file": nsynconn_model_file},
    )
    assert (
        res.shape[0]
        == edges_table.shape[0]
        + (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * n_syn_conn
    ), "ERROR: Wrong number of synapses!"
    syn_props = list(res.keys())
    assert np.all(
        [
            np.sum(np.all(edges_table.iloc[i][syn_props] == res[syn_props], 1)) == 1
            for i in range(edges_table.shape[0])
        ]
    ), "ERROR: Existing synapses changed!"  # Check if all existing synapses still exist exactly once
    assert np.all(np.isin(required_properties, syn_props)), "ERROR: Synapse properties missing!"

    ## (c) Standalone wiring per pathway
    pathway_models = []
    for pre_mt in src_mtypes:
        for post_mt in tgt_mtypes:
            pathway_models.append(
                {
                    "pre": pre_mt,
                    "post": post_mt,
                    "prob_model": prob_model,
                    "nsynconn_model": nsynconn_model,
                    "delay_model": None,
                }
            )
    res = manipulation.connectome_wiring_per_pathway(
        pathway_nodes, pathway_models, seed=0, morph_ext="swc"
    )
    assert (
        res.shape[0]
        == (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * n_syn_conn
    ), "ERROR: Wrong number of synapses!"  # Check #synapses
    assert np.all(np.isin(required_properties, res.columns)), "ERROR: Synapse properties missing!"
    assert np.all(
        np.unique(res[["@source_node", "@target_node"]], axis=0, return_counts=True)[1]
        == n_syn_conn
    ), "ERROR: Wrong #syn/conn!"  # Check #synapses/connection

    # Case 3: Check pct
    for pct in np.linspace(0, 100, 6):
        res = manipulation(nodes).apply(
            edges_table_empty,
            tgt_ids,
            aux_dict,
            amount_pct=pct.tolist(),
            prob_model_spec={"file": prob_model_file},
            nsynconn_model_spec={"file": nsynconn_model_file},
        )
        assert (
            res.shape[0]
            == (
                len(src_ids) * len(tgt_ids) * pct / 100
                - len(np.intersect1d(src_ids, tgt_ids)) * pct / 100
            )
            * n_syn_conn
        ), f"ERROR: Wrong number of synapses! for pct ({pct})"  # Check #synapses

    # Case 4: Check src/tgt_sel
    pct = 100.0

    ## (a) Per synapse class
    for src_class in ["EXC", "INH"]:
        for tgt_class in ["EXC", "INH"]:
            sel_src = {"synapse_class": src_class}
            sel_dest = {"synapse_class": tgt_class}
            res = manipulation(nodes).apply(
                edges_table_empty,
                tgt_ids,
                aux_dict,
                sel_src=sel_src,
                sel_dest=sel_dest,
                amount_pct=pct,
                prob_model_spec={"file": prob_model_file},
                nsynconn_model_spec={"file": nsynconn_model_file},
            )
            assert np.all(
                np.isin(res["@source_node"], nodes[0].ids(sel_src))
            ), "ERROR: Source selection error!"
            assert np.all(
                np.isin(res["@target_node"], nodes[1].ids(sel_dest))
            ), "ERROR: Target selection error!"
            assert (
                res.shape[0]
                == (
                    len(nodes[0].ids(sel_src)) * len(nodes[1].ids(sel_dest))
                    - len(np.intersect1d(nodes[0].ids(sel_src), nodes[1].ids(sel_dest)))
                )
                * n_syn_conn
            ), "ERROR: Wrong number of synapses!"  # Check #synapses

    ## (b) Per mtype
    for src_mt in src_mtypes:
        for tgt_mt in tgt_mtypes:
            sel_src = {"mtype": src_mt}
            sel_dest = {"mtype": tgt_mt}

            ### Integrated wiring
            res = manipulation(nodes).apply(
                edges_table_empty,
                tgt_ids,
                aux_dict,
                sel_src=sel_src,
                sel_dest=sel_dest,
                amount_pct=pct,
                prob_model_spec={"file": prob_model_file},
                nsynconn_model_spec={"file": nsynconn_model_file},
            )
            assert np.all(
                np.isin(res["@source_node"], nodes[0].ids(sel_src))
            ), "ERROR: Source selection error!"
            assert np.all(
                np.isin(res["@target_node"], nodes[1].ids(sel_dest))
            ), "ERROR: Target selection error!"
            assert (
                res.shape[0]
                == (
                    len(nodes[0].ids(sel_src)) * len(nodes[1].ids(sel_dest))
                    - len(np.intersect1d(nodes[0].ids(sel_src), nodes[1].ids(sel_dest)))
                )
                * n_syn_conn
            ), "ERROR: Wrong number of synapses!"  # Check #synapses

            ### Standalone wiring per pathway
            pathway_models = [
                {
                    "pre": src_mt,
                    "post": tgt_mt,
                    "prob_model": prob_model,
                    "nsynconn_model": nsynconn_model,
                    "delay_model": None,
                }
            ]
            res = manipulation.connectome_wiring_per_pathway(
                pathway_nodes, pathway_models, seed=0, morph_ext="swc"
            )
            assert np.all(
                np.isin(res["@source_node"], nodes[0].ids(sel_src))
            ), "ERROR: Source selection error!"
            assert np.all(
                np.isin(res["@target_node"], nodes[1].ids(sel_dest))
            ), "ERROR: Target selection error!"
            assert (
                res.shape[0]
                == (
                    len(nodes[0].ids(sel_src)) * len(nodes[1].ids(sel_dest))
                    - len(np.intersect1d(nodes[0].ids(sel_src), nodes[1].ids(sel_dest)))
                )
                * n_syn_conn
            ), "ERROR: Wrong number of synapses!"  # Check #synapses

    # Case 5: Check block-based processing
    split_ids_list = [tgt_ids[: len(tgt_ids) >> 1], tgt_ids[len(tgt_ids) >> 1 :]]
    res_list = []
    for i_split, split_ids in enumerate(split_ids_list):
        id_selection = Selection(split_ids).ranges
        aux_dict_split = {
            "N_split": len(split_ids_list),
            "i_split": i_split,
            "split_ids": split_ids,
            "id_selection": id_selection,
        }
        print(split_ids)
        res_list.append(
            manipulation(nodes).apply(
                edges_table_empty,
                split_ids,
                aux_dict_split,
                amount_pct=pct,
                prob_model_spec={"file": prob_model_file},
                nsynconn_model_spec={"file": nsynconn_model_file},
            )
        )
    res = pd.concat(res_list, ignore_index=True)
    assert (
        res.shape[0]
        == (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * n_syn_conn
    ), "ERROR: Wrong number of synapses!"  # Check #synapses

    # Case 6: Check delays (from PRE neuron (soma) to POST synapse position)
    def check_delay(nodes, delay_model, res):
        for i in range(res.shape[0]):
            delay_offset = delay_model.get_param_dict()["delay_mean_coeff_a"]
            delay_scale = delay_model.get_param_dict()["delay_mean_coeff_b"]
            src_pos = nodes[0].positions(res.iloc[i]["@source_node"]).to_numpy()
            syn_pos = res.iloc[i][
                ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
            ].to_numpy()
            dist = np.sqrt(np.sum((src_pos - syn_pos) ** 2))
            delay = delay_scale * dist + delay_offset
            assert np.all(np.isclose(res.iloc[i]["delay"], delay)), "ERROR: Delay mismatch!"

    ## (a) Integrated wiring
    res = manipulation(nodes).apply(
        edges_table_empty,
        tgt_ids,
        aux_dict,
        amount_pct=pct,
        prob_model_spec={"file": prob_model_file},
        nsynconn_model_spec={"file": nsynconn_model_file},
        delay_model_spec={"file": delay_model_file},
    )
    check_delay(nodes, delay_model, res)

    ## (b) Standalone wiring per pathway
    pathway_models = []
    for pre_mt in src_mtypes:
        for post_mt in tgt_mtypes:
            pathway_models.append(
                {
                    "pre": pre_mt,
                    "post": post_mt,
                    "prob_model": prob_model,
                    "nsynconn_model": nsynconn_model,
                    "delay_model": delay_model,
                }
            )
    res = manipulation.connectome_wiring_per_pathway(
        pathway_nodes, pathway_models, seed=0, morph_ext="swc"
    )
    check_delay([pathway_nodes, pathway_nodes], delay_model, res)

    # Case 7: Check connectivity with conn. prob. p=0.1
    prob_model_file = os.path.join(TEST_DATA_DIR, "model_config__ConnProb0p1.json")
    prob_model = model_types.AbstractModel.model_from_file(prob_model_file)

    ## (a) Integrated wiring
    np.random.seed(0)
    syn_counts = []
    for rep in range(
        30
    ):  # Estimate synapse counts over N repetitions => May be increased if variation still to large
        res = manipulation(nodes).apply(
            edges_table_empty,
            tgt_ids,
            aux_dict,
            amount_pct=pct,
            prob_model_spec={"file": prob_model_file},
            nsynconn_model_spec={"file": nsynconn_model_file},
        )
        syn_counts.append(res.shape[0])
    assert np.std(syn_counts) > 0, "ERROR: No variability over repetitions!"
    assert np.isclose(
        np.mean(syn_counts),
        (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * 0.1 * n_syn_conn,
        atol=1.5,
    ), f"ERROR: Wrong number of synapses!"  # Accept tolerance of +/-1.5

    ## (b) Standalone wiring per pathway [larger variability expected, since wiring per pathway]
    pathway_models = []
    for pre_mt in src_mtypes:
        for post_mt in tgt_mtypes:
            pathway_models.append(
                {
                    "pre": pre_mt,
                    "post": post_mt,
                    "prob_model": prob_model,
                    "nsynconn_model": nsynconn_model,
                    "delay_model": None,
                }
            )
    syn_counts = []
    for rep in range(
        40
    ):  # Estimate synapse counts over N repetitions => May be increased if variation still to large
        res = manipulation.connectome_wiring_per_pathway(
            pathway_nodes, pathway_models, seed=rep, morph_ext="swc"
        )
        syn_counts.append(res.shape[0])
    assert np.std(syn_counts) > 0, "ERROR: No variability over repetitions!"
    assert np.isclose(
        np.mean(syn_counts),
        (len(src_ids) * len(tgt_ids) - len(np.intersect1d(src_ids, tgt_ids))) * 0.1 * n_syn_conn,
        atol=1.5,
    ), f"ERROR: Wrong number of synapses!"  # Accept tolerance of +/-1.5
