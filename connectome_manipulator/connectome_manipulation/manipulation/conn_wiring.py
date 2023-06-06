"""Manipulation name: conn_wiring

Description: Special case of connectome rewiring, which wires an empty connectome from scratch, or simply
adds connections to an existing connectome (edges table)!
Only specific properties like source/target node, afferent synapse positions, synapse type
(INH: 0, EXC: 100), and delay (optional) will be generated.
"""

import os
from datetime import datetime, timedelta

import libsonata
import neurom as nm
import numpy as np
import pandas as pd
import tqdm

from bluepysnap.morph import MorphHelper
from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import (
    get_node_ids,
    get_enumeration,
)
from connectome_manipulator.connectome_manipulation.manipulation import (
    MorphologyCachingManipulation,
)
from connectome_manipulator.model_building import model_types, conn_prob

# IDEAs for improvements:
#   Add model for synapse placement


class ConnectomeWiring(MorphologyCachingManipulation):
    """Special case of connectome rewiring

    Wires an empty connectome from scratch, or simply adds connections to an existing connectome (edges table)!
    Only specific properties like source/target node, afferent synapse positions, synapse type
    (INH: 0, EXC: 100), and delay (optional) will be generated.
    """

    SYNAPSE_PROPERTIES = [
        "@target_node",
        "@source_node",
        "afferent_section_id",
        "afferent_section_pos",
        "afferent_section_type",
        "afferent_center_x",
        "afferent_center_y",
        "afferent_center_z",
        "syn_type_id",
        "delay",
    ]
    PROPERTY_TYPES = {
        "@target_node": "int64",
        "@source_node": "int64",
        "afferent_section_id": "int32",
        "afferent_section_pos": "float32",
        "afferent_section_type": "int16",
        "afferent_center_x": "float32",
        "afferent_center_y": "float32",
        "afferent_center_z": "float32",
        "syn_type_id": "int16",
        "delay": "float32",
    }

    def __init__(self, nodes):
        """Initialize a ConnectomeWiring manipulator with a node set.

        The initialization will itself initialize the MorphHelper object. Split indices are passed
        via the apply function.
        """
        super().__init__(nodes)
        # Prepare to load target (dendritic) morphologies
        morph_dir = self.nodes[1].config["morphologies_dir"]
        self.tgt_morph = MorphHelper(
            morph_dir,
            self.nodes[1],
            {
                "h5v1": os.path.join(morph_dir, "h5v1"),
                "neurolucida-asc": os.path.join(morph_dir, "ascii"),
            },
        )

    @profiler.profileit(name="conn_wiring")
    def apply(
        self,
        edges_table,
        split_ids,
        aux_dict,
        sel_src=None,
        sel_dest=None,
        pos_map_file=None,
        amount_pct=100.0,
        morph_ext="swc",
        prob_model_spec=None,
        nsynconn_model_spec=None,
        delay_model_spec=None,
        pathway_specs=None,
        **kwargs,
    ):
        """Wiring (generation) of structural connections between pairs of neurons based on given conn. prob. model.

        => Only structural synapse properties will be set: PRE/POST neuron IDs, synapse positions, type, axonal delays
        => Model specs: A dict with model type/attributes or a dict with "file" key pointing to a model file can be passed
        """
        # pylint did not accept to drop kwargs, forced to add this
        assert len(kwargs) == 0
        if not prob_model_spec:
            prob_model_spec = {
                "model": "ConnProb1stOrderModel",
            }  # Default 1st-oder model
        if not nsynconn_model_spec:
            nsynconn_model_spec = {
                "model": "NSynConnModel",
            }  # Default #syn/conn model
        if not delay_model_spec:
            delay_model_spec = {
                "model": "LinDelayModel",
            }  # Default linear delay model
        for spec in (prob_model_spec, nsynconn_model_spec, delay_model_spec):
            # AbstractModel insists that "file" is the only key if present
            if "file" not in spec:
                spec["src_type_map"] = self.src_type_map
                spec["tgt_type_map"] = self.tgt_type_map
                spec["pathway_specs"] = pathway_specs
        # pylint: disable=arguments-differ, arguments-renamed
        log.log_assert(0.0 <= amount_pct <= 100.0, "amount_pct out of range!")
        if edges_table is None:
            edges_table = self._init_edges_table(
                with_delay=delay_model_spec is not None
            )  # Create empty edges table
        elif edges_table.shape[0] == 0:
            log.debug(
                f"Empty connectome with {edges_table.shape[1]} properties! Existing properties may be removed to match newly generated synapses."
            )
        else:
            log.debug(
                f"Initial connectome not empty ({edges_table.shape[0]} synapses, {edges_table.shape[1]} properties)! Connections will be added to existing connectome. Existing properties may be removed to match newly generated synapses."
            )
        with profiler.profileit(name="conn_wiring/setup"):
            # Intersect target nodes with split IDs and return if intersection is empty
            tgt_node_ids = get_node_ids(self.nodes[1], sel_dest, aux_dict["id_selection"])
            num_tgt_total = len(tgt_node_ids)
            if num_tgt_total == 0:  # Nothing to wire
                log.info("No target nodes selected, nothing to wire")
                return edges_table
            if amount_pct < 100:
                num_tgt = np.round(amount_pct * num_tgt_total / 100).astype(int)
                tgt_sel = np.random.permutation(
                    np.concatenate(
                        (np.full(num_tgt, True), np.full(num_tgt_total - num_tgt, False)), axis=None
                    )
                )
            else:
                num_tgt = num_tgt_total
                tgt_sel = np.full(num_tgt_total, True)
            if num_tgt == 0:  # Nothing to wire
                log.info("No target nodes selected, nothing to wire")
                return edges_table
            # Load connection probability model
            p_model = model_types.AbstractModel.init_model(prob_model_spec)
            log.debug(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')

            # Load #synapses/connection model
            nsynconn_model = model_types.AbstractModel.init_model(nsynconn_model_spec)
            log.debug(
                f'Loaded #synapses/connection model of type "{nsynconn_model.__class__.__name__}"'
            )

            # Load delay model (optional)
            if delay_model_spec is not None:
                delay_model = model_types.AbstractModel.init_model(delay_model_spec)
                log.debug(f'Loaded delay model of type "{delay_model.__class__.__name__}"')
            else:
                delay_model = None
                log.debug("No delay model provided")

            # Load position mapping model (optional) => [NOTE: SRC AND TGT NODES MUST BE INCLUDED WITHIN SAME POSITION MAPPING MODEL]
            _, pos_acc = conn_prob.load_pos_mapping_model(pos_map_file)
            if pos_acc is None:
                log.debug("No position mapping model provided")

            # Determine source/target nodes for wiring
            src_node_ids = get_node_ids(self.nodes[0], sel_src)
            src_class = self.nodes[0].get(src_node_ids, properties="synapse_class").to_numpy()
            src_mtypes = self.nodes[0].get(src_node_ids, properties="mtype").to_numpy()
            log.log_assert(len(src_node_ids) > 0, "No source nodes selected!")

            # Determine source/target nodes for wiring
            src_node_ids = get_node_ids(self.nodes[0], sel_src)
            src_class = self.nodes[0]._population.get_attribute(  # pylint: disable=protected-access
                "synapse_class", libsonata.Selection(src_node_ids)
            )
            src_mtypes = get_enumeration(self.nodes[0], "mtype", src_node_ids)
            log.log_assert(len(src_node_ids) > 0, "No source nodes selected!")

            tgt_node_ids = tgt_node_ids[tgt_sel]  # Select subset of neurons (keeping order)
            tgt_mtypes = get_enumeration(self.nodes[1], "mtype", tgt_node_ids)

            if pos_acc:
                # FIXME: this is going to be VERY SLOW!
                src_pos = conn_prob.get_neuron_positions(pos_acc, [src_node_ids])[
                    0
                ]  # Get neuron positions (incl. position mapping, if provided)
                tgt_pos = conn_prob.get_neuron_positions(pos_acc, [tgt_node_ids])[
                    0
                ]  # Get neuron positions (incl. position mapping, if provided)
            else:
                _src_pop = self.nodes[0]._population  # pylint: disable=protected-access
                _src_sel = libsonata.Selection(src_node_ids)
                src_pos = np.column_stack(
                    (
                        _src_pop.get_attribute("x", _src_sel),
                        _src_pop.get_attribute("y", _src_sel),
                        _src_pop.get_attribute("z", _src_sel),
                    )
                )
                _tgt_pop = self.nodes[1]._population  # pylint: disable=protected-access
                _tgt_sel = libsonata.Selection(tgt_node_ids)
                tgt_pos = np.column_stack(
                    (
                        _tgt_pop.get_attribute("x", _tgt_sel),
                        _tgt_pop.get_attribute("y", _tgt_sel),
                        _tgt_pop.get_attribute("z", _tgt_sel),
                    )
                )

            log.info(
                f"Generating afferent connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest})"
            )

        # Run connection wiring
        all_new_edges = self._connectome_wiring_wrapper(
            src_node_ids,
            src_pos,
            src_mtypes,
            src_class,
            morph_ext,
            tgt_node_ids,
            tgt_pos,
            tgt_mtypes,
            p_model,
            nsynconn_model,
            delay_model,
            edges_table_init=edges_table,
        )

        # Drop empty (NaN) columns [OTHERWISE: Problem converting to SONATA]
        init_prop_count = all_new_edges.shape[1]
        if all_new_edges.shape[0] > 0:
            all_new_edges.dropna(axis=1, inplace=True, how="all")  # Drop empty/unused columns
        unused_props = np.setdiff1d(edges_table.keys(), all_new_edges.keys())
        edges_table = edges_table.drop(
            unused_props, axis=1
        )  # Drop in original table as well, to avoid inconsistencies!
        final_prop_count = all_new_edges.shape[1]

        # Add new synapses to table, re-sort, and assign new index
        init_edge_count = edges_table.shape[0]
        edges_table = pd.concat([edges_table, all_new_edges])
        final_edge_count = edges_table.shape[0]
        if final_edge_count > init_edge_count:
            edges_table.sort_values(["@target_node", "@source_node"], inplace=True)
            edges_table.reset_index(
                inplace=True, drop=True
            )  # [No index offset required when merging files in block-based processing]

        log.info(
            f"Generated {final_edge_count - init_edge_count} (of {edges_table.shape[0]}) new synapses with {final_prop_count} properties ({init_prop_count - final_prop_count} removed)"
        )

        return edges_table

    def _init_edges_table(self, with_delay=True, from_table=None):
        """Initializes empty edges table."""
        if with_delay:
            required_properties = self.SYNAPSE_PROPERTIES
            property_types = self.PROPERTY_TYPES
        else:
            required_properties = list(filter(lambda x: x != "delay", self.SYNAPSE_PROPERTIES))
            property_types = {k: v for k, v in self.PROPERTY_TYPES.items() if k != "delay"}

        if from_table is None:  # Create empty table
            all_new_edges = pd.DataFrame([], columns=required_properties).astype(property_types)
        else:  # Init from existing table
            all_new_edges = pd.DataFrame(
                {
                    cname: pd.Series([], dtype=from_table[cname].dtype)
                    for cname in from_table.columns
                }
            )
            log.log_assert(
                np.all(np.isin(required_properties, all_new_edges.columns)),
                "Required synapse properties missing!",
            )
            if not np.all(
                [
                    property_types[k] == v
                    for k, v in all_new_edges[required_properties].dtypes.items()
                ]
            ):
                log.warning("Unexpected property data types!")

        return all_new_edges

    @profiler.profileit(name="conn_wiring/wiring")
    def _connectome_wiring_wrapper(
        self,
        src_node_ids,
        src_positions,
        src_mtypes,
        src_class,
        morph_ext,
        tgt_node_ids,
        tgt_positions,
        tgt_mtypes,
        p_model,
        nsynconn_model,
        delay_model,
        edges_table_init=None,
    ):
        """Stand-alone wrapper for connectome wiring."""
        # Create new edges table to collect all generated synapses
        all_new_edges = self._init_edges_table(
            with_delay=delay_model is not None, from_table=edges_table_init
        )
        new_edges_list = [all_new_edges]
        # Run connection wiring
        # progress_pct = np.maximum(0, np.round(100 * np.arange(len(tgt_node_ids)) / (len(tgt_node_ids) - 1)).astype(int))

        log_time = datetime.now()
        for tidx, tgt in enumerate(tgt_node_ids):
            new_time = datetime.now()
            if (new_time - log_time) / timedelta(minutes=1) > 1:
                log.info("Processing target node %d out of %d", tidx, len(tgt_node_ids))
                log_time = new_time
            #  if tidx == 0 or progress_pct[tidx - 1] != progress_pct[tidx]:
            #     print(f'{progress_pct[tidx]}%', end=' ' if progress_pct[tidx] < 100.0 else '\n') # Just for console, no logging

            # Determine conn. prob. of all source nodes to be connected with target node (mtype-specific)
            tgt_pos = tgt_positions[
                tidx : tidx + 1, :
            ]  # Get neuron positions (incl. position mapping, if provided)
            p_src = p_model.apply(
                src_pos=src_positions,
                tgt_pos=tgt_pos,
                src_type=src_mtypes,
                tgt_type=[tgt_mtypes[tidx]],
            ).flatten()
            p_src[np.isnan(p_src)] = 0.0  # Exclude invalid values
            p_src[
                src_node_ids == tgt
            ] = 0.0  # Exclude autapses [ASSUMING node IDs are unique across src/tgt node populations!]

            # Sample new presynaptic neurons from list of source nodes according to conn. prob.
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel]  # New source node IDs per connection
            num_new = len(src_new)
            if num_new == 0:
                continue  # Nothing to wire

            # Sample number of synapses per connection (mtype-specific)
            num_syn_per_conn = [
                nsynconn_model.apply(src_type=s, tgt_type=tgt_mtypes[tidx])
                for s in src_mtypes[src_new_sel]
            ]
            syn_conn_idx = np.concatenate(
                [[i] * n for i, n in enumerate(num_syn_per_conn)]
            )  # Create mapping from synapses to connections
            num_gen_syn = len(syn_conn_idx)  # Number of synapses to generate

            # Create new synapses
            new_edges = pd.DataFrame(
                {
                    cname: pd.Series([], dtype=all_new_edges[cname].dtype)
                    for cname in all_new_edges.columns
                }
            )
            new_edges["@source_node"] = src_new[
                syn_conn_idx
            ]  # Source node IDs per connection expanded to synapses
            new_edges["@target_node"] = tgt

            # Place synapses randomly on soma/dendrite sections
            # [TODO: Add model for synapse placement??]
            morph = self._get_tgt_morph(self.tgt_morph, morph_ext, tgt)

            sec_ind = np.hstack(
                [
                    [-1],
                    np.where(np.isin(morph.section_types, [nm.BASAL_DENDRITE, nm.APICAL_DENDRITE]))[
                        0
                    ],
                ]
            )  # Soma/dendrite section indices; soma...-1

            sec_sel = np.random.choice(
                sec_ind, len(syn_conn_idx)
            )  # Randomly choose section indices
            off_sel = np.random.rand(
                len(syn_conn_idx)
            )  # Randomly choose fractional offset within each section
            off_sel[sec_sel == -1] = 0.0  # Soma offsets must be zero
            type_sel = [
                int(morph.section(sec).type) if sec >= 0 else 0 for sec in sec_sel
            ]  # Type 0: Soma (1: Axon, 2: Basal, 3: Apical)
            pos_sel = np.array(
                [
                    nm.morphmath.path_fraction_point(morph.section(sec).points, off)
                    if sec >= 0
                    else morph.soma.center.astype(float)
                    for sec, off in zip(sec_sel, off_sel)
                ]
            )  # Synapse positions, computed from section & offset
            # syn_type = np.select([src_class[new_edges['@source_node']].to_numpy() == 'INH', src_class[new_edges['@source_node']].to_numpy() == 'EXC'], [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)]) # INH: 0-99 (Using 0); EXC: >=100 (Using 100)
            syn_type = np.select(
                [
                    src_class[src_new_sel][syn_conn_idx] == "INH",
                    src_class[src_new_sel][syn_conn_idx] == "EXC",
                ],
                [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)],
            )  # INH: 0-99 (Using 0); EXC: >=100 (Using 100)

            new_edges["afferent_section_id"] = (
                sec_sel + 1
            )  # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
            new_edges["afferent_section_pos"] = off_sel
            new_edges["afferent_section_type"] = type_sel
            new_edges[["afferent_center_x", "afferent_center_y", "afferent_center_z"]] = pos_sel
            new_edges["syn_type_id"] = syn_type

            # Assign distance-dependent delays (mtype-specific), based on (generative) delay model (optional)
            if delay_model is not None:
                src_new_pos = src_positions[src_new_sel, :]
                syn_dist = np.sqrt(
                    np.sum((pos_sel - src_new_pos[syn_conn_idx, :]) ** 2, 1)
                )  # Distance from source neurons (soma) to synapse positions on target neuron
                new_edges["delay"] = delay_model.apply(
                    distance=syn_dist,
                    src_type=src_mtypes[src_new_sel][syn_conn_idx],
                    tgt_type=tgt_mtypes[tidx],
                )

            # Add new_edges to edges table
            #         all_new_edges = all_new_edges.append(new_edges)
            new_edges_list.append(new_edges)

        all_new_edges = pd.concat(new_edges_list)

        return all_new_edges

    @classmethod
    def connectome_wiring_per_pathway(cls, nodes, pathway_models, seed=0, morph_ext="h5"):
        """Stand-alone connectome wiring per pathway, i.e., wiring pathways using pathway-specific probability/nsynconn/delay models."""
        # Init random seed for connectome building and sampling from parameter distributions
        np.random.seed(seed)

        conn_wiring = cls(nodes)

        # Loop over pathways
        new_edges_per_pathway = []
        for pathway_dict in tqdm.tqdm(pathway_models):
            # [OPTIMIZATION: Run wiring of pathways in parallel]

            pre_type = pathway_dict["pre"]
            post_type = pathway_dict["post"]
            prob_model = pathway_dict["prob_model"]
            nsynconn_model = pathway_dict["nsynconn_model"]
            delay_model = pathway_dict["delay_model"]

            # Select source/target nodes
            src_node_ids = nodes.ids({"mtype": pre_type})
            src_class = nodes.get(src_node_ids, properties="synapse_class").to_numpy()
            src_mtypes = get_enumeration(nodes, "mtype", src_node_ids)
            src_positions = nodes.positions(
                src_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            tgt_node_ids = nodes.ids({"mtype": post_type})
            tgt_mtypes = get_enumeration(nodes, "mtype", tgt_node_ids)
            tgt_positions = nodes.positions(
                tgt_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            # Create edges per pathway
            # pylint: disable=protected-access
            new_edges_per_pathway.append(
                conn_wiring._connectome_wiring_wrapper(
                    src_node_ids,
                    src_positions,
                    src_mtypes,
                    src_class,
                    morph_ext,
                    tgt_node_ids,
                    tgt_positions,
                    tgt_mtypes,
                    prob_model,
                    nsynconn_model,
                    delay_model,
                )
            )

            # ALTERNATIVE: Write to .parquet file and merge/convert to SONATA later
            # ... connectome_manipulation.edges_to_parquet(edges_table, output_file)
            # ... connectome_manipulation.parquet_to_sonata(input_file_list, output_file, nodes, nodes_files, keep_parquet=False)

        # Merge edges, re-sort, and assign new index
        edges_table = pd.concat(new_edges_per_pathway)
        edges_table.sort_values(["@target_node", "@source_node"], inplace=True)
        edges_table.reset_index(inplace=True, drop=True)

        return edges_table
