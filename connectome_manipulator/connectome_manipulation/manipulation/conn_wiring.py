# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Connectome wiring module (specialized)."""

from datetime import datetime, timedelta

import libsonata
import neurom as nm
import numpy as np
import tqdm

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import (
    get_attribute,
    get_node_ids,
    get_enumeration,
    get_node_positions,
)
from connectome_manipulator.connectome_manipulation.converters import EdgeWriter
from connectome_manipulator.connectome_manipulation.manipulation import (
    MorphologyCachingManipulation,
)
from connectome_manipulator.model_building import model_types, conn_prob

# IDEAs for improvements:
#   Add model for synapse placement


class ConnectomeWiring(MorphologyCachingManipulation):
    """Special case of connectome manipulation class for wiring a connectome:

    Special operation of connectome wiring, which generates an empty connectome
    from scratch, or simply adds connections to an existing connectome.
    The manipulation can be applied through the :func:`apply` method.

    IMPORTANT: This is a highly optimized operation for wiring huge connectomes by pathways
    (i.e., for each pair of pre-/post-synaptic hemisphere/region/m-type). However, only
    specific properties like source/target node, afferent synapse positions, synapse type
    (INH: 0, EXC: 100), and delay (optional) will be generated.
    For general purpose wiring, please use the
    :func:`connectome_manipulator.connectome_manipulation.manipulation.conn_rewiring.ConnectomeRewiring`
    operation!
    """

    # SONATA section type mapping (as in MorphIO): 1 = soma, 2 = axon, 3 = basal, 4 = apical
    SEC_SOMA = 1
    SEC_TYPE_MAP = {nm.AXON: 2, nm.BASAL_DENDRITE: 3, nm.APICAL_DENDRITE: 4}

    @profiler.profileit(name="conn_wiring")
    def apply(
        self,
        split_ids,
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
        """Applies a wiring (generation) of structural connections between pairs of neurons based on a given connectivity model.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
            sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
            pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
            amount_pct (float): Percentage of randomly sampled target (post-synaptic) neurons that will be wired
            morph_ext (str): Morphology file extension, e.g., "swc", "asc", "h5"
            prob_model_spec (dict): Connection probability model specification; a file can be specified by ``{"file": "path/file.json"}``
            nsynconn_model_spec (dict): Model specifications for #synapses/connection; a file can be specified by ``{"file": "path/file.json"}``
            delay_model_spec (dict): Delay model specification; a file can be specified by ``{"file": "path/file.json"}``
            pathway_specs (dict): Optional model specifications for efficiently setting model coefficients by pathway; will be automatically provided by the manipulator framework in case a .parquet file (containing a coefficient table for all pathways) is specified under "model_pathways" in the manipulation configuration file; only works with specific types of models
            **kwargs: Additional keyword arguments - Not used

        Note:
            Only structural synapse properties will be set: pre-/postsynaptic neuron IDs, synapse positions, type, axonal delays
        """
        # pylint: disable=arguments-differ
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

        with profiler.profileit(name="conn_wiring/setup"):
            # Intersect target nodes with split IDs and return if intersection is empty
            tgt_node_ids = get_node_ids(self.nodes[1], sel_dest, split_ids)
            num_tgt_total = len(tgt_node_ids)
            if num_tgt_total == 0:  # Nothing to wire
                log.info("No target nodes selected, nothing to wire")
                return
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
                return
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

            # Load source/taget position mappings (optional; two types of mappings supported)
            pos_mappings = conn_prob.get_pos_mapping_fcts(pos_map_file)

            # Determine source/target nodes for wiring
            src_node_ids = get_node_ids(self.nodes[0], sel_src)
            src_class = get_attribute(self.nodes[0], "synapse_class", src_node_ids)
            src_mtypes = get_enumeration(self.nodes[0], "mtype", src_node_ids)
            log.log_assert(len(src_node_ids) > 0, "No source nodes selected!")

            tgt_node_ids = tgt_node_ids[tgt_sel]  # Select subset of neurons (keeping order)
            tgt_mtypes = get_enumeration(self.nodes[1], "mtype", tgt_node_ids)

            # Get source/target node positions (optionally: two types of mappings)
            src_pos, tgt_pos = conn_prob.get_neuron_positions(
                self.nodes,
                [src_node_ids, tgt_node_ids],
                pos_acc=pos_mappings[0],
                vox_map=pos_mappings[1],
            )
            # ...and source positions w/o mapping (required for delays)
            if all(_map is None for _map in pos_mappings):
                raw_src_pos = src_pos
            else:
                raw_src_pos, _ = get_node_positions(self.nodes[0], src_node_ids)

            log.info(
                f"Generating afferent connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest})"
            )

        # Run connection wiring
        self._connectome_wiring_wrapper(
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
            raw_src_pos,
        )

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
        raw_src_positions,  # src positions w/o pos mapping (for delays!)
    ):
        """Stand-alone wrapper for connectome wiring."""
        # get morphologies for this selection
        tgt_morphs = self._get_tgt_morphs(morph_ext, libsonata.Selection(tgt_node_ids))

        log_time = datetime.now()
        for tidx, (tgt, morph) in enumerate(zip(tgt_node_ids, tgt_morphs)):
            new_time = datetime.now()
            if (new_time - log_time) / timedelta(minutes=1) > 1:
                log.info(
                    "Processing target node %d out of %d",
                    tidx,
                    len(tgt_node_ids),
                )
                log_time = new_time

            # Determine conn. prob. of all source nodes to be connected with target node
            tgt_pos = tgt_positions[tidx : tidx + 1, :]
            p_src = p_model.apply(
                src_pos=src_positions,
                tgt_pos=tgt_pos,
                src_type=src_mtypes,
                tgt_type=[tgt_mtypes[tidx]],
                src_nid=src_node_ids,
                tgt_nid=[tgt],
            ).flatten()
            p_src[np.isnan(p_src)] = 0.0  # Exclude invalid values
            # Exclude autapses [ASSUMING node IDs are unique across src/tgt
            # node populations!]
            p_src[src_node_ids == tgt] = 0.0

            # Sample new presynaptic neurons from list of source nodes according to conn. prob.
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
            src_new = src_node_ids[src_new_sel]  # New source node IDs per connection
            num_new = len(src_new)
            if num_new == 0:
                continue  # Nothing to wire

            # Sample number of synapses per connection (mtype-specific)
            num_syn_per_conn = nsynconn_model.apply(
                src_type=src_mtypes[src_new_sel], tgt_type=tgt_mtypes[tidx]
            )
            syn_conn_idx = np.concatenate(
                [[i] * n for i, n in enumerate(num_syn_per_conn)]
            )  # Create mapping from synapses to connections
            num_gen_syn = len(syn_conn_idx)  # Number of synapses to generate

            # Get available dendritic sections (plus soma) to place synapses on
            sec_ind = np.hstack(
                [
                    [-1],  # [Soma]
                    np.flatnonzero(
                        np.isin(morph.section_types, [nm.BASAL_DENDRITE, nm.APICAL_DENDRITE])
                    ),
                ]
            )

            # Randomly choose section indices
            sec_sel = np.random.choice(sec_ind, len(syn_conn_idx))

            # Randomly choose fractional offset within each section
            off_sel = np.random.rand(len(syn_conn_idx))
            off_sel[sec_sel == -1] = 0.0  # Soma offsets must be zero

            # Synapse positions & (mapped) section types, computed from section & offset
            type_sel = np.full_like(sec_sel, self.SEC_SOMA)
            pos_sel = np.tile(morph.soma.center.astype(float), (len(sec_sel), 1))
            for idx in np.flatnonzero(sec_sel >= 0):
                type_sel[idx] = self.SEC_TYPE_MAP[morph.section(sec_sel[idx]).type]
                pos_sel[idx] = nm.morphmath.path_fraction_point(
                    morph.section(sec_sel[idx]).points, off_sel[idx]
                )

            # Synapse type assignment [INH: 0-99 (Using 0); EXC: >=100 (Using 100)]
            syn_type = np.select(
                [
                    src_class[src_new_sel][syn_conn_idx] == "INH",
                    src_class[src_new_sel][syn_conn_idx] == "EXC",
                ],
                [np.full(num_gen_syn, 0), np.full(num_gen_syn, 100)],
            )

            # Assign distance-dependent delays (mtype-specific), based on (generative) delay model (optional)
            # IMPORTANT: Distances for delays are computed in them original coordinate system w/o coordinate transformation!
            kwargs = {}
            if delay_model is not None:
                src_new_pos = raw_src_positions[src_new_sel, :]
                syn_dist = np.sqrt(
                    np.sum((pos_sel - src_new_pos[syn_conn_idx, :]) ** 2, 1)
                )  # Distance from source neurons (soma) to synapse positions on target neuron
                delay = delay_model.apply(
                    distance=syn_dist,
                    src_type=src_mtypes[src_new_sel][syn_conn_idx],
                    tgt_type=tgt_mtypes[tidx],
                )
                if np.isscalar(delay):
                    kwargs["delay"] = np.full(syn_type.shape, delay)
                else:
                    kwargs["delay"] = delay

            # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
            #            The tuple representation (section_id, offset) is used here.
            self.writer.append(
                source_node_id=src_new[syn_conn_idx],
                target_node_id=np.full_like(syn_type, tgt),
                afferent_section_id=sec_sel + 1,
                afferent_section_pos=off_sel,
                afferent_section_type=type_sel,
                afferent_center_x=pos_sel[:, 0],
                afferent_center_y=pos_sel[:, 1],
                afferent_center_z=pos_sel[:, 2],
                syn_type_id=syn_type,
                edge_type_id=np.zeros_like(syn_type),
                **kwargs,
            )

    @classmethod
    def connectome_wiring_per_pathway(cls, nodes, pathway_models, seed=0, morph_ext="h5"):
        """Stand-alone connectome wiring per pathway, i.e., wiring pathways using pathway-specific probability/nsynconn/delay models."""
        # Init random seed for connectome building and sampling from parameter distributions
        np.random.seed(seed)

        with_delay = any(d["delay_model"] for d in pathway_models)

        writer = EdgeWriter(None, with_delay=with_delay)
        conn_wiring = cls(nodes, writer)
        src_nodes, tgt_nodes = nodes

        # Loop over pathways
        for pathway_dict in tqdm.tqdm(pathway_models):
            # [OPTIMIZATION: Run wiring of pathways in parallel]

            pre_type = pathway_dict["pre"]
            post_type = pathway_dict["post"]
            prob_model = pathway_dict["prob_model"]
            nsynconn_model = pathway_dict["nsynconn_model"]
            delay_model = pathway_dict["delay_model"]

            # Select source/target nodes
            src_node_ids = src_nodes.ids({"mtype": pre_type})
            src_class = get_attribute(src_nodes, "synapse_class", src_node_ids)
            src_mtypes = get_enumeration(src_nodes, "mtype", src_node_ids)
            src_positions = src_nodes.positions(
                src_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            tgt_node_ids = tgt_nodes.ids({"mtype": post_type})
            tgt_mtypes = get_enumeration(tgt_nodes, "mtype", tgt_node_ids)
            tgt_positions = tgt_nodes.positions(
                tgt_node_ids
            ).to_numpy()  # OPTIONAL: Coordinate system transformation may be added here

            # Create edges per pathway
            # pylint: disable=protected-access
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
                src_positions,
            )

            # ALTERNATIVE: Write to .parquet file and merge/convert to SONATA later
            # ... connectome_manipulation.edges_to_parquet(edges_table, output_file)
            # ... connectome_manipulation.parquet_to_sonata(input_file_list, output_file, nodes, nodes_files, keep_parquet=False)

        # Merge edges, re-sort, and assign new index
        edges_table = writer.to_pandas()
        edges_table.sort_values(["@target_node", "@source_node"], inplace=True)
        edges_table.reset_index(inplace=True, drop=True)

        return edges_table
