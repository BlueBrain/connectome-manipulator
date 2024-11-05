# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Connectome (re)wiring module (general-purpose)."""

import libsonata
import neurom as nm
import numpy as np
import pandas as pd

from connectome_manipulator import log, profiler
from connectome_manipulator.access_functions import (
    get_node_ids,
    get_enumeration,
    get_node_positions,
)
from connectome_manipulator.connectome_manipulation.manipulation import (
    MorphologyCachingManipulation,
)
from connectome_manipulator.model_building import model_types, conn_prob

OPT_NCONN_MAX_ITER = 1000


class ConnectomeRewiring(MorphologyCachingManipulation):
    """Connectome manipulation class for (re)wiring a connectome:

    Rewires an existing connectome, or wires an empty connectome fom scratch, based on
    a given model of connection probability. Different aspects of connectivity can be
    preserved during rewiring.
    The manipulation can be applied through the :func:`apply` method.
    """

    # SONATA section type mapping (as in MorphIO): 1 = soma, 2 = axon, 3 = basal, 4 = apical
    SEC_SOMA = 1
    SEC_TYPE_MAP = {nm.AXON: 2, nm.BASAL_DENDRITE: 3, nm.APICAL_DENDRITE: 4}

    def __init__(self, nodes, writer, split_index=0, split_total=1):
        """Construct ConnectomeRewiring Manipulation and declare state vars..."""
        self.duplicate_sample_synapses_per_mtype_dict = tuple({})
        self.props_sel = []
        self.props_afferent = []
        self.syn_sel_idx_type = None
        super().__init__(nodes, writer, split_index, split_total)

    @profiler.profileit(name="conn_rewiring")
    def apply(
        self,
        split_ids,
        syn_class,
        prob_model_spec,
        delay_model_spec,
        sel_src=None,
        sel_dest=None,
        pos_map_file=None,
        keep_indegree=True,
        reuse_conns=True,
        gen_method=None,
        amount_pct=100.0,
        props_model_spec=None,
        nsynconn_model_spec=None,
        estimation_run=False,
        p_scale=1.0,
        opt_nconn=False,
        pathway_specs=None,
        keep_conns=False,
        rewire_mode=None,
        syn_pos_mode="reuse",
        syn_pos_model_spec=None,
        morph_ext="swc",
    ):
        """Applies a (re)wiring of connections between pairs of neurons based on a given connectivity model.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework
            syn_class (str): Selection of synapse class ("EXC" or "INH"), i.e., outgoing connections from either excitatory or inhibitory neuron types will be rewired at a time
            prob_model_spec (dict): Connection probability model specification; a file can be specified by ``{"file": "path/file.json"}``
            delay_model_spec (dict): Delay model specification; a file can be specified by ``{"file": "path/file.json"}``
            sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
            sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
            pos_map_file (str/list-like): Optional position mapping file pointing to a position mapping model (.json) or voxel data map (.nrrd); one or two files for source/target node populations may be provided
            keep_indegree (bool): If selected, the in-degree (number of incoming connections) of each rewired post-synaptic neuron is preserved
            reuse_conns (bool): If selected, existing (incoming) connections may be reused to form new connections during rewiring; specifically, synapses per connection, synapse positions, as well as synapse physiology are preserved, and only new pre-synaptic neurons are assigned to such connections
            gen_method (str): Method used for generating new synapses; can be "sample" (samples physiological property values independently from existing synapses) or "randomize" (draws physiological property values from model distributions; requires ``props_model_spec``); no ``gen_method`` required in case both ``keep_indegree`` and ``reuse_conns`` are selected
            amount_pct (float): Percentage of randomly sampled target (post-synaptic) neurons that will be wired
            props_model_spec (dict): Physiological properties model specification; must be provided for ``gen_method`` "randomize"; a file can be specified by ``{"file": "path/file.json"}``
            nsynconn_model_spec (dict): Model specifications for #synapses/connection; not required if #synapses/connection are part of ``props_model_spec``, but will override if still provided; a file can be specified by ``{"file": "path/file.json"}``
            estimation_run (bool): If selected, runs rewiring with early stopping, i.e., w/o generating an output connectome; an estimate of the average number of incoming connections for each post-synaptic neuron will be written to a data log file
            p_scale (float): Optional global probability scaling factor
            opt_nconn (bool): If selected, the number of ingoing connections for each post-neuron will be optimized to match its expected number of connections on average. This is done by repeating the random generation process up to ``OPT_NCONN_MAX_ITER=1000`` times and keeping the instance which has the exact or closest match
            pathway_specs (dict): Optional model specifications for efficiently setting model coefficients by pathway; will be automatically provided by the manipulator framework in case a .parquet file (containing a coefficient table for all pathways) is specified under "model_pathways" in the manipulation configuration file; only works with specific types of models
            keep_conns (bool): If selected, an existing connection is kept exactly as it is in case the same connection should be established during rewiring; otherwise, such a connection would be established by reusing another existing one (if ``reuse_conns`` selected) or by generating new synapses forming that connection
            rewire_mode (str): Optional selection of specific rewiring modes, such as "add_only" (only new connections can be added, all existing ones will be kept) or "delete_only" (only existing connections can be deleted, no new ones will be added); otherwise, there are no restrictions on rewiring, i.e., new connections may be added and existing ones deleted
            syn_pos_mode (str): Selection of synapse position mode for generating new synapses, such as "reuse" (reuses all existing synapse positions on the post-synaptic dendrites), "reuse_strict" (reuses only synapse positions on the post-synaptic dendrites that are incoming from the selected source neurons), "random" (randomly places new synapses on the actual dendritic morphologies; slower since access to morphologies is required), or "external" (synapse positions provided externally through ``syn_pos_model_spec``)
            syn_pos_model_spec (dict): External synapse position model specification of type ``PropsTableModel``; only required if ``syn_pos_mode`` "external" is selected; a file can be specified by ``{"file": "path/file.json"}``
            morph_ext (str): Morphology file extension, e.g., "swc", "asc", "h5"; only used if ``syn_pos_mode`` "random" is selected

        Note:
            Input/output edges (synapse) tables are accessed through the ``writer`` object:

            * Loading input edges: ``edges_table = self.writer.to_pandas()``
            * Writing output edges: ``self.writer.from_pandas(edges_table_manip)``

            The input edges table is assumed to be sorted by ""@target_node", and the
            output edges table will again be sorted by "@target_node".
        """
        # pylint: disable=arguments-differ
        edges_table = self.writer.to_pandas()
        log.log_assert(
            np.all(np.diff(edges_table["@target_node"]) >= 0),
            "Edges table must be ordered by @target_node!",
        )
        log.log_assert(
            syn_class in ["EXC", "INH"],
            f'Synapse class "{syn_class}" not supported (must be "EXC" or "INH")!',
        )
        log.log_assert(0.0 <= amount_pct <= 100.0, '"amount_pct" out of range!')
        log.log_assert(p_scale >= 0.0, '"p_scale" cannot be negative!')

        if estimation_run:
            log.log_assert(
                keep_indegree is False,
                'Connectivity estimation not supported with "keep_indegree" option!',
            )
            log.debug("*** Estimation run enabled ***")

        if opt_nconn:
            log.log_assert(
                keep_indegree is False,
                '#Connections optimization not supported with "keep_indegree" option!',
            )
            log.log_assert(
                estimation_run is False,
                '#Connections optimization not supported with "estimation_run" option!',
            )
            log.debug(
                f"Enabled optimization of #connections to match expected number on average (max. {OPT_NCONN_MAX_ITER} iterations)"
            )

        if rewire_mode is not None:
            log.log_assert(
                rewire_mode in ["add_only", "delete_only"],
                f'Rewire mode "{rewire_mode}" not supported (must be "add_only", "delete_only", or None for full rewiring)!',
            )
            log.log_assert(
                keep_indegree is False,
                f'"keep_indegree" not supported for rewire mode "{rewire_mode}"!',
            )

        log.log_assert(
            syn_pos_mode in ["reuse", "reuse_strict", "random", "external"],
            f'Synapse position mode "{syn_pos_mode}" not supported (must be "reuse", "reuse_strict", "random", or "external")!',
        )

        if keep_indegree and reuse_conns:
            log.log_assert(
                gen_method is None,
                'No generation method required for "keep_indegree" and "reuse_conns" options!',
            )
            log.log_assert(
                syn_pos_mode in ["reuse", "reuse_strict"],
                '"reuse[_strict]" synapse position mode required when using "keep_indegree" and "reuse_conns" options!',
            )
        else:
            log.log_assert(
                gen_method in ["sample", "randomize"],
                'Valid generation method required (must be "sample" or "randomize")!',
            )

        if gen_method == "sample" and self.split_total > 1:
            log.warning(
                f'Generation method "{gen_method}" samples only from synapses within same data split! Reduce number of splits to 1 to sample from all synapses!'
            )

        if "file" not in prob_model_spec:
            prob_model_spec["src_type_map"] = self.src_type_map
            prob_model_spec["tgt_type_map"] = self.tgt_type_map
            prob_model_spec["pathway_specs"] = pathway_specs

        # Load connection probability model
        p_model = model_types.AbstractModel.init_model(prob_model_spec)
        log.debug(f'Loaded conn. prob. model of type "{p_model.__class__.__name__}"')
        if p_scale != 1.0:
            log.debug(f"Using probability scaling factor p_scale={p_scale}")

        # Load delay model
        delay_model = model_types.AbstractModel.init_model(delay_model_spec)
        log.debug(f'Loaded delay model of type "{delay_model.__class__.__name__}"')

        # Load source/taget position mappings (optional; two types of mappings supported)
        pos_mappings = conn_prob.get_pos_mapping_fcts(pos_map_file)

        # Load connection/synapse properties model [required for "randomize" generation method]
        if gen_method == "randomize":
            log.log_assert(
                props_model_spec is not None,
                f'Properties model required for generation method "{gen_method}"!',
            )
            props_model = model_types.AbstractModel.init_model(props_model_spec)
            log.debug(f'Loaded properties model of type "{props_model.__class__.__name__}"')
            if nsynconn_model_spec is None:
                nsynconn_model = None
                log.log_assert(
                    props_model.has_nsynconn,
                    "#Syn/conn model required when using a properties model w/o nsynconn!",
                )
            else:
                nsynconn_model = model_types.AbstractModel.init_model(nsynconn_model_spec)
                log.debug(f'Loaded #syn/conn model of type "{nsynconn_model.__class__.__name__}"')
                if props_model.has_nsynconn:
                    log.warning(
                        "Separate #syn/conn model provided! #Syn/conn given by properties model will be ignored!",
                    )
        else:
            log.log_assert(
                props_model_spec is None,
                f'Properties model incompatible with generation method "{gen_method}"!',
            )
            props_model = None
            log.log_assert(
                nsynconn_model_spec is None,
                f'#Syn/conn model incompatible with generation method "{gen_method}"!',
            )
            nsynconn_model = None

        # Load synapse position model [required for "external" position mode]
        if syn_pos_mode == "external":
            log.log_assert(
                syn_pos_model_spec is not None,
                f'Synapse position model required for position mode "{syn_pos_mode}"!',
            )
            syn_pos_model = model_types.AbstractModel.init_model(syn_pos_model_spec)
            log.debug(f'Loaded synapse position model of type "{syn_pos_model.__class__.__name__}"')
        else:
            log.log_assert(
                syn_pos_model_spec is None,
                f'Synapse position model incompatible with position mode "{syn_pos_mode}"!',
            )
            syn_pos_model = None

        # Initialize statistics dict
        stats_dict = {}
        # Number of synapses and connections removed/rewired/added/kept
        stats_dict["num_syn_removed"] = []
        stats_dict["num_conn_removed"] = []
        stats_dict["num_syn_rewired"] = []
        stats_dict["num_conn_rewired"] = []
        stats_dict["num_syn_added"] = []
        stats_dict["num_conn_added"] = []
        stats_dict["num_syn_kept"] = []
        stats_dict["num_conn_kept"] = []
        # Total input synapse count
        stats_dict["input_syn_count"] = edges_table.shape[0]
        # Number of tgt neurons unable to rewire
        stats_dict["unable_to_rewire_nrn_count"] = 0
        # Number of input connections within src/tgt node selection
        stats_dict["input_conn_count_sel"] = []
        # Number of output connections within src/tgt node selection (based on prob. model; for specific seed)
        stats_dict["output_conn_count_sel"] = []
        # Average number of output connections within src/tgt node selection (based on prob. model)
        stats_dict["output_conn_count_sel_avg"] = []

        # Determine source/target nodes for rewiring
        src_node_ids = get_node_ids(self.nodes[0], sel_src)
        src_class = self.nodes[0].get(src_node_ids, properties="synapse_class")
        src_node_ids = src_class[
            src_class == syn_class
        ].index.to_numpy()  # Select only source nodes with given synapse class (EXC/INH)
        log.log_assert(len(src_node_ids) > 0, f"No {syn_class} source nodes found!")
        stats_dict["source_nrn_count_all"] = len(
            src_node_ids
        )  # All source neurons (corresponding to chosen sel_src and syn_class)
        syn_sel_idx_src = np.isin(edges_table["@source_node"], src_node_ids)
        log.log_assert(
            (
                np.all(edges_table.loc[syn_sel_idx_src, "syn_type_id"] >= 100)
                if syn_class == "EXC"
                else np.all(edges_table.loc[syn_sel_idx_src, "syn_type_id"] < 100)
            ),
            "Synapse class error!",
        )

        # Only select target nodes that are actually in current split of edges_table
        tgt_node_ids = get_node_ids(self.nodes[1], sel_dest, split_ids)
        num_tgt_total = len(tgt_node_ids)
        stats_dict["target_nrn_count_all"] = (
            num_tgt_total  # All target neurons in current split (corresponding to chosen sel_dest)
        )
        num_tgt = np.round(amount_pct * num_tgt_total / 100).astype(int)
        stats_dict["target_nrn_count_sel"] = (
            num_tgt  # Selected target neurons in current split (based on amount_pct)
        )
        tgt_sel = np.random.permutation([True] * num_tgt + [False] * (num_tgt_total - num_tgt))
        if num_tgt_total > 0:
            tgt_node_ids = tgt_node_ids[tgt_sel]  # Select subset of neurons (keeping order)
        if num_tgt == 0:  # Nothing to rewire
            log.debug("No target nodes selected, nothing to rewire")
            if estimation_run:
                log.data(
                    f"EstimationStats_{self.split_index + 1}_{self.split_total}",
                    input_syn_count=stats_dict["input_syn_count"],
                    source_nrn_count_all=stats_dict["source_nrn_count_all"],
                    target_nrn_count_all=stats_dict["target_nrn_count_all"],
                    target_nrn_count_sel=stats_dict["target_nrn_count_sel"],
                    unable_to_rewire_nrn_count=stats_dict["unable_to_rewire_nrn_count"],
                    input_conn_count_sel=stats_dict["input_conn_count_sel"],
                    output_conn_count_sel_avg=stats_dict["output_conn_count_sel_avg"],
                )
                self.writer.from_pandas(edges_table.iloc[[]].copy())
                return
            else:
                log.data(
                    f"RewiringIndices_{self.split_index + 1}_{self.split_total}",
                    i_split=self.split_index,
                    N_split=self.split_total,
                    split_ids=split_ids,
                    tgt_node_ids=tgt_node_ids,
                    tgt_sel=tgt_sel,
                )
                self.writer.from_pandas(edges_table)
                return

        # Get source/target node positions (optionally: two types of mappings)
        src_pos, tgt_pos = conn_prob.get_neuron_positions(
            self.nodes,
            [src_node_ids, tgt_node_ids],
            pos_acc=pos_mappings[0],
            vox_map=pos_mappings[1],
        )

        # Load target morphologies, if needed
        if syn_pos_mode == "random":
            tgt_morphs = self._get_tgt_morphs(morph_ext, libsonata.Selection(tgt_node_ids))
        else:  # "reuse", "reuse_strict", or "external"
            tgt_morphs = [None] * num_tgt

        log.info(
            f"Rewiring afferent {syn_class} connections to {num_tgt} ({amount_pct}%) of {len(tgt_sel)} target neurons in current split (total={num_tgt_total}, sel_src={sel_src}, sel_dest={sel_dest}, keep_indegree={keep_indegree}, gen_method={gen_method}, keep_conns={keep_conns}, reuse_conns={reuse_conns}, syn_pos_mode={syn_pos_mode}{', morph_ext=' + morph_ext if syn_pos_mode == 'random' else ''}, rewire_mode={rewire_mode})"
        )

        # Init/reset static variables (function attributes) related to generation methods which need only be initialized once [for better performance]
        self._reinit(edges_table, syn_class)
        # Index of input connections (before rewiring) [for data logging]
        inp_conns, inp_syn_conn_idx, inp_syn_per_conn = np.unique(
            edges_table[["@target_node", "@source_node"]],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        inp_conns = np.fliplr(
            inp_conns
        )  # Restore ['@source_node', '@target_node'] order of elements
        stats_dict["input_conn_count"] = len(inp_syn_per_conn)
        stats_dict["input_syn_per_conn"] = list(inp_syn_per_conn)

        # Run connection rewiring
        syn_del_idx = np.full(
            edges_table.shape[0], False
        )  # Global synapse indices to keep track of all unused synapses to be deleted
        syn_rewire_idx = np.full(
            edges_table.shape[0], False
        )  # Global synapse indices to keep track of all rewired synapses [for data logging]
        new_edges_list = []  # New edges list to collect all generated synapses
        for tidx, (tgt, morph) in enumerate(zip(tgt_node_ids, tgt_morphs)):
            syn_sel_idx_tgt = edges_table["@target_node"] == tgt
            syn_sel_idx = np.logical_and(syn_sel_idx_tgt, syn_sel_idx_src)

            if syn_pos_mode == "reuse":
                syn_sel_idx_reuse = syn_sel_idx_tgt.copy()  # Reuse all synapses
            elif syn_pos_mode == "reuse_strict":
                syn_sel_idx_reuse = syn_sel_idx.copy()  # Reuse synapses restricted to sel_src
            else:
                syn_sel_idx_reuse = None

            if (keep_indegree and np.sum(syn_sel_idx) == 0) or (np.sum(syn_sel_idx_reuse) == 0):
                stats_dict["unable_to_rewire_nrn_count"] += 1  # (Neurons)
                # Nothing to rewire: either keeping indegree zero, or no target synapses exist
                # that could be rewired or positions reused from
                continue

            # Determine conn. prob. of all source nodes to be connected with target node
            p_src = (
                p_model.apply(
                    src_pos=src_pos,
                    tgt_pos=tgt_pos[tidx : tidx + 1, :],
                    src_type=get_enumeration(self.nodes[0], "mtype", src_node_ids),
                    tgt_type=get_enumeration(self.nodes[1], "mtype", [tgt]),
                    src_nid=src_node_ids,
                    tgt_nid=[tgt],
                ).flatten()
                * p_scale
            )
            p_src[np.isnan(p_src)] = 0.0  # Exclude invalid values
            p_src[src_node_ids == tgt] = (
                0.0  # Exclude autapses [ASSUMING node IDs are unique across src/tgt node populations!]
            )

            # Currently existing sources for given target node
            src, src_syn_idx = np.unique(
                edges_table.loc[syn_sel_idx, "@source_node"], return_inverse=True
            )
            num_src = len(src)
            stats_dict["input_conn_count_sel"] = stats_dict["input_conn_count_sel"] + [num_src]

            # Apply rewiring modes ("add_only", "delete_only", or full rewiring otherwise)
            conn_src = np.isin(src_node_ids, src)  # Existing source connectivity
            if rewire_mode == "add_only":  # Only new connections can be added, nothing deleted
                p_src = np.maximum(p_src, conn_src.astype(float))
            elif rewire_mode == "delete_only":  # Connections can only be deleted, nothing added
                p_src = np.minimum(p_src, conn_src.astype(float))

            # Sample new presynaptic neurons from list of source nodes according to conn. prob.
            if keep_indegree:  # Keep the same number of ingoing connections
                log.log_assert(
                    len(src_node_ids) >= num_src,
                    f"Not enough source neurons for target neuron {tgt} available for rewiring!",
                )
                log.log_assert(
                    np.sum(p_src) > 0.0,
                    "Keeping indegree not possible since connection probability zero!",
                )
                src_new = np.random.choice(
                    src_node_ids, size=num_src, replace=False, p=p_src / np.sum(p_src)
                )  # New source node IDs per connection
            else:  # Number of ingoing connections NOT necessarily kept the same
                stats_dict["output_conn_count_sel_avg"].append(np.round(np.sum(p_src)).astype(int))
                if estimation_run:
                    continue

                # Select source neurons (with or without optimizing numbers of connections)
                src_new_sel = self._select_sources(src_node_ids, p_src, opt_nconn)
                src_new = src_node_ids[src_new_sel]  # New source node IDs per connection
                stats_dict["output_conn_count_sel"].append(np.sum(src_new_sel))

            # Keep existing connections as they are (i.e., exclude them from any rewiring)
            if keep_conns:
                # Identify connections to keep
                keep_sel = np.logical_and(src_new_sel, conn_src)
                keep_ids = src_node_ids[keep_sel]  # Source node IDs to keep connections from

                # Remove from list of new connections
                src_new_sel[keep_sel] = False
                src_new = src_node_ids[src_new_sel]

                # Recompute source nodes selection used for rewiring
                keep_syn_idx = np.isin(edges_table["@source_node"], keep_ids)
                keep_syn_sel = np.logical_and(keep_syn_idx, syn_sel_idx)
                log.log_assert(
                    np.all(syn_sel_idx[keep_syn_sel]),
                    "ERROR: Inconsistent synapse indices to keep!",
                )
                syn_sel_idx[keep_syn_sel] = False

                src, src_syn_idx = np.unique(
                    edges_table.loc[syn_sel_idx, "@source_node"], return_inverse=True
                )
                num_src = len(src)

                # Synapse and connection statistics
                stats_dict["num_syn_kept"] = stats_dict["num_syn_kept"] + [np.sum(keep_syn_sel)]
                stats_dict["num_conn_kept"] = stats_dict["num_conn_kept"] + [len(keep_ids)]
            else:
                # Synapse and connection statistics
                stats_dict["num_syn_kept"] = stats_dict["num_syn_kept"] + [0]
                stats_dict["num_conn_kept"] = stats_dict["num_conn_kept"] + [0]

            # Randomize rewiring order of new source neurons
            src_new = np.random.permutation(src_new)
            num_new = len(src_new)

            # Re-use (up to) num_src existing connections (incl. #synapses/connection) for rewiring of (up to) num_new new connections (optional)
            if reuse_conns:
                num_src_to_reuse = num_src
                num_new_reused = num_new
            else:
                num_src_to_reuse = 0
                num_new_reused = 0

            if num_src > num_new_reused:  # Delete unused connections/synapses (randomly)
                src_syn_idx = self._shuffle_conns(src_syn_idx)
                syn_del_idx[syn_sel_idx] = (
                    src_syn_idx >= num_new_reused
                )  # Set global indices of connections to be deleted
                syn_sel_idx[syn_del_idx] = False  # Remove to-be-deleted indices from selection
                stats_dict["num_syn_removed"] = stats_dict["num_syn_removed"] + [
                    np.sum(src_syn_idx >= num_new_reused)
                ]
                stats_dict["num_conn_removed"] = stats_dict["num_conn_removed"] + [
                    num_src - num_new_reused
                ]
                src_syn_idx = src_syn_idx[src_syn_idx < num_new_reused]
            else:
                stats_dict["num_syn_removed"] = stats_dict["num_syn_removed"] + [0]
                stats_dict["num_conn_removed"] = stats_dict["num_conn_removed"] + [0]

            if num_src_to_reuse < num_new:  # Generate new synapses/connections, if needed
                num_gen_conn = num_new - num_src_to_reuse  # Number of new connections to generate
                src_gen = src_new[
                    -num_gen_conn:
                ]  # Split new sources into ones used for newly generated ...
                src_new = src_new[:num_src_to_reuse]  # ... and existing connections

                # Create new_edges and add them to global new edges table [ignoring duplicate indices]
                new_edges = self._generate_edges(
                    src_gen,
                    tidx,
                    tgt_node_ids,
                    syn_sel_idx_reuse,
                    edges_table,
                    gen_method,
                    props_model,
                    nsynconn_model,
                    delay_model,
                    morph,
                    syn_pos_model,
                )
                new_edges_list.append(new_edges)

                stats_dict["num_syn_added"] = stats_dict["num_syn_added"] + [new_edges.shape[0]]
                stats_dict["num_conn_added"] = stats_dict["num_conn_added"] + [len(src_gen)]
            else:
                stats_dict["num_syn_added"] = stats_dict["num_syn_added"] + [0]
                stats_dict["num_conn_added"] = stats_dict["num_conn_added"] + [0]

            # Assign new source nodes = rewiring of existing connections
            syn_rewire_idx = np.logical_or(syn_rewire_idx, syn_sel_idx)  # [for data logging]
            edges_table.loc[syn_sel_idx, "@source_node"] = src_new[
                src_syn_idx
            ]  # Source node IDs per connection expanded to synapses
            stats_dict["num_syn_rewired"] = stats_dict["num_syn_rewired"] + [len(src_syn_idx)]
            stats_dict["num_conn_rewired"] = stats_dict["num_conn_rewired"] + [len(src_new)]

            # Assign new distance-dependent delays (in-place), based on (generative) delay model
            self._assign_delays_from_model(
                delay_model, edges_table, src_new, src_syn_idx, syn_sel_idx
            )

        # Estimate resulting number of connections for computing a global probability scaling factor [returns empty edges table!!]
        if estimation_run:
            stat_sel = [
                "input_syn_count",
                "input_conn_count",
                "input_syn_per_conn",
                "source_nrn_count_all",
                "target_nrn_count_all",
                "target_nrn_count_sel",
                "unable_to_rewire_nrn_count",
                "input_conn_count_sel",
                "output_conn_count_sel_avg",
            ]
            stat_str = [
                (
                    f"      {k}: COUNT {len(v)}, MEAN {np.mean(v):.2f}, MIN {np.min(v)}, MAX {np.max(v)}, SUM {np.sum(v)}"
                    if isinstance(v, list) and len(v) > 0
                    else f"      {k}: {v}"
                )
                for k, v in stats_dict.items()
                if k in stat_sel
            ]
            log.debug("CONNECTIVITY ESTIMATION:\n%s", "\n".join(stat_str))
            log.data(
                f"EstimationStats_{self.split_index + 1}_{self.split_total}",
                **{k: v for k, v in stats_dict.items() if k in stat_sel},
            )
            self.writer.from_pandas(edges_table.iloc[[]].copy())
            return

        # Update statistics
        stats_dict["num_syn_unchanged"] = (
            stats_dict["input_syn_count"]
            - np.sum(stats_dict["num_syn_removed"])
            - np.sum(stats_dict["num_syn_rewired"])
        )

        # Delete unused synapses (if any)
        if np.any(syn_del_idx):
            edges_table = edges_table[~syn_del_idx].copy()
            log.debug(f"Deleted {np.sum(syn_del_idx)} unused synapses")

        # Add new synapses to table, re-sort, and assign new index
        if len(new_edges_list) > 0:
            all_new_edges = pd.concat(new_edges_list)
            syn_new_dupl_idx = np.array(
                all_new_edges.index
            )  # Index of duplicated synapses [for data logging]
            if edges_table.size == 0:
                max_idx = 0
            else:
                max_idx = np.max(edges_table.index)
            all_new_edges.index = range(
                max_idx + 1, max_idx + 1 + all_new_edges.shape[0]
            )  # Set index to new range, so as to keep track of new edges
            edges_table = pd.concat([edges_table, all_new_edges])
            edges_table.sort_values(
                "@target_node", kind="mergesort", inplace=True
            )  # Stable sorting, i.e., preserving order of input edges!!
            syn_new_idx = (
                edges_table.index > max_idx
            )  # Global synapse indices to keep track of all new synapses [for data logging]
            syn_new_dupl_idx = syn_new_dupl_idx[
                edges_table.index[syn_new_idx] - max_idx - 1
            ]  # Restore sorting, so that in same order as in merged & sorted edges table
            log.debug(f"Generated {all_new_edges.shape[0]} new synapses")
        else:  # No new synapses
            syn_new_dupl_idx = np.array([])
            syn_new_idx = np.full(edges_table.shape[0], False)

        # Reset index
        edges_table.reset_index(
            inplace=True, drop=True
        )  # Reset index [No index offset required when merging files in block-based processing]

        # [TESTING] #
        # Check if output indeed sorted
        log.log_assert(
            np.all(np.diff(edges_table["@target_node"]) >= 0),
            "ERROR: Output edges table not sorted by @target_node!",
        )
        # ######### #

        # Index of output connections (after rewiring) [for data logging]
        out_conns, out_syn_conn_idx, out_syn_per_conn = np.unique(
            edges_table[["@target_node", "@source_node"]],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        out_conns = np.fliplr(
            out_conns
        )  # Restore ['@source_node', '@target_node'] order of elements
        stats_dict["output_syn_count"] = edges_table.shape[0]
        stats_dict["output_conn_count"] = len(out_syn_per_conn)
        stats_dict["output_syn_per_conn"] = list(out_syn_per_conn)

        # Log statistics
        stat_str = [
            (
                f"      {k}: COUNT {len(v)}, MEAN {np.mean(v):.2f}, MIN {np.min(v)}, MAX {np.max(v)}, SUM {np.sum(v)}"
                if isinstance(v, list) and len(v) > 0
                else f"      {k}: {v}"
            )
            for k, v in stats_dict.items()
        ]
        log.debug("STATISTICS:\n%s", "\n".join(stat_str))
        log.log_assert(
            stats_dict["num_syn_unchanged"]
            == stats_dict["output_syn_count"]
            - np.sum(stats_dict["num_syn_added"])
            - np.sum(stats_dict["num_syn_rewired"]),
            "ERROR: Unchanged synapse count mismtach!",
        )  # Consistency check
        log.data(f"RewiringStats_{self.split_index + 1}_{self.split_total}", **stats_dict)

        # Write index data log [book-keeping for validation purposes]
        inp_syn_unch_idx = np.zeros_like(
            syn_del_idx
        )  # Global synapse indices to keep track of all unchanged synapses [for data logging]
        inp_syn_unch_idx = np.logical_and(~syn_del_idx, ~syn_rewire_idx)
        out_syn_rew_idx = np.zeros_like(
            syn_new_idx
        )  # Global output synapse indices to keep track of all rewired synapses [for data logging]
        out_syn_rew_idx[~syn_new_idx] = syn_rewire_idx[
            ~syn_del_idx
        ]  # [ASSUME: Input edges table order preserved]
        out_syn_unch_idx = np.zeros_like(
            syn_new_idx
        )  # Global output synapse indices to keep track of all unchanged synapses [for data logging]
        out_syn_unch_idx[~syn_new_idx] = inp_syn_unch_idx[
            ~syn_del_idx
        ]  # [ASSUME: Input edges table order preserved]
        log.log_assert(
            np.sum(stats_dict["num_syn_rewired"]) == np.sum(syn_rewire_idx),
            "ERROR: Rewired (input) synapse count mismtach!",
        )
        log.log_assert(
            np.sum(stats_dict["num_syn_rewired"]) == np.sum(out_syn_rew_idx),
            "ERROR: Rewired (output) synapse count mismtach!",
        )
        log.log_assert(
            stats_dict["num_syn_unchanged"] == np.sum(inp_syn_unch_idx),
            "ERROR: Unchanged (input) synapse count mismtach!",
        )
        log.log_assert(
            stats_dict["num_syn_unchanged"] == np.sum(out_syn_unch_idx),
            "ERROR: Unchanged (output) synapse count mismtach!",
        )

        log.data(
            f"RewiringIndices_{self.split_index + 1}_{self.split_total}",
            inp_syn_del_idx=syn_del_idx,
            inp_syn_rew_idx=syn_rewire_idx,
            inp_syn_unch_idx=inp_syn_unch_idx,
            out_syn_new_idx=syn_new_idx,
            syn_new_dupl_idx=syn_new_dupl_idx,
            out_syn_rew_idx=out_syn_rew_idx,
            out_syn_unch_idx=out_syn_unch_idx,
            inp_conns=inp_conns,
            inp_syn_conn_idx=inp_syn_conn_idx,
            inp_syn_per_conn=inp_syn_per_conn,
            out_conns=out_conns,
            out_syn_conn_idx=out_syn_conn_idx,
            out_syn_per_conn=out_syn_per_conn,
            i_split=self.split_index,
            N_split=self.split_total,
            split_ids=split_ids,
            src_node_ids=src_node_ids,
            tgt_node_ids=tgt_node_ids,
            tgt_sel=tgt_sel,
        )
        # inp_syn_del_idx ... Binary index vector of deleted synapses w.r.t. input edges table (of current block)
        # inp_syn_rew_idx ... Binary index vector of rewired synapses w.r.t. input edges table (of current block)
        # inp_syn_unch_idx ... Binary index vector of unchanged synapses w.r.t. input edges table (of current block)
        # out_syn_new_idx ... Binary index vector of new synapses w.r.t. output edges table (of current block)
        # syn_new_dupl_idx ... Index vector of duplicated synapses (positions) w.r.t. input edges table (globally, i.e., across all blocks), corresponding to new synapses in out_syn_new_idx
        # out_syn_rew_idx ... Binary index vector of rewired synapses w.r.t. output edges table (of current block)
        # out_syn_unch_idx ... Binary index vector of unchanged synapses w.r.t. output edges table (of current block)
        # inp_conns ... Input connections (of current block)
        # inp_syn_conn_idx ... Index vector of input connections w.r.t. inp_conns (of current block)
        # inp_syn_per_conn: Number of synapses per connection w.r.t. inp_conns (of current block)
        # out_conns ... Input connections (of current block)
        # out_syn_conn_idx ... Index vector of input connections w.r.t. out_conns (of current block)
        # out_syn_per_conn ... Number of synapses per connection w.r.t. out_conns (of current block)
        # i_split ... Index of current block
        # N_split ... Total number of splits (blocks)
        # split_ids ... Neuron ids of current block
        # src_node_ids ... Selected source neuron ids
        # tgt_node_ids ... Selected target neuron ids within current block
        # tgt_sel ... Binary (random) target neuron selection index within current block, according to given amount_pct

        # [TESTING] #
        # Overflow/value check
        if edges_table.size > 0:
            log.log_assert(np.all(edges_table.abs().max() < 1e9), "Value overflow in edges table")
            if "n_rrp_vesicles" in edges_table.columns:
                log.log_assert(
                    np.all(edges_table["n_rrp_vesicles"] >= 1),
                    "Value error in edges table (n_rrp_vesicles)!",
                )
        # ######### #

        self.writer.from_pandas(edges_table)

    def _select_sources(self, src_node_ids, p_src, opt_nconn):
        """Select source neurons with or without optimizing numbers of connections."""
        if opt_nconn:
            # Optimizing #connections: Repeat random generation up to OPT_NCONN_MAX_ITER times
            #                          and keep the one with #connestions closest to average

            # Number of connections on average (=target count)
            num_conns_avg = np.round(np.sum(p_src)).astype(int)

            # Iterate OPT_NCONN_MAX_ITER times to find optimum
            new_conn_count = -np.inf
            for _ in range(OPT_NCONN_MAX_ITER):
                src_new_sel_tmp = np.random.rand(len(src_node_ids)) < p_src
                if np.abs(np.sum(src_new_sel_tmp) - num_conns_avg) < np.abs(
                    new_conn_count - num_conns_avg
                ):
                    # Keep closest value among all tries
                    src_new_sel = src_new_sel_tmp
                    new_conn_count = np.sum(src_new_sel)
                if new_conn_count == num_conns_avg:
                    break  # Optimum found
        else:  # Just draw once (w/o optimization)
            src_new_sel = np.random.rand(len(src_node_ids)) < p_src
        return src_new_sel

    def _shuffle_conns(self, syn_conn_idx):
        """Shuffles assignment of synapses to connections.

        e.g. [0, 0, 1, 1, 1, 2] -> [2, 2, 0, 0, 0, 1]
        """
        conn_map = np.random.permutation(np.max(syn_conn_idx) + 1)
        return conn_map[syn_conn_idx]

    def _generate_edges(
        self,
        src_gen,
        tidx,
        tgt_node_ids,
        syn_sel_idx_reuse,
        edges_table,
        gen_method,
        props_model,
        nsynconn_model,
        delay_model,
        morph,
        syn_pos_model,
    ):
        """Generates a new set of edges (=synapses), based on the chosen generation options.

        The generation method, use of morphologies, and delay model must be specified.
        """
        tgt = tgt_node_ids[tidx]

        # Create new synapses with pyhsiological parameters
        if gen_method == "sample":
            # Sample (non-morphology-related) property values independently from existing synapses
            new_edges, syn_conn_idx = self._create_synapses_by_sampling(
                src_gen, tidx, tgt_node_ids, edges_table
            )
        elif gen_method == "randomize":
            # Randomize (non-morphology-related) property values based on pathway-specific model distributions
            new_edges, syn_conn_idx = self._create_synapses_by_randomization(
                src_gen, tgt, props_model, nsynconn_model, edges_table
            )
        else:
            log.log_assert(False, f"Generation method {gen_method} unknown!")

        # Assign synapses to connections from src_gen to tgt
        # pylint: disable=E0601, E0606
        new_edges["@source_node"] = src_gen[syn_conn_idx]
        new_edges["@target_node"] = tgt

        # Fill-in synapse positions (in-place)
        if morph is None and syn_pos_model is None:  # i.e., syn_pos_mode "reuse" or "reuse_strict"
            # Duplicate synapse positions on target neuron
            self._reuse_synapse_positions(
                new_edges, edges_table, syn_sel_idx_reuse, syn_conn_idx, tgt
            )
        elif syn_pos_model is None and morph is not None:  # i.e., syn_pos_mode "random"
            # Randomly generate new synapse positions on target neuron
            self._generate_synapse_positions(morph, new_edges, syn_conn_idx)
        elif syn_pos_model is not None and morph is None:  # i.e., syn_pos_mode "external"
            # Load synapse positions externally from model
            self._load_synapse_positions(syn_pos_model, new_edges, src_gen, tgt)
        else:
            log.log_assert(False, "Synapse position mode error!")

        # Assign distance-dependent delays (in-place), based on (generative) delay model
        self._assign_delays_from_model(delay_model, new_edges, src_gen, syn_conn_idx)

        # Restore original data types
        new_edges = new_edges.astype(edges_table.dtypes)
        # new_edges = new_edges.astype(edges_table.dtypes[new_edges.columns])  # [ALTERNATIVE: In case of column mismatch!]

        return new_edges

    def _reuse_synapse_positions(
        self, new_edges, edges_table, syn_sel_idx_reuse, syn_conn_idx, tgt
    ):
        """Assigns (in-place) duplicate synapse positions on target neuron (w/o accessing dendritic morphologies).

        If possible, synapses will be selected such that no duplicated synapses belong to same connection.
        """
        if len(self.props_afferent) == 0:
            # No afferent properties to duplicate (i.e., point neurons)
            return

        conns, nsyns = np.unique(syn_conn_idx, return_counts=True)
        draw_from = np.where(syn_sel_idx_reuse)[0]
        sel_dupl = []
        unique_per_conn_warning = False
        for dupl_count in nsyns:
            if len(draw_from) >= dupl_count:
                sel_dupl.append(
                    np.random.choice(draw_from, dupl_count, replace=False)
                )  # Random sampling from existing synapses WITHOUT replacement, if possible
            else:
                sel_dupl.append(
                    np.random.choice(draw_from, dupl_count, replace=True)
                )  # Random sampling from existing synapses WITH replacement, otherwise
                unique_per_conn_warning = True
        sel_dupl = np.hstack(sel_dupl)

        if unique_per_conn_warning:
            log.warning(
                f"Duplicated synapse position belonging to same connection (target neuron {tgt})! Unique synapse positions per connection not possible!"
            )

        # [TESTING] #
        # Check if indeed no duplicates per connection
        if not unique_per_conn_warning:
            for cidx in conns:
                log.log_assert(
                    np.all(np.unique(sel_dupl[syn_conn_idx == cidx], return_counts=True)[1] == 1),
                    f"ERROR: Duplicated synapse positions within connection (target neuron {tgt})!",
                )
        # ######### #

        # Duplicate and assign afferent position properties
        new_edges[self.props_afferent] = edges_table.iloc[sel_dupl][self.props_afferent].to_numpy()

    def _generate_synapse_positions(self, morph, new_edges, syn_conn_idx):
        """Assign (in-place) new synapse positions on target neuron (with accessing dendritic morphologies).

        Synapses are randomly (uniformly) placed on soma/dendrite sections.
        Only afferent_... properties following the tuple representation (section_id, offset) will be set.
        """
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

        # Assign afferent position properties
        # IMPORTANT: Section IDs in NeuroM morphology don't include soma, so they need to be shifted by 1 (Soma ID is 0 in edges table)
        new_edges["afferent_section_id"] = sec_sel + 1
        new_edges["afferent_section_pos"] = off_sel
        new_edges["afferent_section_type"] = type_sel
        new_edges["afferent_center_x"] = pos_sel[:, 0]
        new_edges["afferent_center_y"] = pos_sel[:, 1]
        new_edges["afferent_center_z"] = pos_sel[:, 2]

    def _load_synapse_positions(self, syn_pos_model, new_edges, src_gen, tgt):
        """Assign (in-place) new synapse positions on target neuron (w/o accessing dendritic morphologies).

        Synapse positions are directly loaded from position table provided as PropsTableModel. An error is raised
        if not enough positions are available. No consistency checks against actual morphologies are done.
        Only afferent_... properties following the tuple representation (section_id, offset) will be set.
        """
        prop_names = [f"afferent_section_{_p}" for _p in ["id", "pos", "type"]] + [
            f"afferent_center_{_p}" for _p in ["x", "y", "z"]
        ]

        # Load synapse positions from model (order is arbitrary!)
        syn_pos = syn_pos_model.apply(
            src_nid=src_gen, tgt_nid=tgt, prop_names=["@source_node"] + prop_names
        ).set_index("@source_node")

        # Assign positions to source nodes
        for sid in src_gen:  # List of source node IDs
            conn_sel = new_edges["@source_node"] == sid
            n_sel = np.sum(conn_sel)
            new_edges.loc[conn_sel, prop_names] = syn_pos.loc[[sid]].to_numpy()[:n_sel, :]

    def _reinit(self, edges_table, syn_class):
        # Dict to keep computed values per target m-type (instead of re-computing them for each target neuron)
        self.duplicate_sample_synapses_per_mtype_dict = {}

        # Non-morphology-related property selection (to be sampled/randomized)
        self.props_sel = list(
            filter(
                lambda x: not np.any(
                    [
                        excl in x
                        for excl in [
                            "_node",
                            "_x",
                            "_y",
                            "_z",
                            "_section",
                            "_segment",
                            "_length",
                            "_morphology",
                            "delay",
                        ]
                    ]
                ),
                edges_table.columns,
            )
        )

        # Afferent morphology-related synapse properties (for duplicating synapses)
        self.props_afferent = list(filter(lambda nm: "afferent_" in nm, edges_table.columns))
        if len(self.props_afferent) == 0:
            log.warning('No "afferent_..." synapse properties - point neurons assumed!')

        # Synapse class selection (EXC or INH)
        if syn_class == "EXC":  # EXC: >=100
            self.syn_sel_idx_type = edges_table["syn_type_id"] >= 100
        elif syn_class == "INH":  # INH: 0-99
            self.syn_sel_idx_type = edges_table["syn_type_id"] < 100
        else:
            log.log_assert(False, f"Synapse class {syn_class} not supported!")

    def _create_synapses_by_sampling(self, src_gen, tidx, tgt_node_ids, edges_table):
        """Creates new synapses with pyhsiological parameter values by sampling.

        Works by sampling (non-morphology-related) property values, including numbers of
        synapses per connection, independently from existing synapses.
        All other properties will be initialized as zero (to be filled in later).
        """
        # Sample #synapses/connection from other existing synapses targetting neurons of the same mtype (or layer) as tgt (incl. tgt)
        tgt = tgt_node_ids[tidx]
        tgt_layers = self.nodes[1].get(tgt_node_ids, properties="layer").to_numpy()
        tgt_mtypes = self.nodes[1].get(tgt_node_ids, properties="mtype").to_numpy()
        tgt_mtype = tgt_mtypes[tidx]
        num_gen_conn = len(src_gen)
        if (
            tgt_mtype in self.duplicate_sample_synapses_per_mtype_dict
        ):  # Load from dict, if already exists [optimized for speed]
            syn_sel_idx_mtype = self.duplicate_sample_synapses_per_mtype_dict[tgt_mtype][
                "syn_sel_idx_mtype"
            ]
            num_syn_per_conn = self.duplicate_sample_synapses_per_mtype_dict[tgt_mtype][
                "num_syn_per_conn"
            ]
        else:  # Otherwise compute
            syn_sel_idx_mtype = np.logical_and(
                self.syn_sel_idx_type,
                np.isin(edges_table["@target_node"], tgt_node_ids[tgt_mtypes == tgt_mtype]),
            )
            if np.sum(syn_sel_idx_mtype) == 0:  # Ignore m-type, consider matching layer
                syn_sel_idx_mtype = np.logical_and(
                    self.syn_sel_idx_type,
                    np.isin(
                        edges_table["@target_node"], tgt_node_ids[tgt_layers == tgt_layers[tidx]]
                    ),
                )
            if np.sum(syn_sel_idx_mtype) == 0:  # Otherwise, ignore m-type & layer
                syn_sel_idx_mtype = self.syn_sel_idx_type
                log.warning(
                    f"No synapses with matching m-type or layer to sample connection property values for target neuron {tgt} from!"
                )
            log.log_assert(
                np.sum(syn_sel_idx_mtype) > 0,
                f"No synapses to sample connection property values for target neuron {tgt} from!",
            )
            _, num_syn_per_conn = np.unique(
                edges_table[syn_sel_idx_mtype][["@source_node", "@target_node"]],
                axis=0,
                return_counts=True,
            )
            self.duplicate_sample_synapses_per_mtype_dict[tgt_mtype] = {
                "syn_sel_idx_mtype": syn_sel_idx_mtype,
                "num_syn_per_conn": num_syn_per_conn,
            }
        num_syn_per_conn = num_syn_per_conn[
            np.random.choice(len(num_syn_per_conn), num_gen_conn)
        ]  # Sample #synapses/connection
        syn_conn_idx = np.concatenate(
            [[i] * n for i, n in enumerate(num_syn_per_conn)]
        )  # Create mapping from synapses to connections

        # Initialize new edges table with zeros (preserving data types)
        new_edges = pd.DataFrame(
            np.zeros((len(syn_conn_idx), len(edges_table.columns))), columns=edges_table.columns
        ).astype(edges_table.dtypes)

        # Sample (non-morphology-related) property values independently from other existing synapses targetting neurons of the same mtype as tgt (incl. tgt)
        # => Assume identical (non-morphology-related) property values for synapses belonging to same connection
        for p in self.props_sel:
            new_edges[p] = (
                edges_table.loc[syn_sel_idx_mtype, p]
                .sample(num_gen_conn, replace=True)
                .to_numpy()[syn_conn_idx]
            )

        return new_edges, syn_conn_idx

    def _create_synapses_by_randomization(
        self, src_gen, tgt, props_model, nsynconn_model, edges_table
    ):
        """Creates new synapses with pyhsiological parameter values by randomization.

        Works by randomly drawing (non-morphology-related) property values, including
        numbers of synapses per connection, from pathway-specific model distributions.
        All other properties will be initialized as zero (to be filled in later).
        """
        log.log_assert(
            np.all(np.isin(self.props_sel, props_model.get_prop_names())),
            f"Required properties missing in properties model (must include: {self.props_sel})!",
        )
        # Generate new synapse properties based on properties model
        src_mtypes = self.nodes[0].get(src_gen, properties="mtype").to_numpy()
        tgt_mtype = self.nodes[1].get(tgt, properties="mtype")
        if nsynconn_model is None:  # #Syn/conn part of props_model
            new_syn_props = [props_model.apply(src_type=s, tgt_type=tgt_mtype) for s in src_mtypes]
        else:  # Draw #syn/conn from nsynconn_model
            nsynconn = nsynconn_model.apply(src_nid=src_gen, tgt_nid=[tgt]).flatten()
            new_syn_props = [
                props_model.apply(src_type=s, tgt_type=tgt_mtype, n_syn=n)
                for s, n in zip(src_mtypes, nsynconn)
            ]
        num_syn_per_conn = [syn.shape[0] for syn in new_syn_props]
        syn_conn_idx = np.concatenate(
            [[i] * n for i, n in enumerate(num_syn_per_conn)]
        )  # Create mapping from synapses to connections

        # Initialize new edges table with zeros (preserving data types)
        new_edges = pd.DataFrame(
            np.zeros((len(syn_conn_idx), len(edges_table.columns))), columns=edges_table.columns
        ).astype(edges_table.dtypes)

        # Assign non-morphology-related property values
        new_edges[self.props_sel] = pd.concat(new_syn_props, ignore_index=True)[
            self.props_sel
        ].to_numpy()

        return new_edges, syn_conn_idx

    def _assign_delays_from_model(
        self, delay_model, edges_table, src_new, src_syn_idx, syn_sel_idx=None
    ):
        """Assign new distance-dependent delays, drawn from truncated normal distribution, to new synapses within edges_table (in-place)."""
        log.log_assert(delay_model is not None, "Delay model required!")

        if syn_sel_idx is None:
            syn_sel_idx = np.full(edges_table.shape[0], True)

        if len(src_new) == 0 or len(src_syn_idx) == 0 or np.sum(syn_sel_idx) == 0:
            # No synapses specified
            return

        # Determine distance from source neuron (soma) to synapse on target neuron
        # IMPORTANT: Distances for delays are computed in them original coordinate system w/o coordinate transformation!
        src_new_pos, _ = get_node_positions(self.nodes[0], src_new)
        # Synapse position on post-synaptic dendrite
        syn_pos = edges_table.loc[
            syn_sel_idx, ["afferent_center_x", "afferent_center_y", "afferent_center_z"]
        ].to_numpy()
        syn_dist = np.sqrt(np.sum((syn_pos - src_new_pos[src_syn_idx, :]) ** 2, 1))

        # Obtain delay values from (generative) model
        delay_new = delay_model.apply(distance=syn_dist)

        # Assign to edges_table (in-place)
        edges_table.loc[syn_sel_idx, "delay"] = delay_new
