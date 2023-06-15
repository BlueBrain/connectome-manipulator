"""Main module for connectome manipulations:

- Loads a SONATA connectome using SNAP
- Applies manipulation(s) to the connectome, as specified by the manipulation config dict
- Writes back the manipulated connectome to a SONATA edges file, together with a new circuit config
"""
import copy
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List
from pathlib import Path
import logging

import libsonata
import numpy as np
import pandas as pd
from bluepysnap.circuit import Circuit

from .. import log, utils, profiler
from ..access_functions import get_edges_population, get_nodes_population
from . import executors
from .tracker import JobTracker
from .converters import create_parquet_metadata, edges_to_parquet, parquet_to_sonata
from .manipulation import Manipulation

logger = logging.getLogger(__name__)


def load_circuit(sonata_config, N_split=1, popul_name=None):
    """Load given edges population of a SONATA circuit using SNAP."""
    # Load circuit
    log.info(f"Loading circuit from {sonata_config} (N_split={N_split})")
    c = Circuit(sonata_config)

    if (
        hasattr(c, "edges")
        and hasattr(c.edges, "population_names")
        and len(c.edges.population_names) > 0
    ):
        # Select edge population
        edges = get_edges_population(c, popul_name)
        edges_file = c.config["networks"]["edges"][0]["edges_file"]

        # Select corresponding source/target nodes populations
        src_nodes = edges.source
        tgt_nodes = edges.target
        if src_nodes is not tgt_nodes:
            log.log_assert(
                len(np.intersect1d(src_nodes.ids(), tgt_nodes.ids())) == 0,
                "Node IDs must be unique across different node populations!",
            )
    else:  # Circuit w/o edges
        edges = None
        edges_file = None
        popul_name = None

        src_nodes = tgt_nodes = get_nodes_population(c)

    nodes = (src_nodes, tgt_nodes)

    src_nodes_file = c.to_libsonata.node_population_properties(src_nodes.name).elements_path
    tgt_nodes_file = c.to_libsonata.node_population_properties(tgt_nodes.name).elements_path
    nodes_files = [src_nodes_file, tgt_nodes_file]

    if edges is None:
        log.debug(
            f'No edges population defined between nodes "{src_nodes.name}" and "{tgt_nodes.name}"'
        )
    else:
        log.debug(
            f'Using edges population "{edges.name}" between nodes "{src_nodes.name}" and "{tgt_nodes.name}"'
        )

    # Define target node splits
    tgt_node_ids = tgt_nodes.ids()
    node_ids_split = np.split(
        tgt_node_ids, np.cumsum([np.ceil(len(tgt_node_ids) / N_split).astype(int)] * (N_split - 1))
    )

    return c.config, nodes, nodes_files, node_ids_split, edges, edges_file, popul_name


def apply_manipulation(edges_table, nodes, split_ids, manip, aux_dict):
    """Apply manipulation to connectome (edges_table) as specified in the manip_config."""
    log.info(f'APPLYING MANIPULATION "{manip["name"]}"')

    for fun, cfg in enumerate(manip["fcts"]):
        source = cfg.pop("source")
        models = cfg.pop("model_config")

        log.info(f">>Function {fun + 1} of {len(manip['fcts'])}: source={source}")

        if filename := cfg.pop("model_pathways", None):
            filename = os.path.join(os.path.dirname(aux_dict["config_path"]), filename)
            pathways = pd.read_parquet(filename)
        else:
            pathways = None

        m = Manipulation.get(source)(nodes)

        if pathways is not None:
            grouped = pathways.groupby(
                ["src_hemisphere", "src_region", "dst_hemisphere", "dst_region"], observed=True
            )
            for n, ((src_hemi, src_region, dst_hemi, dst_region), group) in enumerate(grouped):
                log.info(f">>Step {n + 1} of {grouped.ngroups}: source={source}")

                # Reset index to match expectations
                group = group.set_index(["src_type", "dst_type"])

                edges_table = m.apply(
                    edges_table,
                    split_ids,
                    aux_dict,
                    sel_src={"hemisphere": src_hemi, "region": src_region},
                    sel_dest={"hemisphere": dst_hemi, "region": dst_region},
                    pathway_specs=group,
                    **cfg,
                    **models,
                )
        else:
            edges_table = m.apply(edges_table, split_ids, aux_dict, **cfg, **models)
    return edges_table


def create_new_file_from_template(new_file, template_file, replacements_dict, skip_comments=True):
    """Create new text file from template with replacements."""
    log.info(f"Creating file {new_file}")
    with open(template_file, "r") as file:
        content = file.read()

    content_lines = []
    for line in content.splitlines():
        if (
            skip_comments and len(line.strip()) > 0 and line.strip()[0] == "#"
        ):  # Skip replacement in commented lines
            content_lines.append(line)
        else:  # Apply replacements
            for src, dest in replacements_dict.items():
                line = line.replace(src, dest)
            content_lines.append(line)
    content = "\n".join(content_lines)

    with open(new_file, "w") as file:
        file.write(content)


def create_sonata_config(existing_config, out_config_path, out_edges_path, population_name):
    """Create new SONATA config (.JSON) from original, incl. modifications."""
    log.info(f"Creating SONATA config {out_config_path}")

    config = copy.deepcopy(existing_config)

    # Ensure there are no multiple edge populations from the original config
    edge_list = config["networks"].get("edges", [])
    if edge_list:
        log.log_assert(len(edge_list) == 1, "Multiple edge populations are not supported.")

    config["networks"]["edges"] = [
        {"edges_file": str(out_edges_path), "populations": {population_name: {"type": "chemical"}}}
    ]

    config = utils.reduce_config_paths(config, config_dir=Path(out_config_path).parent)

    utils.write_json(data=config, filepath=out_config_path)


def create_workflow_config(circuit_path, blue_config, manip_name, output_path, template_file):
    """Create bbp-workflow config for circuit registration (from template)."""
    if manip_name is None:
        manip_name = ""

    if len(manip_name) > 0:
        workflow_file = (
            os.path.split(os.path.splitext(template_file)[0])[1]
            + f"_{manip_name}"
            + os.path.splitext(template_file)[1]
        )
        circuit_name = "_".join(circuit_path.split("/")[-4:] + [manip_name])
        circuit_descr = f"{manip_name} applied to {circuit_path}"
        circuit_type = "Circuit manipulated by connectome_manipulator"
    else:
        workflow_file = os.path.split(template_file)[1]
        circuit_name = "_".join(circuit_path.split("/")[-4:])
        circuit_descr = f"No manipulation applied to {circuit_path}"
        circuit_type = "Circuit w/o manipulation"

    workflow_path = os.path.join(output_path, "workflows")
    if not os.path.exists(workflow_path):
        os.makedirs(workflow_path)

    config_replacements = {
        "$CIRCUIT_NAME": circuit_name,
        "$CIRCUIT_DESCRIPTION": circuit_descr,
        "$CIRCUIT_TYPE": circuit_type,
        "$CIRCUIT_CONFIG": blue_config,
        "$DATE": datetime.today().strftime("%Y-%m-%d %H:%M:%S") + " [generated from template]",
        "$FILE_NAME": workflow_file,
    }
    create_new_file_from_template(
        os.path.join(workflow_path, workflow_file),
        template_file,
        config_replacements,
        skip_comments=False,
    )


@dataclass
class Options:
    """CLI Options which have defaults"""

    output_path: Path
    config_path: Path
    logging_path: Path
    do_profiling: bool = False
    do_resume: bool = False
    keep_parquet: bool = False
    convert_to_sonata: bool = False
    overwrite_edges: bool = False
    splits: int = 0  # sentinel value to use config value
    parallel: bool = False


@dataclass
class JobsCommonInfo:
    """Information shared across all jobs"""

    nodes: libsonata.NodePopulation
    edges: libsonata.EdgePopulation


@dataclass
class JobInfo:
    """An abstraction for a chunk of work, defined by its index and total"""

    i_split: int
    N_split: int
    selection: Union[libsonata.Selection, List]
    out_parquet_file: str


def manip_wrapper(jobs_common: JobsCommonInfo, job: JobInfo, options: Options):
    """Wrapper function (remote) that can be optionally executed by a Dask worker"""
    if options.parallel:
        import socket

        path = os.path.join(options.logging_path, socket.getfqdn())
        log.create_log_file(path, f"connectome_manipulation.task-{job.i_split}")

    log.info("Processing job %s (common: %s)", job, jobs_common)
    profiler.ProfilerManager.set_enabled(options.do_profiling)
    config = utils.load_json(options.config_path)
    np.random.seed(config.get("seed", 123456) * (job.i_split + 1))
    split_ids = libsonata.Selection(job.selection).flatten().astype(np.int64)

    # Apply connectome wiring
    edges_table = _get_afferent_edges_table(split_ids, jobs_common.edges)
    aux_dict = {
        "N_split": job.N_split,
        "i_split": job.i_split,
        "id_selection": job.selection,
        "split_ids": split_ids,
        "config_path": options.config_path,
    }

    with profiler.profileit(name="processing"):
        new_edges_table = _generate_partition_edges(
            nodes=jobs_common.nodes,
            split_ids=split_ids,
            edges_table=edges_table,
            manip_config=config["manip"],
            aux_dict=aux_dict,
        )

    # [TESTING/DEBUGGING]
    N_syn_in = len(edges_table) if edges_table is not None else 0
    N_syn_out = len(new_edges_table)

    # Write back connectome to .parquet file
    with profiler.profileit(name="write_to_parquet"):
        edges_to_parquet(new_edges_table, job.out_parquet_file)

    return N_syn_in, N_syn_out, profiler.ProfilerManager


@profiler.profileit(name="connectome_manipulation_main")
def main(options, log_file, executor_args=()):
    """Build local connectome."""
    config = utils.load_json(options.config_path)
    sonata_config_file = config["circuit_config"]

    if "circuit_path" in config:
        circuit_path = Path(config["circuit_path"])
        sonata_config_file = circuit_path / sonata_config_file
    else:
        circuit_path = Path(os.path.split(config["circuit_config"])[0])
    log.log_assert(options.output_path != circuit_path, "Input directory == Output directory")

    # Initialize the profiler
    csv_file = log_file + ".csv"
    profiler.ProfilerManager.set_csv_file(csv_file=csv_file)

    if options.splits > 0:
        if "N_split_nodes" in config:
            log.debug(
                f"Overwriting N_split_nodes ({config['N_split_nodes']}) from configuration file with command line argument --split {options.splits}"
            )
        config["N_split_nodes"] = options.splits

    # Load circuit (nodes only)
    sonata_config_file = config["circuit_config"]
    if "circuit_path" in config:
        sonata_config_file = os.path.join(config["circuit_path"], sonata_config_file)

    N_split = max(config.get("N_split_nodes", 1), 1)
    log.info(f"Setting up {N_split} processing batch jobs...")

    sonata_config, nodes, _, node_ids_split, edges, _, _ = load_circuit(sonata_config_file, N_split)
    edges_file = options.output_path / "edges.h5"
    log.log_assert(
        options.overwrite_edges or not edges_file.exists(),
        f'Edges file "{edges_file}" already exists! Enable "overwrite_edges" flag to overwrite!',
    )

    # Follow SONATA convention for edge population naming
    edge_population_name = f"{nodes[0].name}__{nodes[1].name}__chemical"

    # Prepare params to the executor. Slurm executor will add the prefix itself
    executor_params = dict(x.split("=", 1) for x in executor_args)
    tracker = JobTracker(options.output_path, N_split)

    with tracker.follow_jobs() as result_hook:
        with executors.in_context(options, executor_params, result_hook=result_hook) as executor:
            log.info("Start job submission")
            jobs_common_info = JobsCommonInfo(nodes, edges)

            for i_split, parquet_file in tracker.prepare_parquet_dir(options.do_resume):
                split_ids = node_ids_split[i_split]
                sonata_selection = libsonata.Selection(split_ids)
                job_info = JobInfo(i_split, N_split, sonata_selection, parquet_file)
                _submit_part(executor, jobs_common_info, job_info, options)

    if options.convert_to_sonata:
        # [IMPORTANT: .parquet to SONATA converter requires same column data types in all files!! Otherwise, value over-/underflows may occur due to wrong interpretation of numbers!!]
        parquet_to_sonata(
            tracker.parquet_dir,
            edges_file,
            nodes,
            tracker.parquet_done_file,
            population_name=edge_population_name,
            keep_parquet=options.keep_parquet,
        )

        # Create new SONATA config (.JSON) from original config file
        out_config_path = Path(options.output_path, "circuit_config.json")
        create_sonata_config(
            existing_config=sonata_config,
            out_config_path=out_config_path,
            out_edges_path=edges_file,
            population_name=edge_population_name,
        )
    else:
        if not options.keep_parquet:
            log.warning(
                "--keep-parquet and --convert-to-sonata are set to false. I will keep the parquet files anyway"
            )
        create_parquet_metadata(tracker.parquet_dir, nodes)


def _submit_part(executor, jobs_common: JobsCommonInfo, job: JobInfo, options: Options):
    # Resume option: Don't recompute, if .parquet file of current split already exists
    split_id = job.i_split + 1
    n_neurons = job.selection.flat_size
    log.info(f"Split {split_id}/{job.N_split}: Wiring connectome targeting {n_neurons} neurons")

    job.selection = job.selection.ranges  # ! transform because selection is not serializable
    manip_params = (jobs_common, job, options)
    executor.submit(manip_wrapper, manip_params, {"out_parquet_file": job.out_parquet_file})


def _get_afferent_edges_table(node_ids, edges):
    if edges is None:
        return None
    return edges.afferent_edges(node_ids, properties=sorted(edges.property_names))


def _generate_partition_edges(nodes, split_ids, edges_table, manip_config, aux_dict):
    if edges_table is None:
        new_edges_table = apply_manipulation(edges_table, nodes, split_ids, manip_config, aux_dict)
    else:
        column_types = {col: edges_table[col].dtype for col in edges_table.columns}

        new_edges_table = apply_manipulation(edges_table, nodes, split_ids, manip_config, aux_dict)
        # Filter column type dict in case of removed columns (e.g., conn_wiring operation)
        column_types = {col: column_types[col] for col in new_edges_table.columns}
        new_edges_table = new_edges_table.astype(column_types)

    log.log_assert(
        new_edges_table["@target_node"].is_monotonic_increasing,
        "Target nodes not monotonically increasing!",
    )

    return new_edges_table


def _write_blue_config(manip_config, output_path, edges_fn_manip, edges_file_manip):
    # Create new symlinks and circuit config
    if not manip_config.get("blue_config_to_update") is None:
        blue_config = os.path.join(
            manip_config["circuit_path"], manip_config["blue_config_to_update"]
        )
        log.log_assert(
            os.path.exists(blue_config),
            f'Blue config "{manip_config["blue_config_to_update"]}" does not exist!',
        )
        with open(blue_config, "r") as file:  # Read blue config
            config = file.read()
        nrn_path = (
            list(
                filter(
                    lambda x: x.find("nrnPath") >= 0 and not x.strip()[0] == "#",
                    config.splitlines(),
                )
            )[0]
            .replace("nrnPath", "")
            .strip()
        )  # Extract path to edges file from BlueConfig
        circ_path_entry = list(
            filter(
                lambda x: x.find("CircuitPath") >= 0 and not x.strip()[0] == "#",
                config.splitlines(),
            )
        )[
            0
        ].strip()  # Extract circuit path entry from BlueConfig
        log.log_assert(
            os.path.abspath(nrn_path).find(os.path.abspath(manip_config["circuit_path"])) == 0,
            "nrnPath not within circuit path!",
        )
        nrn_path_manip = os.path.join(
            output_path, os.path.relpath(nrn_path, manip_config["circuit_path"])
        )  # Re-based path
        if not os.path.exists(os.path.split(nrn_path_manip)[0]):
            os.makedirs(os.path.split(nrn_path_manip)[0])

        # Symbolic link for edges.sonata
        symlink_src = os.path.relpath(edges_file_manip, os.path.split(nrn_path_manip)[0])
        symlink_dst = os.path.join(
            os.path.split(nrn_path_manip)[0], os.path.splitext(edges_fn_manip)[0] + ".sonata"
        )
        if os.path.isfile(symlink_dst) or os.path.islink(symlink_dst):
            os.remove(symlink_dst)  # Remove if already exists
        os.symlink(symlink_src, symlink_dst)
        log.info(f"Creating symbolic link ...{symlink_dst} -> {symlink_src}")

        # Create BlueConfig for manipulated circuit
        blue_config_manip = os.path.join(
            output_path,
            os.path.splitext(manip_config["blue_config_to_update"])[0]
            + f'_{manip_config["manip"]["name"]}'
            + os.path.splitext(manip_config["blue_config_to_update"])[1],
        )
        config_replacement = {
            nrn_path: symlink_dst,  # BETTER (???): nrn_path_manip
            circ_path_entry: f"CircuitPath {output_path}",
        }
        create_new_file_from_template(blue_config_manip, blue_config, config_replacement)

        # Symbolic link for start.target (if not existing)
        symlink_src = os.path.join(manip_config["circuit_path"], "start.target")
        symlink_dst = os.path.join(output_path, "start.target")
        if os.path.isfile(symlink_src) and not os.path.isfile(symlink_dst):
            os.symlink(symlink_src, symlink_dst)
            log.info(f"Creating symbolic link ...{symlink_dst} -> {symlink_src}")

        # Symbolic link for CellLibraryFile (if not existing)
        cell_lib_fn = (
            list(filter(lambda x: x.find("CellLibraryFile") >= 0, config.splitlines()))[0]
            .replace("CellLibraryFile", "")
            .strip()
        )  # Extract cell library file from BlueConfig
        if len(os.path.split(cell_lib_fn)[0]) == 0:  # Filename only, no path
            symlink_src = os.path.join(manip_config["circuit_path"], cell_lib_fn)
            symlink_dst = os.path.join(output_path, cell_lib_fn)
            if os.path.isfile(symlink_src) and not os.path.isfile(symlink_dst):
                os.symlink(symlink_src, symlink_dst)
                log.info(f"Creating symbolic link ...{symlink_dst} -> {symlink_src}")

        # Create bbp-workflow config from template to register manipulated circuit
        if manip_config.get("workflow_template") is not None:
            if os.path.exists(manip_config["workflow_template"]):
                create_workflow_config(
                    manip_config["circuit_path"],
                    blue_config_manip,
                    manip_config["manip"]["name"],
                    output_path,
                    manip_config["workflow_template"],
                )
            else:
                log.error(
                    f'Unable to create workflow config! Workflow template file "{manip_config["workflow_template"]}" not found!'
                )
