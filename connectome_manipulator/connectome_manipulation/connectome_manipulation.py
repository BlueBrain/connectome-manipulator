"""Main module for connectome manipulations:

- Loads a SONATA connectome using SNAP
- Applies manipulation(s) to the connectome, as specified by the manipulation config dict
- Writes back the manipulated connectome to a SONATA edges file, together with a new circuit config
"""
import copy
import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List
from pathlib import Path
import logging

import libsonata
import numpy as np

from bluepysnap.circuit import Circuit

from .. import log, utils, profiler
from ..access_functions import get_edges_population, get_nodes_population
from . import executors
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

    src_file_idx = np.where(np.array(c.nodes.population_names) == src_nodes.name)[0]
    log.log_assert(len(src_file_idx) == 1, "Source nodes population file index error!")
    tgt_file_idx = np.where(np.array(c.nodes.population_names) == tgt_nodes.name)[0]
    log.log_assert(len(tgt_file_idx) == 1, "Target nodes population file index error!")

    src_nodes_file = c.config["networks"]["nodes"][src_file_idx[0]]["nodes_file"]
    tgt_nodes_file = c.config["networks"]["nodes"][tgt_file_idx[0]]["nodes_file"]
    nodes_files = [src_nodes_file, tgt_nodes_file]

    if edges is None:
        log.info(
            f'No edges population defined between nodes "{src_nodes.name}" and "{tgt_nodes.name}"'
        )
    else:
        log.info(
            f'Using edges population "{edges.name}" between nodes "{src_nodes.name}" and "{tgt_nodes.name}"'
        )

    # Define target node splits
    tgt_node_ids = tgt_nodes.ids()
    node_ids_split = np.split(
        tgt_node_ids, np.cumsum([np.ceil(len(tgt_node_ids) / N_split).astype(int)] * (N_split - 1))
    )

    return c.config, nodes, nodes_files, node_ids_split, edges, edges_file, popul_name


def apply_manipulation(edges_table, nodes, split_ids, manip_config, aux_dict):
    """Apply manipulation to connectome (edges_table) as specified in the manip_config."""
    log.info(f'APPLYING MANIPULATION "{manip_config["manip"]["name"]}"')
    n_fcts = len(manip_config["manip"]["fcts"])
    for m_step, m_config in enumerate(manip_config["manip"]["fcts"]):
        manip_source = m_config["source"]
        manip_kwargs = m_config["kwargs"]
        log.info(f">>Step {m_step + 1} of {n_fcts}: source={manip_source}")
        m = Manipulation.get(manip_source)(nodes)
        edges_table = m.apply(edges_table, split_ids, aux_dict, **manip_kwargs)
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


def prepare_parquet_dir(parquet_path, edges_fn, N_split, do_resume):
    """Setup and check of output parquet directory and preparation of .parquet file list.

    In addition, sets up a completed file list to resume from.
    """
    parquet_path = utils.create_dir(parquet_path)

    # File to keep track of parquet files already completed (to resume from with do_resume option)
    done_file = os.path.join(parquet_path, "parquet.DONE")

    # Potential list of .parquet file names [NOTE: Empty files won't actually exist!!]
    if N_split == 1:
        split_ext = [""]
    else:
        ext_len = len(str(N_split))
        split_ext = [f".{N_split}-{i_split:0{ext_len}d}" for i_split in range(N_split)]
    parquet_file_list = [
        os.path.join(parquet_path, os.path.splitext(edges_fn)[0]) + split_ext[i_split] + ".parquet"
        for i_split in range(N_split)
    ]

    # Check if parquet folder is clean
    existing_parquet_files = glob.glob(os.path.join(parquet_path, "*.parquet"))
    if (
        do_resume
    ):  # Resume from an existing run: Done file must be compatible with current run, and all existing .parquet files in the parquet folder must be from the list of expected (done) files (these files will be skipped over and merged later)
        if not os.path.exists(done_file):
            # Initialize empty list
            done_list = []
            utils.write_json(data=done_list, filepath=done_file)
        else:
            # Load completed files from existing list [can be in arbitrary order!!]
            done_list = utils.load_json(done_file)
            log.log_assert(
                all(
                    os.path.join(parquet_path, f) + ".parquet" in parquet_file_list
                    for f in done_list
                ),
                f'Unable to resume! "{os.path.split(done_file)[-1]}" contains unexpected entries!',
            )
        parquet_file_list_done = list(
            filter(
                lambda f: os.path.splitext(os.path.split(f)[-1])[0] in done_list, parquet_file_list
            )
        )
        log.log_assert(
            np.all([f in parquet_file_list_done for f in existing_parquet_files]),
            "Unable to resume! Parquet output directory contains unexpected .parquet files, please clean your output dir!",
        )
        # [NOTE: Empty files don't exist but may be marked as done!]
    else:  # Running from scratch: Parquet folder must not contain any .parquet files (so not to mix up existing and new files!!)
        log.log_assert(
            len(existing_parquet_files) == 0,
            'Parquet output directory contains .parquet files, please clean your output dir or use "do_resume" to resume from an existing run!',
        )
        utils.write_json(data=[], filepath=done_file)

    return parquet_file_list, done_file


def mark_as_done(parquet_file, done_file):
    """Marks the given parquet file as "done" in the done file

    (i.e., adding the file name to the list of done files).
    """
    log.log_assert(os.path.exists(done_file), f'"{done_file}" does not exist!')

    # Load list of completed files [can be in arbitrary order!!]
    done_list = utils.load_json(done_file)

    # Update list and write back
    done_list.append(os.path.splitext(os.path.split(parquet_file)[-1])[0])

    utils.write_json(data=done_list, filepath=done_file)


def check_if_done(parquet_file, done_file):
    """Checks if given parquet file is done already

    (i.e., file name existing in the list of done files).
    """
    log.log_assert(os.path.exists(done_file), f'"{done_file}" does not exist!')

    # Load list of completed files [can be in arbitrary order!!]
    done_list = utils.load_json(done_file)

    # Check if done
    is_done = os.path.splitext(os.path.split(parquet_file)[-1])[0] in done_list

    return is_done


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
    max_parallel_jobs: int = 256


@dataclass
class JobsCommonInfo:
    """Information shared across all jobs"""

    nodes: libsonata.NodePopulation
    edges: libsonata.EdgePopulation
    done_file: str


@dataclass
class JobInfo:
    """An abstraction for a chunk of work, defined by its index and total"""

    i_split: int
    N_split: int
    selection: Union[libsonata.Selection, List]
    out_parquet_file: str


def manip_wrapper(jobs_common: JobsCommonInfo, job: JobInfo, options: Options):
    """Wrapper function (remote) that can be optionally executed by a Slurm/Dask worker"""
    if options.parallel:
        log.create_log_file(options.logging_path, f"connectome_manipulation.task-{job.i_split}")

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
    }

    with profiler.profileit(name="processing"):
        new_edges_table = _generate_partition_edges(
            nodes=jobs_common.nodes,
            split_ids=split_ids,
            edges_table=edges_table,
            manip_config=config,
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
            log.warning(
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

    parquet_dir = options.output_path / "parquet"
    parquet_file_list, done_file = prepare_parquet_dir(
        parquet_dir, edges_file.name, N_split, options.do_resume
    )

    # Follow SONATA convention for edge population naming
    edge_population_name = f"{nodes[0].name}__{nodes[1].name}__chemical"

    # Prepare global result variables
    # Provide a result hook so that we automatically block for results
    syn_count_in = 0
    syn_count_out = 0
    jobs_done = 0

    def result_hook(result, info):
        nonlocal syn_count_in, syn_count_out, jobs_done
        syn_count_in += result[0]
        syn_count_out += result[1]
        mark_as_done(info["out_parquet_file"], done_file)
        resource_manager = result[2]
        profiler.ProfilerManager.merge(resource_manager)
        jobs_done += 1
        done_percent = jobs_done * 100 / N_split
        log.info(f"[{done_percent:3.0f}%] Finished {jobs_done} (out of {N_split}) splits")

    # Prepare params to the executor. Slurm executor will add the prefix itself
    executor_params = dict(x.split("=", 1) for x in executor_args)
    params = {"executor_params": executor_params}

    with executors.in_context(options, params, result_hook=result_hook) as executor:
        log.info("Start job submission")
        jobs_common_info = JobsCommonInfo(nodes, edges, done_file)

        for i_split, split_ids in enumerate(node_ids_split):
            sonata_selection = libsonata.Selection(split_ids)
            job_info = JobInfo(i_split, N_split, sonata_selection, parquet_file_list[i_split])
            _submit_part(executor, jobs_common_info, job_info, options)

    sdiff = syn_count_out - syn_count_in
    log.info("Done processing")
    log.info(f"  Total input/output synapse counts: {syn_count_in}/{syn_count_out} (Diff: {sdiff})")

    if options.convert_to_sonata:
        # [IMPORTANT: .parquet to SONATA converter requires same column data types in all files!! Otherwise, value over-/underflows may occur due to wrong interpretation of numbers!!]
        parquet_to_sonata(
            parquet_dir,
            edges_file,
            nodes,
            done_file,
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
        create_parquet_metadata(parquet_dir, nodes)


def _submit_part(executor, jobs_common: JobsCommonInfo, job: JobInfo, options: Options):
    # Resume option: Don't recompute, if .parquet file of current split already exists
    split_id = job.i_split + 1
    if options.do_resume and check_if_done(job.out_parquet_file, jobs_common.done_file):
        log.info(
            f"Split {split_id}/{job.N_split}: Parquet file already exists - SKIPPING (do_resume={options.do_resume})"
        )
        return

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
                log.warning(
                    f'Unable to create workflow config! Workflow template file "{manip_config["workflow_template"]}" not found!'
                )
