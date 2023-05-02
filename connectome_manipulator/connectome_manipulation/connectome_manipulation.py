"""Main module for connectome manipulations:

- Loads a SONATA connectome using SNAP
- Applies manipulation(s) to the connectome, as specified by the manipulation config dict
- Writes back the manipulated connectome to a SONATA edges file, together with a new circuit config
"""
import copy
import glob
import os
import resource
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import libsonata
import numpy as np
import pandas as pd
from bluepysnap.circuit import Circuit

from .. import log, utils
from ..access_functions import get_edges_population, get_nodes_population
from .converters import create_parquet_metadata, edges_to_parquet, parquet_to_sonata
from .manipulation import Manipulation


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

    nodes = [src_nodes, tgt_nodes]

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


def resource_profiling(
    enabled=False, description="", reset=False, csv_file=None
):  # pragma: no cover
    """Resources profiling (memory consumption, execution time) and writing to log file.

    Optional: If csv_file is provided, data is also written to a .csv file for
              better machine readability.
    """
    if not enabled:
        return

    mem_curr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if not hasattr(resource_profiling, "mem_before") or reset:
        mem_diff = None
    else:
        mem_diff = mem_curr - resource_profiling.mem_before
    resource_profiling.mem_before = mem_curr

    if not hasattr(resource_profiling, "t_init") or reset:
        resource_profiling.t_init = time.time()
        resource_profiling.t_start = resource_profiling.t_init
        t_tot = None
        t_dur = None
    else:
        t_end = time.time()
        t_tot = t_end - resource_profiling.t_init
        t_dur = t_end - resource_profiling.t_start
        resource_profiling.t_start = t_end

    if len(description) > 0:
        description_log = " [" + description + "]"

    field_width = 36 + max(len(description_log) - 14, 0)

    log_msg = "\n"
    log_msg = log_msg + "*" * field_width + "\n"
    log_msg = (
        log_msg
        + "* "
        + "RESOURCE PROFILING{}".format(description_log).ljust(field_width - 4)
        + " *"
        + "\n"
    )
    log_msg = log_msg + "*" * field_width + "\n"

    log_msg = (
        log_msg
        + "* "
        + "Max. memory usage (GB):"
        + "{:.3f}".format(mem_curr).rjust(field_width - 27)
        + " *"
        + "\n"
    )

    if mem_diff is not None:
        log_msg = (
            log_msg
            + "* "
            + "Max. memory diff. (GB):"
            + "{:.3f}".format(mem_diff).rjust(field_width - 27)
            + " *"
            + "\n"
        )

    if t_tot is not None and t_dur is not None:
        log_msg = log_msg + "*" * field_width + "\n"

        if t_tot > 3600:
            t_tot_log = t_tot / 3600
            t_tot_unit = "h"
        else:
            t_tot_log = t_tot
            t_tot_unit = "s"
        log_msg = (
            log_msg
            + "* "
            + f"Total time ({t_tot_unit}):        "
            + "{:.3f}".format(t_tot_log).rjust(field_width - 27)
            + " *"
            + "\n"
        )

        if t_dur > 3600:
            t_dur_log = t_dur / 3600
            t_dur_unit = "h"
        else:
            t_dur_log = t_dur
            t_dur_unit = "s"
        log_msg = (
            log_msg
            + "* "
            + f"Elapsed time ({t_dur_unit}):      "
            + "{:.3f}".format(t_dur_log).rjust(field_width - 27)
            + " *"
            + "\n"
        )

    log_msg = log_msg + "*" * field_width + "\n"

    log.profiling(log_msg)

    # Add entry to performance table and write to .csv file (optional)
    if csv_file is not None:
        if not hasattr(resource_profiling, "perf_table") or reset:
            resource_profiling.perf_table = pd.DataFrame(
                [],
                columns=["label", "i_split", "N_split", "mem_curr", "mem_diff", "t_tot", "t_dur"],
            )
            resource_profiling.perf_table.index.name = "id"
            if not os.path.exists(os.path.split(csv_file)[0]):
                os.makedirs(os.path.split(csv_file)[0])

        label, *spec = description.split("-")
        i_split = np.nan
        N_split = np.nan

        if len(spec) == 1:
            spec = spec[0].split("/")
            if len(spec) == 2:
                i_split = spec[0]
                N_split = spec[1]

        resource_profiling.perf_table.loc[resource_profiling.perf_table.shape[0]] = [
            label,
            i_split,
            N_split,
            mem_curr,
            mem_diff,
            t_tot,
            t_dur,
        ]
        resource_profiling.perf_table.to_csv(csv_file)


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


def manip_wrapper(
    nodes,
    edges,
    N_split,
    i_split,
    id_selection,
    options,
    output_parquet_file,
    csv_file,
    in_job,
):
    """Wrapper function that can be optionally executed by a submitit Slurm job"""
    if in_job:
        import submitit

        job_env = submitit.JobEnvironment()
        log_file = log.logging_init(
            options.logging_path, name="connectome_manipulation." + str(job_env.job_id)
        )
        csv_file = os.path.splitext(log_file)[0] + ".csv"
    config = utils.load_json(options.config_path)
    np.random.seed(config.get("seed", 123456) * (i_split + 1))
    split_ids = libsonata.Selection(id_selection).flatten().astype(np.int64)
    # Apply connectome wiring
    edges_table = _get_afferent_edges_table(split_ids, edges)
    resource_profiling(options.do_profiling, f"loaded-{i_split + 1}/{N_split}", csv_file=csv_file)
    aux_dict = {
        "N_split": N_split,
        "i_split": i_split,
        "id_selection": id_selection,
        "split_ids": split_ids,
    }
    new_edges_table = _generate_partition_edges(
        nodes=nodes,
        split_ids=split_ids,
        edges_table=edges_table,
        manip_config=config,
        aux_dict=aux_dict,
    )
    # [TESTING/DEBUGGING]
    N_syn_in = len(edges_table) if edges_table is not None else 0
    N_syn_out = len(new_edges_table)
    resource_profiling(
        options.do_profiling,
        f"{'manipulated' if edges else 'wired'}-{i_split + 1}/{N_split}",
        csv_file=csv_file,
    )

    # Write back connectome to .parquet file
    edges_to_parquet(new_edges_table, output_parquet_file)
    resource_profiling(
        options.do_profiling,
        f"saved-{aux_dict['i_split'] + 1}/{aux_dict['N_split']}",
        csv_file=csv_file,
    )
    return N_syn_in, N_syn_out


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


class ChunkId:
    """An abstraction for a chunk of work, defined by its index and total"""

    def __init__(self, i_split, count):
        """Initialize a ChunkId"""
        self.count = count
        self.i_split = i_split


def main(options, log_file, slurm_args=()):
    """Build local connectome."""
    config = utils.load_json(options.config_path)

    sonata_config_file = config["circuit_config"]

    if "circuit_path" in config:
        circuit_path = Path(config["circuit_path"])
        sonata_config_file = circuit_path / sonata_config_file
    else:
        circuit_path = Path(os.path.split(config["circuit_config"])[0])
    log.log_assert(options.output_path != circuit_path, "Input directory == Output directory")

    csv_file = os.path.splitext(log_file)[0] + ".csv"

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

    if options.parallel:
        extra_params = [N_split, node_ids_split, log_file, done_file, parquet_file_list, slurm_args]
        syn_count_in, syn_count_out = _run_submitit(nodes, edges, options, *extra_params)
    else:
        syn_count_in = 0
        syn_count_out = 0

        for i_split, split_ids in enumerate(node_ids_split):
            chunk = ChunkId(i_split, N_split)
            output_parquet_file = parquet_file_list[i_split]
            extra_params = [split_ids, chunk, output_parquet_file, done_file, csv_file]
            n_in, n_out = _run_part(nodes, edges, options, *extra_params)
            syn_count_in += n_in
            syn_count_out += n_out

    log.info(
        f"Done processing.\nTotal input/output synapse counts: {syn_count_in}/{syn_count_out} (Diff: {syn_count_out-syn_count_in})\n"
    )

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

    resource_profiling(options.do_profiling, "final", csv_file=csv_file)


def _run_part(
    nodes, edges, options: Options, split_ids, chunk: ChunkId, output_parquet_file, done_f, csv_f
):
    # Resume option: Don't recompute, if .parquet file of current split already exists
    if options.do_resume and check_if_done(output_parquet_file, done_f):
        log.info(
            f"Split {chunk.i_split + 1}/{chunk.count}: Parquet file already exists - SKIPPING (do_resume={options.do_resume})"
        )
        return 0, 0

    log.info(
        f"Split {chunk.i_split + 1}/{chunk.count}: Wiring connectome targeting {len(split_ids)} neurons"
    )

    nsyn_in, nsyn_out = manip_wrapper(
        nodes,
        edges,
        chunk.count,
        chunk.i_split,
        libsonata.Selection(split_ids).ranges,
        options,
        output_parquet_file,
        csv_f,
        False,
    )
    mark_as_done(output_parquet_file, done_f)
    return nsyn_in, nsyn_out


def _run_submitit(
    nodes,
    edges,
    options: Options,
    N_split,
    node_ids_split,
    log_file,
    done_file,
    parquet_file_list,
    slurm_args,
):
    import submitit

    job_logs = str(options.logging_path) + "/%j"
    jobs = []

    # Initialize profiler
    csv_file = os.path.splitext(log_file)[0] + ".csv"
    resource_profiling(options.do_profiling, "initial", reset=True, csv_file=csv_file)

    executor = submitit.AutoExecutor(folder=job_logs)
    executor.update_parameters(
        slurm_array_parallelism=options.max_parallel_jobs,
        slurm_partition="prod",
        name="connectome_manipulator",
        timeout_min=120,
    )
    extra_args = {"slurm_" + k: v for k, v in map(lambda x: x.split("="), slurm_args)}
    executor.update_parameters(**extra_args)

    log.info(f"Setting up {N_split} processing batch jobs...")
    resource_profiling(options.do_profiling, "start_processing", csv_file=csv_file)

    # FIXME: There is a known corner-case when 1 split is chosen and parallelization is enabled.
    # This should be addressed.
    with executor.batch():
        # Start processing
        for i_split, split_ids in enumerate(node_ids_split):
            # Resume option: Don't recompute, if .parquet file of current split already exists
            output_parquet_file = parquet_file_list[i_split]
            if options.do_resume and check_if_done(output_parquet_file, done_file):
                log.info(
                    f"Split {i_split + 1}/{N_split}: Parquet file already exists - SKIPPING (do_resume={options.do_resume})"
                )
                continue

            log.info(
                f"Split {i_split + 1}/{N_split}: Wiring connectome targeting {len(split_ids)} neurons"
            )
            job = executor.submit(
                manip_wrapper,
                nodes,
                edges,
                N_split,
                i_split,
                libsonata.Selection(split_ids).ranges,
                options,
                output_parquet_file,
                csv_file,
                True,
            )
            jobs.append((job, output_parquet_file))

    # Wait for jobs to return

    N_syn_in = 0
    N_syn_out = 0

    for idx, (j, output_parquet_file) in enumerate(jobs):
        nsyn_in, nsyn_out = j.result()
        log.info(f"Job {idx} (SLURM job id: {j.job_id}) has completed")
        N_syn_in += nsyn_in
        N_syn_out += nsyn_out
        mark_as_done(output_parquet_file, done_file)

    resource_profiling(options.do_profiling, "end_processing", csv_file=csv_file)

    return N_syn_in, N_syn_out


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
