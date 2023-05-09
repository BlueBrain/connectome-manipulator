"""Main CLI entry point"""


import sys
from pathlib import Path
import logging

import click

from . import utils, log, profiler
from .connectome_manipulation import connectome_manipulation

logger = logging.getLogger(__name__)


@click.group(context_settings={"show_default": True})
@click.version_option()
@click.option("-v", "--verbose", count=True, default=0, help="-v for INFO, -vv for DEBUG")
def app(verbose):
    """Connectome manipulation tools."""
    level = min(verbose, 2)
    log.setup_logging(log_level=level)


@app.command()
@click.argument("config", required=True, type=Path)
@click.option("--output-dir", required=True, type=Path, help="Output directory.")
@click.option("--profile", required=False, is_flag=True, type=bool, help="Enable profiling.")
@click.option(
    "--resume",
    required=False,
    is_flag=True,
    type=bool,
    help="Resume from exisiting .parquet files.",
)
@click.option("--keep-parquet", required=False, is_flag=True, help="Keep temporary parquet files.")
@click.option(
    "--convert-to-sonata",
    required=False,
    is_flag=True,
    help="Convert parquet to sonata and generate circuit config",
)
@click.option(
    "--overwrite-edges", required=False, is_flag=True, help="Overwrite existing edges file"
)
@click.option(
    "--splits",
    required=False,
    default=0,
    type=int,
    help="Number of blocks, overwrites value in config file",
)
@click.option(
    "--parallel",
    required=False,
    is_flag=True,
    default=False,
    help="Run using SLURM job array",
)
@click.option(
    "--max-parallel-jobs",
    "-n",
    required=False,
    default=256,
    type=int,
    help="Maximum number of parallel jobs to run in array",
)
@click.option(
    "--sbatch-arg",
    "-s",
    required=False,
    multiple=True,
    type=str,
    help="Overwrite sbatch arguments with key=value",
)
def manipulate_connectome(
    config,
    output_dir,
    profile,
    resume,
    keep_parquet,
    convert_to_sonata,
    overwrite_edges,
    splits,
    parallel,
    max_parallel_jobs,
    sbatch_arg,
):
    """Manipulate or build a circuit's connectome."""
    # until we start using verbosity with the logging refactoring
    # pylint: disable=unused-argument
    # Initialize logger
    logging_path = output_dir / "logs"
    log_file = log.create_log_file(logging_path, "connectome_manipulation")

    if parallel:
        if utils.clear_slurm_env():
            logger.info("Prepared environment for parallel run from within SLURM job.")

    output_path = utils.create_dir(output_dir)
    profiler.ProfilerManager.set_enabled(profile)

    options = connectome_manipulation.Options(
        config_path=config,
        output_path=output_path,
        logging_path=logging_path,
        do_profiling=profile,
        do_resume=resume,
        keep_parquet=keep_parquet,
        convert_to_sonata=convert_to_sonata,
        overwrite_edges=overwrite_edges,
        splits=splits,
        parallel=parallel,
        max_parallel_jobs=max_parallel_jobs,
    )

    connectome_manipulation.main(options, log_file, slurm_args=sbatch_arg)
    profiler.ProfilerManager.show_stats()


if __name__ == "__main__":
    click.get_current_context().invoke(app, sys.argv[1:])
