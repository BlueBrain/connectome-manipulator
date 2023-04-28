"""Main CLI entry point"""


import logging
import sys

import click

from connectome_manipulator import utils
from connectome_manipulator.connectome_manipulation import connectome_manipulation


log = logging.getLogger(__name__)


@click.group(context_settings={"show_default": True})
@click.version_option()
@click.option("-v", "--verbose", count=True, default=0, help="-v for INFO, -vv for DEBUG")
def app(verbose):
    """Connectome manipulation tools."""
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
@click.argument("config", required=True, type=str)
@click.option("--output-dir", required=True, type=str, help="Output directory.")
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
    default=None,
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
    _manipulate_connectome(
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
    )


def _manipulate_connectome(
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
    if parallel:
        if utils.clear_slurm_env():
            log.info("Prepared environment for parallel run from within SLURM job.")

    output_dir = utils.create_dir(output_dir)

    connectome_manipulation.main(
        config_path=config,
        output_dir=output_dir,
        do_profiling=profile,
        do_resume=resume,
        keep_parquet=keep_parquet,
        convert_to_sonata=convert_to_sonata,
        overwrite_edges=overwrite_edges,
        splits=splits,
        parallel=parallel,
        max_parallel_jobs=max_parallel_jobs,
        slurm_args=sbatch_arg,
    )


if __name__ == "__main__":
    app(sys.argv[1:])
