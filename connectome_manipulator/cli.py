"""Main CLI entry point"""


import logging

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
    help="Resume from exisiting .parquet files.",
)
@click.option("--keep-parquet", required=False, is_flag=True, help="Keep temporary parquet files.")
@click.option(
    "--overwrite-edges", required=False, is_flag=True, help="Overwrite existing edges file"
)
def manipulate_connectome(config, output_dir, profile, resume, keep_parquet, overwrite_edges):
    """Manipulate or build a circuit's connectome."""
    _manipulate_connectome(config, output_dir, profile, resume, keep_parquet, overwrite_edges)


def _manipulate_connectome(config, output_dir, profile, resume, keep_parquet, overwrite_edges):
    config_dict = utils.load_json(config)

    output_dir = utils.create_dir(output_dir)

    connectome_manipulation.main(
        config=config_dict,
        output_dir=output_dir,
        do_profiling=profile,
        do_resume=resume,
        keep_parquet=keep_parquet,
        overwrite_edges=overwrite_edges,
    )
