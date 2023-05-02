"""Useful converters for circuits data"""
import glob
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq

from .. import log, __version__


def edges_to_parquet(edges_table, output_file):
    """Write edge properties table to parquet file (if non-empty!)."""
    if edges_table.size == 0:
        log.info(f"Edges table empty - SKIPPING {os.path.split(output_file)[-1]}")
    else:
        edges_table = edges_table.rename(
            columns={"@target_node": "target_node_id", "@source_node": "source_node_id"}
        )  # Convert column names
        edges_table["edge_type_id"] = 0  # Add type ID, required for SONATA
        edges_table.to_parquet(output_file, index=False)


def create_parquet_metadata(parquet_path, nodes):
    """Adding metadata file required for parquet conversion using parquet-converters/0.8.0

    [Modified from: https://bbpgitlab.epfl.ch/hpc/circuit-building/spykfunc/-/blob/main/src/spykfunc/functionalizer.py#L328-354]
    """
    schema: pq.ParquetSchema = pq.ParquetDataset(parquet_path, use_legacy_dataset=False).schema
    metadata = {k.decode(): v.decode() for k, v in schema.metadata.items()}
    metadata.update(
        {
            "source_population_name": nodes[0].name,
            "source_population_size": str(nodes[0].size),
            "target_population_name": nodes[1].name,
            "target_population_size": str(nodes[1].size),
            "version": __version__,
        }
    )  # [Will be reflected as "version" attribute under edges/<population_name> in the resulting SONATA file]
    new_schema = schema.with_metadata(metadata)
    metadata_file = os.path.join(parquet_path, "_metadata")
    pq.write_metadata(new_schema, metadata_file)

    return metadata_file


def parquet_to_sonata(
    input_path, output_file, nodes, done_file, population_name="default", keep_parquet=False
):
    """Convert all parquet file(s) from input path to SONATA format (using parquet-converters tool; recomputes indices!!).

    [IMPORTANT: .parquet to SONATA converter requires same column data types in all files!!
                Otherwise, value over-/underflows may occur due to wrong interpretation of numbers!!]
    [IMPORTANT: All .parquet files used for conversion must be non-empty!!
                Otherwise, value errors (zeros) may occur in resulting SONATA file!!]
    """
    # Check if .parquet files exist and are non-empty [Otherwise, value errors (zeros) may occur in resulting SONATA file]
    input_file_list = glob.glob(os.path.join(input_path, "*.parquet"))
    log.log_assert(len(input_file_list) > 0, "No .parquet files to convert!")
    log.log_assert(
        np.all([pq.read_metadata(f).num_rows > 0 for f in input_file_list]),
        "All .parquet files must be non-empty to be converted to SONATA!",
    )
    log.info(f"Converting {len(input_file_list)} (non-empty) .parquet file(s) to SONATA")

    # Creating metadata file [Required by parquet-converters/0.8.0]
    metadata_file = create_parquet_metadata(input_path, nodes)

    # Running parquet conversion [Requires parquet-converters/0.8.0]
    if os.path.exists(output_file):
        os.remove(output_file)
    with subprocess.Popen(
        ["parquet2hdf5", str(input_path), str(output_file), population_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"PATH": os.getenv("PATH", "")},
    ) as proc:
        log.info(proc.communicate()[0].decode())
    log.log_assert(
        os.path.exists(output_file),
        "Parquet conversion error - SONATA file not created successfully!",
    )

    # Delete temporary parquet files (optional)
    if not keep_parquet:
        log.info(
            f'Deleting {len(input_file_list)} temporary .parquet file(s), "{os.path.split(metadata_file)[-1]}" file, and "{os.path.split(done_file)[-1]}"'
        )
        for fn in input_file_list:
            os.remove(fn)
        os.remove(metadata_file)
        os.remove(done_file)
