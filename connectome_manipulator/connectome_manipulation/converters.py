"""Useful converters for circuits data"""
import glob
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import log, __version__


class EdgeWriter:
    """Writes data to a Parquet file continuously.

    Pandas has facilities to write to Parquet, but the functionality is limited and does
    not allow to append to the Parquet file.

    This class will use the underlying PyArrow methods to open a Parquet file when needed,
    and repeatedly writing more data into the file as required.  Compared to storing a
    growing Pandas DataFrame and dumping it to disk at once, this will allow us to reduce
    our memory consumption.
    """

    def __init__(self, output_file: Path):
        """Initializes the object"""
        self._path = output_file
        self._writer = None
        self._total_edges = 0

    # Note: should accumulate edges_table until ~64 MiB are reached, then dump
    # it to disk to take full advantage of Parquet row group compression.
    def write(self, edges_table):
        """Write data to disk"""
        edges_table["edge_type_id"] = 0  # Add type ID, required for SONATA
        edges_table = edges_table.rename(
            columns={"@target_node": "target_node_id", "@source_node": "source_node_id"}
        )
        self._total_edges += len(edges_table)
        table = pa.Table.from_pandas(edges_table)
        if not self._writer:
            self._writer = pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)

        return None, self._total_edges

    def close(self):
        """Close any open Parquet files"""
        if self._writer:
            self._writer.close()


class EdgeNoop:
    """To be removed"""

    def write(self, edges_table):
        """To be removed"""
        return edges_table, len(edges_table)

    def close(self):
        """To be removed"""


@contextmanager
def process_edges(write: bool, output_file: Path):
    """To be removed"""
    if write:
        writer = EdgeWriter(output_file)
    else:
        writer = EdgeNoop()
    try:
        yield writer
    finally:
        writer.close()


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
        log.debug(proc.communicate()[0].decode())
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
