# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Useful converters for circuits data"""

import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import log, profiler, __version__


_WRITE_THRESHOLD = 128 * 1024**2

_SYNAPSE_PROPERTIES = [
    "source_node_id",
    "target_node_id",
    "afferent_section_id",
    "afferent_section_pos",
    "afferent_section_type",
    "afferent_center_x",
    "afferent_center_y",
    "afferent_center_z",
    "syn_type_id",
    "edge_type_id",
    "delay",
]
_PROPERTY_TYPES = {
    "source_node_id": "int64",
    "target_node_id": "int64",
    "afferent_section_id": "int64",
    "afferent_section_pos": "float32",
    "afferent_section_type": "int64",
    "afferent_center_x": "float32",
    "afferent_center_y": "float32",
    "afferent_center_z": "float32",
    "syn_type_id": "int64",
    "edge_type_id": "int64",
    "delay": "float32",
}
_ID_COLUMN_MAP = {"@target_node": "target_node_id", "@source_node": "source_node_id"}


class EdgeWriter:
    """Writes data to a Parquet file continuously.

    Pandas has facilities to write to Parquet, but the functionality is limited and does
    not allow to append to the Parquet file.

    This class will use the underlying PyArrow methods to open a Parquet file when needed,
    and repeatedly writing more data into the file as required.  Compared to storing a
    growing Pandas DataFrame and dumping it to disk at once, this will allow us to reduce
    our memory consumption.
    """

    def __init__(self, output_file: Path, existing_edges=None, with_delay: bool = True):
        """Initializes the writer

        If `existing_edges` is a Pandas DataFrame, it will be used to pre-populate the
        write buffer.  When appending to the writer, columns may be removed from the
        pre-existing data.

        A default schema is constructed, taking the `with_delay` parameter into account to
        include or exclude a delay column.  This default schema will be used for appending
        data to the writer if there is a column mismatch with any pre-existing data.

        If `output_file` is `None`, no writing of data to disk will happen.  Data will
        have to be retrieved with the `to_pandas` method.
        """
        self._batches = []
        self._batches_size = 0
        self._total_edges = 0
        self._path = output_file
        self._with_delay = with_delay
        self._drop_edge_type_column = False
        self._schema = None
        self._writer = None

        if existing_edges is None:
            self._schema = self._build_schema(with_delay)
        else:
            self.from_pandas(existing_edges)

    def __len__(self):
        """Returns the total number of rows written and cached"""
        return self._total_edges + sum(map(len, self._batches))

    def clear(self):
        """Reset the saved edges in the buffer"""
        self._batches = []

    def __enter__(self):
        """Entry point for context management"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit point for context management"""
        self.close()
        return False

    @staticmethod
    def _build_schema(with_delay):
        return pa.schema(
            [
                pa.field(n, pa.from_numpy_dtype(_PROPERTY_TYPES[n]))
                for n in _SYNAPSE_PROPERTIES
                if with_delay or n != "delay"
            ]
        )

    def from_pandas(self, df):
        """Replace any stored edges with the Pandas DataFrame given

        If a schema has already been set, it will be used to convert the given Pandas
        DataFrame to the internal representation. If no schema has been set, it will be
        derived from the DataFrame.
        """
        self._drop_edge_type_column = "edge_type_id" not in df.columns
        if self._drop_edge_type_column:
            df["edge_type_id"] = 0
        self._batches = [
            pa.record_batch(
                df.rename(columns=_ID_COLUMN_MAP).reset_index(drop=True),
                schema=self._schema,
            )
        ]
        if not self._schema:
            self._schema = self._batches[0].schema

    def append(self, **kwargs):
        """Append new data to the buffer.

        Requires that `kwargs` contains all required data from the default schema or
        pre-existing data. If the pre-existing data has less columns than the passed in
        data, additional columns will be removed from the buffered data.

        Once the internal buffer exceeds the write threshold, data
        will be appended to the associated parquet file.
        """
        missing = {f.name for f in self._schema} - set(kwargs.keys())
        if missing:
            log.warning(
                "Dropping columns '%s' and reverting to default schema", "', '".join(missing)
            )
            self._schema = self._build_schema(self._with_delay)
            columns = [f.name for f in self._schema]
            self._batches = [b.select(columns) for b in self._batches]

        data = [kwargs.pop(f.name) for f in self._schema]
        log.log_assert(not kwargs, "Additional data for edge writing")
        self._batches.append(pa.record_batch(data, schema=self._schema))
        self._batches_size += self._batches[-1].nbytes
        if self._path and self._batches_size > _WRITE_THRESHOLD:
            self._write_batches()

    def _write_batches(self):
        """Write buffers to parquet"""
        with profiler.profileit(name="write_to_parquet"):
            table = pa.Table.from_batches(self._batches)
            self._batches = []
            self._batches_size = 0
            self._total_edges += len(table)
            log.debug("Writing %d edges to disk, total of %d edges", len(table), self._total_edges)
            if len(table) > 0:  # Only write file if any edges exist
                if not self._writer:
                    log.log_assert(not self._path.exists(), "Can't append to open file")
                    log.debug("Opening %s to write", self._path)
                    self._writer = pq.ParquetWriter(self._path, table.schema)
                self._writer.write_table(table)

    def to_pandas(self):
        """Return the buffer as a Pandas DataFrame"""
        df = (
            pa.Table.from_batches(self._batches, schema=self._schema)
            .to_pandas()
            .rename(columns={"target_node_id": "@target_node", "source_node_id": "@source_node"})
        )
        if self._drop_edge_type_column:
            del df["edge_type_id"]
        return df

    def close(self):
        """Close any open Parquet files"""
        if self._path and self._batches:
            self._write_batches()
        if self._writer:
            self._writer.close()
            self._writer = None


def create_parquet_metadata(parquet_path, nodes):
    """Adding metadata file required for parquet conversion using parquet-converters/0.8.0

    [Modified from: https://bbpgitlab.epfl.ch/hpc/circuit-building/spykfunc/-/blob/main/src/spykfunc/functionalizer.py#L328-354]
    """
    parquet_file_list = glob.glob(os.path.join(parquet_path, "*.parquet"))
    if len(parquet_file_list) == 0:
        log.warning("No .parquet files exist, no metadata created!")
        return None  # No metadata if no .parquet files exist

    schema = pq.ParquetDataset(parquet_path, use_legacy_dataset=False).schema
    if schema.metadata:
        metadata = {k.decode(): v.decode() for k, v in schema.metadata.items()}
    else:
        metadata = {}
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
