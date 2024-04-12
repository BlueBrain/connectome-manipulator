# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Module to facilitate easier job management"""

from contextlib import contextmanager
from pathlib import Path
from .. import log, utils, profiler


class JobTracker:
    """Class to track the execution of jobs"""

    def __init__(self, output_dir: Path, count: int):
        """Create a new tracker given an output directory and a job count"""
        self._jobs_done = 0
        self._jobs_total = count
        self._syn_count_in = 0
        self._syn_count_out = 0

        self.parquet_dir = output_dir / "parquet"
        self.parquet_done_file = self.parquet_dir / "parquet.DONE"

    def prepare_parquet_dir(self, resume: bool):
        """Setup and check the output parquet directory

        Will attempt to resume execution if `resume` is given, and will yield the job
        index and output file name for all jobs not yet process.
        """
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        # Potential list of .parquet file names [NOTE: Empty files won't actually exist!!]
        if self._jobs_total == 1:
            file_list = [self.parquet_dir / "edges.parquet"]
        else:
            ext_len = len(str(self._jobs_total))
            file_list = [
                self.parquet_dir / f"edges.{self._jobs_total}-{idx:0{ext_len}d}.parquet"
                for idx in range(self._jobs_total)
            ]

        # Check if parquet folder is clean
        existing_parquet_files = {p.stem for p in self.parquet_dir.glob("*.parquet")}
        done_list = set()
        if resume:
            # Resume from an existing run:
            # * Done file must be compatible with current run
            # * All existing .parquet files in the parquet folder must be from the list of expected (done) files (these files will be skipped over and merged later)
            if not self.parquet_done_file.exists():
                utils.write_json(data=[], filepath=self.parquet_done_file)
            else:
                # Load completed files from existing list [can be in arbitrary order!!]
                done_list = set(utils.load_json(self.parquet_done_file))
                unexpected = list(done_list - {p.stem for p in file_list})
                log.log_assert(
                    not unexpected,
                    f'Unable to resume! "{self.parquet_done_file}" contains unexpected entries: {unexpected}',
                )
            unexpected = list(existing_parquet_files - done_list)
            log.log_assert(
                not unexpected,
                f"Unable to resume! Parquet output directory contains unexpected .parquet files, please clean your output dir: {len(unexpected)} files",
            )
            # [NOTE: Empty files don't exist but may be marked as done!]
        else:  # Running from scratch: Parquet folder must not contain any .parquet files (so not to mix up existing and new files!!)
            log.log_assert(
                len(existing_parquet_files) == 0,
                'Parquet output directory contains .parquet files, please clean your output dir or use "do_resume" to resume from an existing run!',
            )
            utils.write_json(data=[], filepath=self.parquet_done_file)

        for i, file in enumerate(file_list):
            if file.stem in done_list:
                log.info(
                    f"Split {i + 1}/{self._jobs_total}: Parquet file already exists - "
                    f"SKIPPING (do_resume={resume})"
                )
            else:
                yield i, file

    def _mark_done(self, file):
        """Marks the given parquet file as "done" in the done file

        (i.e., adding the file name to the list of done files).
        """
        # Load list of completed files [can be in arbitrary order!!]
        done_list = utils.load_json(self.parquet_done_file)
        # Update list and write back
        done_list.append(file.stem)
        utils.write_json(data=done_list, filepath=self.parquet_done_file)

    @contextmanager
    def follow_jobs(self):
        """Start tracking jobs and provide a result hook to post-process them"""

        def _hook(result, info):
            self._syn_count_in += result[0]
            self._syn_count_out += result[1]
            self._mark_done(info["out_parquet_file"])
            profiler.ProfilerManager.merge(result[2])

            self._jobs_done += 1
            done_percent = self._jobs_done * 100 // self._jobs_total
            log.info(
                f"[{done_percent:3d}%] Finished {self._jobs_done} (out of {self._jobs_total}) splits"
            )

        yield _hook

        diff = self._syn_count_out - self._syn_count_in
        log.info("Done processing")
        log.info(
            "  Total input/output synapse counts: "
            f"{self._syn_count_in}/{self._syn_count_out} (Diff: {diff})"
        )
