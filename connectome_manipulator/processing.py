# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Auxiliary classes to process many nodes split into chunks"""

from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import Union

import numpy as np
import pandas as pd

from . import log
from .access_functions import get_enumeration, get_enumeration_list


def _get_region_map(pop):
    """Get a mapping from region to node ids.

    Takes a bluepysnap population, will return a dataframe with hemisphere and region as
    index, node ids as column object.
    """
    hemis = get_enumeration(pop, "hemisphere")
    hemi_list = get_enumeration_list(pop, "hemisphere")
    regions = get_enumeration(pop, "region")
    region_list = get_enumeration_list(pop, "region")

    df = (
        pd.DataFrame(
            {
                "hemisphere": hemis,
                "region": regions,
                "id": np.arange(len(regions)),
            }
        )
        .groupby(["hemisphere", "region"])["id"]
        .apply(lambda x: x.values)
    )
    df.index = df.index.map(lambda x: (hemi_list[x[0]], region_list[x[1]]))
    return df


def _get_simple_node_splits(config, options, nodes):
    """Returns a simple, even split of target node ids"""
    if options.splits > 0:
        if "N_split_nodes" in config:
            log.debug(
                f"Overwriting N_split_nodes ({config['N_split_nodes']}) from configuration file with command line argument --split {options.splits}"
            )
        config["N_split_nodes"] = options.splits

    N_split = max(config.get("N_split_nodes", 1), 1)
    log.info(f"Setting up {N_split} processing batch jobs...")
    tgt_node_ids = nodes[1].ids()
    node_ids_split = np.split(
        tgt_node_ids, np.cumsum([np.ceil(len(tgt_node_ids) / N_split).astype(int)] * (N_split - 1))
    )
    return node_ids_split


def get_node_splits(config, options, nodes):
    """Split the target nodes into a series of lists of `BatchInfo` objects.

    If more than one manipulation is present in the passed configuration, or if no
    pathway-specific configuration is found, the resulting splitting is obtained by simply
    dividing the target nodes as evenly as possible.

    Otherwise, the pathway configuration is analyzed, and the target nodes are going to be
    grouped by region and hemisphere.  If a group is too large given the targeted workload
    defined by the options, the node ids within that group may be split. Similarly, if
    several groups together are smaller than the target workload, they may be grouped.

    The result is a list of lists containing `BatchInfo` objects, where each inner lists
    tries to have a workload as close as possible to the targeted value.

    Args:
        config: dictionary with settings for processing (the same as stored in JSON)
        options: command line options
        nodes: a tuple containing source and target node populations
    Returns:
        A list of `BatchInfo` groups splitting the passed in nodes
    """
    functions = config["manip"]["fcts"]
    if not (len(functions) == 1 and "model_pathways" in functions[0]):
        return [
            [BatchInfo(None, None, ids)] for ids in _get_simple_node_splits(config, options, nodes)
        ]

    src_ids = _get_region_map(nodes[0])
    dst_ids = _get_region_map(nodes[1])

    filename = options.config_path.parent / functions[0]["model_pathways"]
    pathways = pd.read_parquet(filename)

    payloads = (
        pathways.index.unique()
        .to_frame(index=False)
        .join(src_ids.map(len), on=[f"src_{c}" for c in src_ids.index.names])
        .groupby(["dst_hemisphere", "dst_region"])
        .sum(numeric_only=True)
    )

    # Consider two queues: jobs that split a region go into full_batches, whereas other
    # regions are too small to be split and can be grouped further and end up in
    # partial_batches.
    full_batches = []
    partial_batches = []
    for (dst_hemisphere, dst_region), (payload,) in payloads.iterrows():
        dst_count = max(1, int(options.target_payload / payload))
        ids = dst_ids.loc[dst_hemisphere, dst_region]
        sel = {"hemisphere": dst_hemisphere, "region": dst_region}
        if len(ids) <= dst_count:
            # Don't split, merge later
            partial_batches.append(BatchInfo(len(ids) * payload, sel, ids))
        else:
            splits = (len(ids) + dst_count - 1) // dst_count
            for split in np.array_split(ids, splits):
                # Single full batch, to be processed by itself
                full_batches.append([BatchInfo(len(split) * payload, sel, split)])
    # Groups partial batches
    full_batches.extend(BatchInfo.group_batches(partial_batches, options.target_payload))

    return full_batches


@dataclass
class BatchInfo:
    """An abstraction for a chunk of work"""

    payload: Union[int, None]
    """How much work this batch does, counting the number of src-node connections"""

    selection: Union[dict, None]
    """A SONATA nodest-style selection akin to property: value"""

    node_ids: list
    """Either a plain list of ids or a list of SONATA selection style ranges"""

    def __repr__(self):
        """Condensed representation of the class"""
        if len(self.node_ids) > 5:
            node_str = f"[{self.node_ids[0]}, ..., {self.node_ids[-1]}]"
        else:
            node_str = str(self.node_ids)
        return f"BatchInfo(payload={self.payload}, selection={self.selection}, node_ids={node_str})"

    @staticmethod
    def group_batches(all_batches: ["BatchInfo"], target_payload: int) -> [["BatchInfo"]]:
        """Partition batches into groups given a desired target_payload"""
        jobs = []
        while all_batches:
            batches = [all_batches.pop(0)]
            payload = batches[-1].payload
            while all_batches and target_payload >= payload + all_batches[0].payload:
                batches.append(all_batches.pop(0))
                payload += batches[-1].payload
            jobs.append(batches)
        return jobs

    def process_pathways(self, pathways=None):
        """Iterates over relevant pathway configurations for this batch

        Takes a Pandas Dataframe and yields relevant node ids, source and target selection
        dictionaries, as well as the pathway group configuration.
        """
        if pathways is None:
            yield self.node_ids, None, self.selection, None
            return

        if self.selection:
            index = pathways.index.to_frame()
            sel = []
            for k, v in self.selection.items():
                sel.append(index[f"dst_{k}"] == v)
            pathways = pathways[reduce(and_, sel)]

        grouped = pathways.groupby(pathways.index.names, observed=True)
        for idx, group in grouped:
            sel_src = {}
            sel_dst = {}
            for k, v in zip(pathways.index.names, idx):
                if k.startswith("src_"):
                    sel_src[k[4:]] = v
                elif k.startswith("dst_"):
                    sel_dst[k[4:]] = v
                else:
                    raise ValueError(f"Incompatible pathway index column: {k}")
            group = group.set_index(["src_type", "dst_type"])
            yield self.node_ids, sel_src, sel_dst, group
