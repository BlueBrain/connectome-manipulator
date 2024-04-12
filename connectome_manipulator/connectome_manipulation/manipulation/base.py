# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Manipulation base module.

Description: This module contains the Manipulation abstract base class of which all manipulation
classes must inherit and implement its methods.
"""

from abc import ABCMeta, abstractmethod
import inspect
import os.path

from bluepysnap.morph import MorphHelper
from bluepysnap.sonata_constants import Node
from morphio.mut import Morphology
import numpy as np

from connectome_manipulator import access_functions
from connectome_manipulator import log
from connectome_manipulator import utils
from connectome_manipulator.access_functions import get_enumeration_map


class MetaManipulation(ABCMeta):
    """Meta class to manage Manipulation algorithm classes.

    The only purpose of this Meta class derived from ABCMeta is to automatically register
    existing manipulation algorithms and give the programmer a helper function to load a class from
    an fct string in the config file.
    """

    __manipulations = {}

    def __init__(cls, name, bases, attrs) -> None:
        """Register the implementing class (if concrete) into a dictionary for later lookup"""
        if not inspect.isabstract(cls):
            modpath = inspect.getfile(cls)
            mod = os.path.splitext(os.path.basename(modpath))[0]
            cls.__manipulations[mod] = cls
        ABCMeta.__init__(cls, name, bases, attrs)

    @classmethod
    def get(mcs, name):
        """Returns a concrete Manipulation class given a string"""
        log.log_assert(name in mcs.__manipulations, f"Manipulation algorithm {name} does not exist")
        return mcs.__manipulations[name]


class Manipulation(metaclass=MetaManipulation):
    """Manipulation algorithm base class

    The abstract base class of which all manipulation classes must inherit and implement its methods.
    """

    def __init__(self, nodes, writer=None, split_index=0, split_total=1):
        """Initialize with the nodes and split_ids"""
        self.split_index = split_index
        self.split_total = split_total
        self.nodes = nodes
        self.writer = writer

        self.src_type_map = get_enumeration_map(self.nodes[0], "mtype")
        self.tgt_type_map = get_enumeration_map(self.nodes[1], "mtype")

    @abstractmethod
    def apply(self, split_ids, **kwargs):
        """An abstract method for the actual application of the algorithm

        This funciton is to be implemented by concrete Manipulation subclasses.
        """


class MorphologyCachingManipulation(Manipulation):
    """An abstract Manipulation with morphology caching

    This is a abstract Manipulation class that additionally provides a cache for morphologies,
    such that they can be reused on different invokations of apply without having to load them from
    the filesystem.
    """

    # pylint: disable=abstract-method

    def __init__(self, nodes, writer=None, split_index=0, split_total=1):
        """Initialize the MorphHelper object needed later."""
        super().__init__(nodes, writer, split_index, split_total)
        self.morpho_helper = MorphHelper(
            self.nodes[1].config.get("morphologies_dir"),
            self.nodes[1],
            self.nodes[1].config.get("alternate_morphologies"),
        )

    def _get_tgt_morphs(self, morph_ext, tgt_node_sel):
        """Access function (incl. transformation!), using specified format (swc/h5/...)"""
        morphology_paths = access_functions.get_morphology_paths(
            self.nodes[1], tgt_node_sel, self.morpho_helper, morph_ext
        )
        morphologies = []
        for mp in morphology_paths:
            morphologies.append(Morphology(mp))
        return self._transform(morphologies, tgt_node_sel)

    def _transform(self, morphs, node_sel):
        rotations = access_functions.orientations(self.nodes[1], node_sel)
        positions = access_functions.get_nodes(self.nodes[1], node_sel)
        positions = positions[[Node.X, Node.Y, Node.Z]].to_numpy()
        for m, r, p in zip(morphs, rotations, positions):
            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = p
            utils.transform(m, T)
            yield m.as_immutable()
