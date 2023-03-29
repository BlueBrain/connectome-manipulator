"""Manipulation base module.

Description: This module contains the Manipulation abstract base class of which all manipulation
classes must inherit and implement its methods.
"""
from abc import ABCMeta, abstractmethod
import inspect
import os.path

from connectome_manipulator import log


class MetaManipulation(ABCMeta):
    """Meta class to manage Manipulation algorithm classes.

    The only purpose of this Meta class derived from ABCMeta is to automatically register
    existing manipulation algorithms and give the programmer a helper function to load a class from
    an fct string in the config file.
    """

    __manipulations = {}
    __instances = {}

    def __init__(cls, name, bases, attrs) -> None:
        """Register the implementing class (if concrete) into a dictionary for later lookup"""
        if not inspect.isabstract(cls):
            modpath = inspect.getfile(cls)
            mod = os.path.splitext(os.path.basename(modpath))[0]
            cls.__manipulations[mod] = cls
        ABCMeta.__init__(cls, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        """We want the Manipulation subclasses to be singletons"""
        if cls not in cls.__instances:
            cls.__instances[cls] = super(MetaManipulation, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]

    @classmethod
    def get(mcs, name):
        """Returns a concrete Manipulation class given a string"""
        log.log_assert(name in mcs.__manipulations, f"Manipulation algorithm {name} does not exist")
        return mcs.__manipulations[name]


class Manipulation(metaclass=MetaManipulation):
    """Manipulation algorithm base class

    The abstract base class of which all manipulation classes must inherit and implement its methods.
    """

    @abstractmethod
    def apply(self, edges_table, nodes, aux_config, **kwargs):
        """An abstract method for the actual application of the algorithm

        This funciton is to be implemented by concrete Manipulation subclasses.
        """


class MorphologyCachingManipulation(Manipulation):
    """An abstract Manipulation with morphology caching

    This is a abstract Manipulation class that additionally provides a cache for morphologies,
    such that they can be reused on different invokations of apply without having to lad them from
    the filesystem.
    """

    # pylint: disable=abstract-method

    def __init__(self):
        """Setup a morphologies cache when initializing"""
        self.morphologies = {}

    def _get_tgt_morph(self, tgt_morph, morph_ext, node_id):
        """Access function (incl. transformation!), using specified format (swc/h5/...)"""
        if node_id in self.morphologies:
            return self.morphologies[node_id]
        else:
            morph = tgt_morph.get(node_id, transform=True, extension=morph_ext)
            self.morphologies[node_id] = morph
            return morph
