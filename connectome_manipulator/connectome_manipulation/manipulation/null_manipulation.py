# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Null manipulation module - does not do anything."""

from connectome_manipulator import log
from connectome_manipulator.connectome_manipulation.manipulation import Manipulation


class NullManipulation(Manipulation):
    """Dummy manipulation class performing no manipulation at all:

    This manipulation is intended as a control condition to run the manipulation
    pipeline without actually manipulating the connectome, and can be applied through
    the :func:`apply` method.
    """

    def apply(self, split_ids, **kwargs):
        """Applies a null manipulation, i.e., the connectome is left unchanged.

        Args:
            split_ids (list-like): List of neuron IDs that are part of the current data split; will be automatically provided by the manipulator framework - Not used
            **kwargs: Additional keyword arguments - Not used
        """
        log.info("Nothing to do")
