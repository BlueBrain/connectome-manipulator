# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import numpy as np
import pytest
from mock import patch, Mock
from numpy.testing import assert_equal

from utils import create_fake_module
import connectome_manipulator.connectome_comparison.structural_comparator as test_module


def test_compute_results():
    module_name = "fake_compute"
    comp_dict = {
        "fct": {"source": module_name, "kwargs": {}},
        "res_sel": {"fake_key_1": "", "fake_key_2": ""},
    }

    # Create a fake module in which compute is a Mock and returns comp_dict['res_sel']
    full_module_name = f"connectome_manipulator.connectome_comparison.{module_name}"
    code = f'from mock import Mock; compute=Mock(return_value={comp_dict["res_sel"]})'
    fake_module = create_fake_module(full_module_name, code)

    # NOTE by herttuai on 06/10/2021:
    # this raises an exception, since
    #   np.isin(comp_dict['res_sel'], list(res_dict.keys()))
    #    > array(False)
    # but
    #   np.isin(list(comp_dict['res_sel']), list(res_dict))
    #    > array([True, True])
    #
    # I assume the latter is the expected behavior?
    with pytest.raises(AssertionError, match="ERROR: Specified results entry not found!"):
        test_module.compute_results(None, comp_dict)

    # Check that compute gets called
    fake_module.compute.assert_called()


def test_results_diff():
    diff = np.random.randint(1000)

    res_dict1 = {
        "dict_res": {"data": np.array((666, 1000))},
        "data": np.reshape(np.arange(4), (2, 2)),
        "ultimate_answer": 42,
    }

    res_dict2 = {
        "dict_res": {"data": res_dict1["dict_res"]["data"] + diff},
        "data": res_dict1["data"] + diff,
        "ultimate_answer": res_dict1["ultimate_answer"],
    }

    expected_dict = {
        "dict_res": {"data": np.full_like(res_dict1["dict_res"]["data"], diff)},
        "data": np.full_like(res_dict1["data"], diff),
        "ultimate_answer": res_dict1["ultimate_answer"],
    }

    assert_equal(test_module.results_diff(res_dict1, res_dict2), expected_dict)

    with pytest.raises(AssertionError, match="Results type mismatch!"):
        res_dict3 = res_dict2.copy()
        res_dict3["data"] = "will fail"
        test_module.results_diff(res_dict1, res_dict3)

    with pytest.raises(AssertionError, match="Results data shape mismatch!"):
        res_dict3 = res_dict2.copy()
        res_dict3["data"] = np.zeros(2)
        test_module.results_diff(res_dict1, res_dict3)

    with pytest.raises(AssertionError, match="Results inconsistency!"):
        res_dict3 = res_dict2.copy()
        res_dict3["ultimate_answer"] = 0
        test_module.results_diff(res_dict1, res_dict3)


def test_plot_results():
    module_name = "fake_plot"
    comp_dict = {"fct": {"source": module_name, "kwargs": {}}}
    full_module_name = f"connectome_manipulator.connectome_comparison.{module_name}"

    # Create fake module in which plot is an instance of Mock()
    fake_module = create_fake_module(full_module_name, "from mock import Mock; plot=Mock()")

    test_module.plot_results(None, None, {}, comp_dict)

    # Check that plot was called
    fake_module.plot.assert_called()
