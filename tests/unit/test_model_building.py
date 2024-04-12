# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import types

import numpy as np
from numpy.testing import assert_array_equal

from utils import setup_tempdir, TEST_DATA_DIR
import connectome_manipulator.model_building.model_building as test_module


# def test_get_model():
#     model = '(lambda a,b,c,d: (a,b,c,d))(test_arg_1, test_arg_2, test_kwarg_1, test_kwarg_2)'
#     model_inputs = ['test_arg_1', 'test_arg_2']
#     model_params = {'test_kwarg_1': 'value_1', 'test_kwarg_2': 'value_2'}

#     res = test_module.get_model(model, model_inputs, model_params)
#     assert isinstance(res, types.FunctionType)

#     args = ['test_value_1', 'test_value_2']
#     assert_array_equal(res(*args), args + list(model_params.values()))

#     # NOTE by herttuai on 08/10/2021:
#     # Is it intentional that kwargs are given as (kwargs) and not (**kwargs)?
#     kwargs = {'test_kwarg_1': 'test_value_3', 'test_kwarg_2': 'test_value_4'}
#     assert_array_equal(res(*args, kwargs), args + list(kwargs.values()))


def test_create_model_config_per_pathway():
    with setup_tempdir(__name__) as tempdir:
        model_config = {
            "model": {
                "name": "TestCreateModelConfigPerPathway",
                "fct": {"source": "test_model", "kwargs": {}},
            },
            "working_dir": os.path.join(tempdir, "working_dir"),
            "out_dir": os.path.join(tempdir, "out_dir"),
            "circuit_config": os.path.join(TEST_DATA_DIR, "circuit_sonata.json"),
        }

        res = test_module.create_model_config_per_pathway(model_config, "synapse_class")

        # Check for parameters that are same in each model
        assert all([c["circuit_config"] == model_config["circuit_config"] for c in res])

        for key in ["out_dir", "working_dir"]:
            assert len({c[key] for c in res}) == 1

        assert len({c["model"]["fct"]["source"] for c in res}) == 1

        # Check that models have different name
        assert len({c["model"]["name"] for c in res}) == len(res)

        # Check that kwargs are unique
        for i in range(len(res) - 1):
            for j in range(i + 1, len(res)):
                assert res[i]["model"]["fct"] != res[j]["model"]["fct"]
