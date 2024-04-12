# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

import os
import numpy as np
import pandas as pd
import pytest
import re
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix, csc_matrix

from utils import setup_tempdir, TEST_DATA_DIR
import connectome_manipulator.model_building.model_types as test_module


def test_PathwayModel():
    model_dict = {"model": "ConnProbModel", "order": 1, "coeff_a": 3.14}

    pos = [0.0, 1.0, 2.0]
    spos = np.array([pos, pos, pos])
    tpos = np.array([pos])

    expected_default = np.array([[3.14], [3.14], [3.14]])

    with pytest.raises(ValueError):
        # Not specifying any mapping: can't specify pathway_specs
        new_dict = model_dict | {"pathway_specs": pd.DataFrame({})}
        test_module.AbstractModel.model_from_dict(new_dict)

    # Passing types with a default-only model
    default_model = test_module.AbstractModel.model_from_dict(model_dict)
    assert np.all(
        default_model.apply(src_type=[0, 1, 2], src_pos=spos, tgt_type=[34], tgt_pos=tpos) == 3.14
    )

    # Passing types
    better_model = test_module.AbstractModel.model_from_dict(
        model_dict
        | {
            "pathway_specs": pd.DataFrame(
                {"src_type": ["A"], "tgt_type": ["A"], "connprob_coeff_a": [2.72]}
            ).set_index(["src_type", "tgt_type"]),
            "src_type_map": {"A": 0, "B": 1},
            "tgt_type_map": {"A": 0, "B": 1},
        }
    )
    assert np.all(
        better_model.apply(src_type=["A", "A", "A"], src_pos=spos, tgt_type=["A"], tgt_pos=tpos)
        == 2.72
    )
    assert np.all(
        better_model.apply(src_type=[0, 0, 0], src_pos=spos, tgt_type=[0], tgt_pos=tpos) == 2.72
    )

    assert_array_equal(
        better_model.apply(src_type=["A", "A", "A"], src_pos=spos, tgt_type=["B"], tgt_pos=tpos),
        expected_default,
    )
    assert_array_equal(
        better_model.apply(src_type="A", src_pos=spos, tgt_type=["B"], tgt_pos=tpos),
        expected_default,
    )
    assert_array_equal(
        better_model.apply(src_type=[1, 1, 1], src_pos=spos, tgt_type=[0], tgt_pos=tpos),
        expected_default,
    )
    assert_array_equal(
        better_model.apply(src_type=1, src_pos=spos, tgt_type=["B"], tgt_pos=tpos), expected_default
    )
    assert_array_equal(better_model.apply(src_pos=spos, tgt_pos=tpos), expected_default)

    with pytest.raises(KeyError):
        better_model.apply(src_type=["A", "B", "C"], src_pos=spos, tgt_pos=tpos, tgt_type=["D"])
    with pytest.raises(IndexError):
        better_model.apply(src_type=[123, 456, 789], src_pos=spos, tgt_pos=tpos, tgt_type=[-666])
    with pytest.raises(ValueError):
        better_model.apply(src_type=[1.0, 2.0, 3.0], src_pos=spos, tgt_pos=tpos, tgt_type=[-6.66])


def test_LinDelayModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        # Check deterministic delay
        d_coef = [0.1, 0.003]
        d_min = 0.2
        model_dict = {
            "model": "LinDelayModel",
            "delay_mean_coeff_a": d_coef[0],
            "delay_mean_coeff_b": d_coef[1],
            "delay_std": 0.0,
            "delay_min": d_min,
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        dist = np.arange(0.0, 1000.0, 1.0)
        d = model.apply(distance=dist)
        assert_array_equal(d, np.maximum(d_min, d_coef[0] + d_coef[1] * dist))

        # Check delay statistics
        d_std = 0.5
        model_dict = {
            "model": "LinDelayModel",
            "delay_mean_coeff_a": d_coef[0],
            "delay_mean_coeff_b": d_coef[1],
            "delay_std": d_std,
            "delay_min": d_min,
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        dist = 1000
        N = 10000
        d = model.apply(distance=np.full(N, dist))
        assert np.isclose(np.mean(d), d_coef[0] + d_coef[1] * dist, rtol=1e-2)
        assert np.isclose(np.std(d), d_std, rtol=2e-2)

        # Check model saving/loading
        model_name = "LinDelayModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_PosMapModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        # Check pos table access
        N = 100
        pos_data = np.random.rand(N, 3)
        model_dict = {"model": "PosMapModel"}
        data_dict = {"pos_table": pd.DataFrame(pos_data, columns=["x", "y", "z"])}
        model = test_module.AbstractModel.model_from_dict(model_dict, data_dict)
        pos_mapping = model.apply(gids=np.arange(N))
        assert_array_equal(pos_mapping, pos_data)

        # Check model saving/loading
        model_name = "PosMapModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnPropsModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        N_syn = 10000
        cond_mean = 5.0
        cond_std = 0.5
        # Check properties within connection
        model_dict = {
            "model": "ConnPropsModel",
            "src_types": ["L4_MC"],
            "tgt_types": ["L4_MC"],
            "prop_stats": {
                "n_syn_per_conn": {
                    "L4_MC": {"L4_MC": {"type": "constant", "mean": N_syn, "dtype": "int"}}
                },
                "conductance": {
                    "L4_MC": {
                        "L4_MC": {
                            "type": "normal",
                            "mean": cond_mean,
                            "std": cond_std,
                            "shared_within": True,
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(
            src_type="L4_MC", tgt_type="L4_MC"
        )  # Single connection with N_syn synapses
        assert props.shape[0] == N_syn  # Number of synapses per connection
        assert len(np.unique(props["conductance"])) == 1  # No variation within connection

        with pytest.raises(KeyError, match="WRONG_TYPE"):
            props = model.apply(src_type="WRONG_TYPE", tgt_type="WRONG_TYPE")

        # Check with #synapses externally provided (overwriting internal N_syn)
        nsyn_ext = 10
        props = model.apply(
            src_type="L4_MC", tgt_type="L4_MC", n_syn=nsyn_ext
        )  # Single connection with nsyn synapses
        assert props.shape[0] == nsyn_ext  # Number of synapses per connection
        assert len(np.unique(props["conductance"])) == 1  # No variation within connection

        # Check model w/o #syn/conn provided
        model_dict = {
            "model": "ConnPropsModel",
            "src_types": ["L4_MC"],
            "tgt_types": ["L4_MC"],
            "prop_stats": {
                "conductance": {
                    "L4_MC": {
                        "L4_MC": {
                            "type": "normal",
                            "mean": cond_mean,
                            "std": cond_std,
                            "shared_within": True,
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        with pytest.raises(AssertionError, match=re.escape('"n_syn_per_conn" missing')):
            props = model.apply(src_type="L4_MC", tgt_type="L4_MC")
        props = model.apply(src_type="L4_MC", tgt_type="L4_MC", n_syn=N_syn)
        assert props.shape[0] == N_syn  # Number of synapses per connection
        assert len(np.unique(props["conductance"])) == 1  # No variation within connection

        # Check properties across connection
        model_dict = {
            "model": "ConnPropsModel",
            "src_types": ["L4_MC"],
            "tgt_types": ["L4_MC"],
            "prop_stats": {
                "n_syn_per_conn": {
                    "L4_MC": {"L4_MC": {"type": "constant", "mean": N_syn, "dtype": "int"}}
                },
                "conductance": {
                    "L4_MC": {
                        "L4_MC": {
                            "type": "normal",
                            "mean": cond_mean,
                            "std": cond_std,
                            "shared_within": False,
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(
            src_type="L4_MC", tgt_type="L4_MC"
        )  # Single connection with N_syn synapses
        assert props.shape[0] == N_syn  # Number of synapses per connection
        assert np.isclose(cond_mean, np.mean(props["conductance"]), rtol=1e-2)
        assert np.isclose(cond_std, np.std(props["conductance"]), rtol=2e-2)

        # Check property bounds
        cond_std = 5.0
        cond_range = [cond_mean - 1.0, cond_mean + 1.0]
        model_dict = {
            "model": "ConnPropsModel",
            "src_types": ["L4_MC"],
            "tgt_types": ["L4_MC"],
            "prop_stats": {
                "n_syn_per_conn": {
                    "L4_MC": {"L4_MC": {"type": "constant", "mean": N_syn, "dtype": "int"}}
                },
                "conductance": {
                    "L4_MC": {
                        "L4_MC": {
                            "type": "normal",
                            "mean": cond_mean,
                            "std": cond_std,
                            "shared_within": False,
                            "lower_bound": cond_range[0],
                            "upper_bound": cond_range[1],
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(
            src_type="L4_MC", tgt_type="L4_MC"
        )  # Single connection with N_syn synapses
        assert np.all(props["conductance"] <= cond_range[1])
        assert np.all(props["conductance"] >= cond_range[0])

        # Check model saving/loading
        model_name = "ConnPropsModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProb1stOrderModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        src_pos = np.zeros([100, 3])
        tgt_pos = np.random.rand(1000, 3) * 1000 - 500
        # Check prob model access
        p_ref = 0.01
        model_dict = {"model": "ConnProb1stOrderModel", "p_conn": p_ref}
        model = test_module.AbstractModel.model_from_dict(model_dict)
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert p.shape[0] == src_pos.shape[0] and p.shape[1] == tgt_pos.shape[0]
        assert np.all(p == p_ref)  # Constant conn. prob.

        # Check model saving/loading
        model_name = "ConnProb1stOrderModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProb2ndOrderExpModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        src_pos = np.zeros([100, 3])
        tgt_pos = np.random.rand(1000, 3) * 1000 - 500
        # Check prob model access
        scale = 0.1
        exponent = 0.006
        model_dict = {"model": "ConnProb2ndOrderExpModel", "scale": scale, "exponent": exponent}
        model = test_module.AbstractModel.model_from_dict(model_dict)
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        dist = np.sqrt(np.sum(tgt_pos**2, 1))
        assert p.shape[0] == src_pos.shape[0] and p.shape[1] == tgt_pos.shape[0]
        assert np.all(
            p[0, :] == scale * np.exp(-exponent * dist)
        )  # Exponential dist.-dep. conn. prob.

        # Check model saving/loading
        model_name = "ConnProb2ndOrderExpModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProb3rdOrderExpModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        src_pos = np.zeros([100, 3])
        tgt_pos = np.random.rand(1000, 3) * 1000 - 500
        # Check prob model access
        scale = [0.09, 0.11]
        exponent = [0.008, 0.004]
        model_dict = {
            "model": "ConnProb3rdOrderExpModel",
            "scale_P": scale[0],
            "scale_N": scale[1],
            "exponent_P": exponent[0],
            "exponent_N": exponent[1],
            "bip_coord": tgt_pos.shape[1] - 1,
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        dist = np.sqrt(np.sum(tgt_pos**2, 1))
        P_sel = tgt_pos[:, -1] > 0
        N_sel = tgt_pos[:, -1] < 0
        assert p.shape[0] == src_pos.shape[0] and p.shape[1] == tgt_pos.shape[0]
        assert np.all(
            p[0, P_sel] == scale[0] * np.exp(-exponent[0] * dist[P_sel])
        )  # Bipolar (pos) exponential dist.-dep. conn. prob.
        assert np.all(
            p[0, N_sel] == scale[1] * np.exp(-exponent[1] * dist[N_sel])
        )  # Bipolar (neg) exponential dist.-dep. conn. prob.

        # Check special case (dist==0.0)
        tgt_pos[:, -1] = 0.0
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        dist = np.sqrt(np.sum(tgt_pos**2, 1))
        assert p.shape[0] == src_pos.shape[0] and p.shape[1] == tgt_pos.shape[0]
        assert np.all(
            p[0, :]
            == 0.5
            * (scale[0] * np.exp(-exponent[0] * dist) + scale[1] * np.exp(-exponent[1] * dist))
        )  # Bipolar (pos/neg average) exponential dist.-dep. conn. prob.

        # Check model saving/loading
        model_name = "ConnProb3rdOrderExpModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProb4thOrderLinInterpnReducedModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        dr_pos = np.arange(0.5, 1.6, 1.0)  # Position at bin centers
        dz_pos = np.arange(0.5, 2.6, 1.0)  # Position at bin centers
        p_ref = np.random.rand(len(dr_pos), len(dz_pos))
        p_table = pd.DataFrame(
            p_ref.flatten(),
            index=pd.MultiIndex.from_product([dr_pos, dz_pos], names=["dr", "dz"]),
            columns=["p"],
        )
        model = test_module.AbstractModel.model_from_dict(
            {"model": "ConnProb4thOrderLinInterpnReducedModel", "axial_coord": 2},
            {"p_conn_table": p_table},
        )

        # Check prob model access (at center positions)
        src_pos = np.zeros([1, 3])
        tgt_pos = np.array(
            [
                p_table.index.get_level_values(0).to_numpy(),
                np.zeros(p_table.shape[0]),
                p_table.index.get_level_values(1).to_numpy(),
            ]
        ).T
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(p_ref.flatten(), p))

        # Check prob model access (interpolation)
        src_pos = np.zeros([1, 3])
        tgt_pos = np.array([[np.mean(dr_pos[:2]), 0.0, dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(np.mean(p_ref[:2, 0]), p))
        tgt_pos = np.array([[dr_pos[0], 0.0, np.mean(dz_pos[:2])]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(np.mean(p_ref[0, :2]), p))
        tgt_pos = np.array([[np.mean(dr_pos[:2]), 0.0, np.mean(dz_pos[:2])]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(np.mean(p_ref[:2, :2]), p))

        # Check prob model access (extrapolation)
        src_pos = np.zeros([1, 3])
        tgt_pos = np.array([[0.0, 0.0, dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(p_ref[0, 0], p))  # Symmetric towards dr=0
        tgt_pos = np.array([[max(dr_pos) + 0.1, 0.0, dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        tgt_pos = np.array([[dr_pos[0], 0.0, min(dz_pos) - 0.1]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        tgt_pos = np.array([[dr_pos[0], 0.0, max(dz_pos) + 0.1]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions

        # Check model saving/loading
        model_name = "ConnProb4thOrderLinInterpnReducedModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProb5thOrderLinInterpnReducedModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        z_pos = np.arange(0.5, 3.6, 1.0)  # Position at bin centers
        dr_pos = np.arange(0.5, 1.6, 1.0)  # Position at bin centers
        dz_pos = np.arange(0.5, 2.6, 1.0)  # Position at bin centers
        p_ref = np.random.rand(len(z_pos), len(dr_pos), len(dz_pos))
        p_table = pd.DataFrame(
            p_ref.flatten(),
            index=pd.MultiIndex.from_product([z_pos, dr_pos, dz_pos], names=["z", "dr", "dz"]),
            columns=["p"],
        )
        model = test_module.AbstractModel.model_from_dict(
            {"model": "ConnProb5thOrderLinInterpnReducedModel", "axial_coord": 2},
            {"p_conn_table": p_table},
        )

        # Check prob model access (at center positions)
        for z_idx in range(len(z_pos)):
            z = z_pos[z_idx]
            src_pos = np.array([[0.0, 0.0, z]])
            tgt_pos = np.array(
                [
                    p_table[p_table.index.get_level_values(0) == z]
                    .index.get_level_values(1)
                    .to_numpy(),
                    np.zeros(np.sum(p_table.index.get_level_values(0) == z)),
                    z
                    + p_table[p_table.index.get_level_values(0) == z]
                    .index.get_level_values(2)
                    .to_numpy(),
                ]
            ).T
            p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
            assert np.all(np.isclose(p_ref[z_idx, :, :].flatten(), p))

        # Check prob model access (interpolation)
        for z_idx in range(len(z_pos)):
            z = z_pos[z_idx]
            src_pos = np.array([[0.0, 0.0, z]])
            tgt_pos = np.array([[np.mean(dr_pos[:2]), 0.0, z + dz_pos[0]]])
            p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
            assert np.all(np.isclose(np.mean(p_ref[z_idx, :2, 0]), p))
            tgt_pos = np.array([[dr_pos[0], 0.0, z + np.mean(dz_pos[:2])]])
            p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
            assert np.all(np.isclose(np.mean(p_ref[z_idx, 0, :2]), p))
            tgt_pos = np.array([[np.mean(dr_pos[:2]), 0.0, z + np.mean(dz_pos[:2])]])
            p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
            assert np.all(np.isclose(np.mean(p_ref[z_idx, :2, :2]), p))
        src_pos = np.array([[0.0, 0.0, np.mean(z_pos[:2])]])
        tgt_pos = np.array([[dr_pos[0], 0.0, np.mean(z_pos[:2]) + dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(np.mean(p_ref[:2, 0, 0]), p))
        tgt_pos = np.array([[np.mean(dr_pos[:2]), 0.0, np.mean(z_pos[:2]) + np.mean(dz_pos[:2])]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(np.mean(p_ref[:2, :2, :2]), p))

        # Check prob model access (extrapolation)
        src_pos = np.array([[0.0, 0.0, z_pos[0]]])
        tgt_pos = np.array([[0.0, 0.0, z_pos[0] + dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(p_ref[0, 0, 0], p))  # Symmetric towards dr=0
        tgt_pos = np.array([[max(dr_pos) + 0.1, 0.0, z_pos[0] + dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        tgt_pos = np.array([[dr_pos[0], 0.0, z_pos[0] + min(dz_pos) - 0.1]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        tgt_pos = np.array([[dr_pos[0], 0.0, z_pos[0] + max(dz_pos) + 0.1]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        src_pos = np.array([[0.0, 0.0, min(z_pos) - 0.1]])
        tgt_pos = np.array([[dr_pos[0], 0.0, min(z_pos) - 0.1 + dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions
        src_pos = np.array([[0.0, 0.0, max(z_pos) + 0.1]])
        tgt_pos = np.array([[dr_pos[0], 0.0, max(z_pos) + 0.1 + dz_pos[0]]])
        p = model.apply(src_pos=src_pos, tgt_pos=tgt_pos)
        assert np.all(np.isclose(0.0, p))  # Zero outside range of data positions

        # Check model saving/loading
        model_name = "ConnProb5thOrderLinInterpnReducedModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )


def test_ConnProbAdjModel():
    np.random.seed(999)
    with setup_tempdir(__name__) as tempdir:
        # Define (random) adjacency matrix
        src_node_ids = np.arange(5, 20)
        tgt_node_ids = np.arange(50, 100)
        adj_mat = csc_matrix(np.random.rand(len(src_node_ids), len(tgt_node_ids)) > 0.75)

        # Prepare data frames
        src_nodes_table = pd.DataFrame(src_node_ids, columns=["src_node_ids"])
        tgt_nodes_table = pd.DataFrame(tgt_node_ids, columns=["tgt_node_ids"])
        rows, cols = adj_mat.nonzero()
        adj_table = pd.DataFrame({"row_ind": rows, "col_ind": cols})

        # Test non-inverted & inverted model
        for inverted in [False, True]:
            ## (a) Build
            model_dict = {"model": "ConnProbAdjModel", "inverted": inverted}
            data_dict = {
                "src_nodes_table": src_nodes_table,
                "tgt_nodes_table": tgt_nodes_table,
                "adj_table": adj_table,
            }
            model = test_module.AbstractModel.model_from_dict(model_dict, data_dict)

            ## (b) Access
            assert_array_equal(src_node_ids, model.get_src_nids())
            assert_array_equal(tgt_node_ids, model.get_tgt_nids())
            assert_array_equal(adj_mat.toarray(), model.get_adj_matrix().toarray())
            assert model.is_inverted() == inverted
            if inverted:
                assert_array_equal(
                    adj_mat.toarray().astype(float),
                    1.0 - model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids),
                )
            else:
                assert_array_equal(
                    adj_mat.toarray().astype(float),
                    model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids),
                )

            ## (c) Load/save
            model_name = "ConnProbAdjModel_TEST"
            model.save_model(tempdir, model_name)
            model2 = test_module.AbstractModel.model_from_file(
                os.path.join(tempdir, model_name + ".json")
            )
            assert model.get_param_dict() == model2.get_param_dict()
            assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
            assert np.all(
                [
                    np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                    for k in model.get_data_dict().keys()
                ]
            )


def test_LookupTableModel():
    np.random.seed(999)
    with setup_tempdir(__name__) as tempdir:
        # Define (random) LUT matrix
        src_node_ids = np.arange(5, 20)
        tgt_node_ids = np.arange(50, 100)
        src_nodes_table = pd.DataFrame(src_node_ids, columns=["src_node_ids"])
        tgt_nodes_table = pd.DataFrame(tgt_node_ids, columns=["tgt_node_ids"])
        mat = np.random.rand(len(src_node_ids), len(tgt_node_ids))
        mat[mat < 0.5] = 0  # Make sparse
        mat_coo = coo_matrix(mat, shape=(len(src_node_ids), len(tgt_node_ids)))
        lookup_table = pd.DataFrame(
            {"row_ind": mat_coo.row, "col_ind": mat_coo.col, "value": mat_coo.data}
        )

        # Init. model
        model = test_module.LookupTableModel(
            src_nodes_table=src_nodes_table,
            tgt_nodes_table=tgt_nodes_table,
            lookup_table=lookup_table,
        )

        # Check model
        assert_array_equal(src_node_ids, model.get_src_nids())
        assert_array_equal(tgt_node_ids, model.get_tgt_nids())
        assert_array_equal(mat, model.lut_mat.todense())
        assert_array_equal(mat, model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids))

        ## Check load/save
        model_name = "LookupTableModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )

        # Check (static) init from sparse matrix
        with pytest.raises(AssertionError, match=re.escape("Matrix must be in sparse format")):
            model = test_module.LookupTableModel.init_from_sparse_matrix(
                mat, src_node_ids, tgt_node_ids
            )
        model = test_module.LookupTableModel.init_from_sparse_matrix(
            mat_coo.tocsc(), src_node_ids, tgt_node_ids
        )
        assert_array_equal(src_node_ids, model.get_src_nids())
        assert_array_equal(tgt_node_ids, model.get_tgt_nids())
        assert_array_equal(mat, model.lut_mat.todense())
        assert_array_equal(mat, model.apply(src_nid=src_node_ids, tgt_nid=tgt_node_ids))


def test_PropsTableModel():
    np.random.seed(999)
    with setup_tempdir(__name__) as tempdir:
        pnames = ["a", "b", "c", "d", "e"]
        ptab = pd.DataFrame(np.random.rand(20, 5), columns=pnames)
        id_max = ptab.shape[0] >> 2
        id_offs = [5, 10]  # Source/target offsets

        ## Init model
        with pytest.raises(AssertionError, match=re.escape("@source_node column required!")):
            model = test_module.PropsTableModel(props_table=ptab)
        ptab["@source_node"] = np.random.randint(id_max, size=ptab.shape[0]) + id_offs[0]
        with pytest.raises(AssertionError, match=re.escape("@target_node column required!")):
            model = test_module.PropsTableModel(props_table=ptab)
        ptab["@target_node"] = np.random.randint(id_max, size=ptab.shape[0]) + id_offs[1]
        model = test_module.PropsTableModel(props_table=ptab)

        ## Check model
        # (a) Check node IDs
        src_nids = np.unique(ptab["@source_node"])
        tgt_nids = np.unique(ptab["@target_node"])
        assert_array_equal(src_nids, model.get_src_nids())
        assert_array_equal(tgt_nids, model.get_tgt_nids())
        conns, cnts = model.get_src_tgt_counts()
        ref_conns, ref_cnts = np.unique(
            ptab[["@source_node", "@target_node"]], axis=0, return_counts=True
        )
        assert_array_equal(conns, ref_conns)
        assert_array_equal(cnts, ref_cnts)
        assert_array_equal(model.get_property_names(), pnames)

        # (b) Check model access (single node IDs)
        for sid in src_nids:
            for tid in tgt_nids:
                ref_tab = ptab[
                    np.logical_and(ptab["@source_node"] == sid, ptab["@target_node"] == tid)
                ][pnames]
                assert_array_equal(model.apply(src_nid=sid, tgt_nid=tid), ref_tab)

        # (c) Check model access (multiple node IDs)
        for sid in src_nids:
            ref_tab = ptab[
                np.logical_and(ptab["@source_node"] == sid, np.isin(ptab["@target_node"], tgt_nids))
            ][pnames]
            assert_array_equal(model.apply(src_nid=sid, tgt_nid=tgt_nids), ref_tab)
        for tid in tgt_nids:
            ref_tab = ptab[
                np.logical_and(np.isin(ptab["@source_node"], src_nids), ptab["@target_node"] == tid)
            ][pnames]
            assert_array_equal(model.apply(src_nid=src_nids, tgt_nid=tid), ref_tab)
        ref_tab = ptab[pnames]
        assert_array_equal(model.apply(src_nid=src_nids, tgt_nid=tgt_nids), ref_tab)

        # (d) Check model access (single properties)
        for p in pnames:
            assert_array_equal(
                model.apply(src_nid=src_nids, tgt_nid=tgt_nids, prop_names=[p]), ptab[[p]]
            )

        # (e) Check model access (full table, including @source/target_node)
        assert_array_equal(
            model.apply(src_nid=src_nids, tgt_nid=tgt_nids, prop_names=ptab.columns), ptab
        )

        # (f) Check model access (selected number of entries)
        for _n in [-1, 2 * ptab.shape[0]]:  # => Assertion error
            with pytest.raises(
                AssertionError, match=re.escape("Selected number of elements out of range!")
            ):
                model.apply(src_nid=src_nids, tgt_nid=tgt_nids, num_sel=_n)
        assert model.apply(src_nid=src_nids, tgt_nid=tgt_nids, num_sel=0).size == 0
        assert_array_equal(
            model.apply(src_nid=src_nids, tgt_nid=tgt_nids, num_sel=ptab.shape[0]), ptab[pnames]
        )
        for _n in range(1, ptab.shape[0], 5):
            assert_array_equal(
                model.apply(src_nid=src_nids, tgt_nid=tgt_nids, num_sel=_n), ptab[pnames][:_n]
            )

        ## Check load/save
        model_name = "PropsTableModel_TEST"
        model.save_model(tempdir, model_name)
        model2 = test_module.AbstractModel.model_from_file(
            os.path.join(tempdir, model_name + ".json")
        )
        assert model.get_param_dict() == model2.get_param_dict()
        assert sorted(model.get_data_dict().keys()) == sorted(model2.get_data_dict().keys())
        assert np.all(
            [
                np.array_equal(model.get_data_dict()[k], model2.get_data_dict()[k])
                for k in model.get_data_dict().keys()
            ]
        )
