import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from utils import setup_tempdir, TEST_DATA_DIR
import connectome_manipulator.model_building.model_types as test_module


def test_LinDelayModel():
    np.random.seed(0)
    with setup_tempdir(__name__) as tempdir:
        # Check deterministic delay
        d_coef = [0.1, 0.003]
        d_min = 0.2
        model_dict = {
            "model": "LinDelayModel",
            "delay_mean_coefs": d_coef,
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
            "delay_mean_coefs": d_coef,
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
                            "std-within": 0.0,
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(src_type="L4_MC", tgt_type="L4_MC")
        assert props.shape[0] == N_syn  # Number of synapses per connection
        assert len(np.unique(props["conductance"])) == 1  # No variation within connection

        with pytest.raises(KeyError, match="WRONG_TYPE"):
            props = model.apply(src_type="WRONG_TYPE", tgt_type="WRONG_TYPE")

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
                            "std": 0.0,
                            "std-within": cond_std,
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(src_type="L4_MC", tgt_type="L4_MC")
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
                            "std": 0.0,
                            "std-within": cond_std,
                            "lower_bound": cond_range[0],
                            "upper_bound": cond_range[1],
                            "dtype": "float",
                        }
                    }
                },
            },
        }
        model = test_module.AbstractModel.model_from_dict(model_dict)
        props = model.apply(src_type="L4_MC", tgt_type="L4_MC")
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
            {"model": "ConnProb4thOrderLinInterpnReducedModel"}, {"p_conn_table": p_table}
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
            {"model": "ConnProb5thOrderLinInterpnReducedModel"}, {"p_conn_table": p_table}
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
