# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Main module for structural connectome comparison:

- Loads two SONATA connectomes
- Extracts structural properties (compute_results), as specified by the structural comparator config dict (or re-loading them from disc, if computed earlier)
- Creates difference map between structural properties of the two connectomes
- Visualizes individual structural properties as well as their difference
"""

import importlib
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bluepysnap.circuit import Circuit

import connectome_manipulator


def compute_results(circuit, comp_dict):
    """Compute structural results used for circuit comparison."""
    comp_source = comp_dict["fct"]["source"]
    comp_kwargs = comp_dict["fct"]["kwargs"]

    comp_module = importlib.import_module(
        "connectome_manipulator.connectome_comparison." + comp_source
    )
    assert hasattr(
        comp_module, "compute"
    ), f'ERROR: Structural comparison module "{comp_source}" requires compute() function!'
    res_dict = comp_module.compute(circuit, **comp_kwargs)
    assert np.all(
        np.isin(comp_dict["res_sel"], list(res_dict.keys()))
    ), "ERROR: Specified results entry not found!"

    return res_dict


def results_diff(res_dict1, res_dict2):
    """Computes difference between two results data sets [recursively iterates through all sub-dicts to find 'data' entries; all other entries must be equal or will be removed]."""
    res_keys = np.intersect1d(list(res_dict1.keys()), list(res_dict2.keys()))
    for _k in np.setdiff1d(list(res_dict1.keys()), res_keys):
        del res_dict1[_k]
    for _k in np.setdiff1d(list(res_dict2.keys()), res_keys):
        del res_dict2[_k]
    assert np.all(
        [isinstance(res_dict1[k], type(res_dict2[k])) for k in res_keys]
    ), "ERROR: Results type mismatch!"

    diff_dict = {}
    for k in res_keys:
        if isinstance(res_dict1[k], dict):
            diff_dict[k] = results_diff(res_dict1[k], res_dict2[k])  # Recursive update
        else:
            if k == "data":
                assert (
                    res_dict1[k].shape == res_dict2[k].shape
                ), "ERROR: Results data shape mismatch!"
                # assert np.all(res_dict1[k] >= 0) and np.all(res_dict2[k] >= 0), 'ERROR: Negative results value(s) found!'
                diff_dict[k] = res_dict2[k].astype(float) - res_dict1[k].astype(
                    float
                )  # Difference matrix
            else:
                assert np.array_equal(res_dict1[k], res_dict2[k]), "ERROR: Results inconsistency!"
                diff_dict[k] = res_dict1[k]

    return diff_dict


def plot_results(res_dict, res_sel, plot_args, comp_dict):
    """Plot structural results."""
    comp_source = comp_dict["fct"]["source"]
    comp_kwargs = comp_dict["fct"]["kwargs"]

    comp_module = importlib.import_module(
        "connectome_manipulator.connectome_comparison." + comp_source
    )
    assert hasattr(
        comp_module, "plot"
    ), f'ERROR: Structural comparison module "{comp_source}" requires plot() function!'
    comp_module.plot(res_dict, res_sel, **plot_args, **comp_kwargs)


def main(structcomp_config, show_fig=False, force_recomp=False):  # pragma: no cover
    """Main entry point for structural connectome comparison."""
    print(f"VERSION INFO: connectome_manipulator {connectome_manipulator.__version__}")

    # Load circuits
    circuit_ids = sorted(structcomp_config["circuits"].keys())
    assert len(circuit_ids) == 2, "ERROR: Exactly two circuits required for comparison!"
    circuit_configs = [structcomp_config["circuits"][c]["circuit_config"] for c in circuit_ids]
    circuit_names = [structcomp_config["circuits"][c]["circuit_name"] for c in circuit_ids]

    circuits = [Circuit(cc) for cc in circuit_configs]
    print(f"INFO: {len(circuits)} circuits loaded:")
    for cc in circuit_configs:
        print("  " + cc)

    # Prepare saving
    out_dir = os.path.join(
        structcomp_config["out_dir"], f"{circuit_names[0]}_vs_{circuit_names[-1]}"
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Computation & plotting
    if np.isscalar(force_recomp):
        force_recomp = [force_recomp] * len(circuit_ids)
    else:
        assert len(force_recomp) == len(
            circuit_ids
        ), f'ERROR: {len(circuit_ids)} "force_recomp" entries expected!'

    for plot_dict in structcomp_config["plot_types"]:
        print(f'INFO: Preparing "{plot_dict["name"]}" plot(s)...')

        # Load results if available, otherwise compute
        res_dicts = []
        for cidx in range(len(circuit_ids)):
            res_file = os.path.join(
                structcomp_config["working_dir"],
                "data",
                circuit_names[cidx] + "_" + plot_dict["name"] + ".pickle",
            )
            if os.path.exists(res_file) and not force_recomp[cidx]:
                # Load from file
                print(f"INFO: Loading results from {res_file}")
                with open(res_file, "rb") as f:
                    res_dict = pickle.load(f)
            else:
                # Compute & save to file
                res_dict = compute_results(circuits[cidx], plot_dict)
                print(f"INFO: Writing resutls to {res_file}")
                if not os.path.exists(os.path.split(res_file)[0]):
                    os.makedirs(os.path.split(res_file)[0])
                with open(res_file, "wb") as f:
                    pickle.dump(res_dict, f, protocol=4)
            res_dicts.append(res_dict)
        res_dicts.append(results_diff(res_dicts[0], res_dicts[-1]))

        def get_flattened_data(d):
            """Returns raw (flattened) data from sparse matrix or numpy array"""
            if hasattr(d, "flatten"):  # Numpy arrays
                return d.flatten()
            else:
                if hasattr(d, "data") and hasattr(d.data, "flatten"):  # Sparse matrix
                    return d.data.flatten()
                else:
                    assert False, "ERROR: Flattened data extraction error!"

        # Plot results
        for res_sel in plot_dict["res_sel"]:
            # Determine common range of values for plotting
            range_prctile = plot_dict.get("range_prctile", 100)
            all_data = np.concatenate(
                [
                    get_flattened_data(res_dicts[cidx][res_sel]["data"])
                    for cidx in range(len(circuit_ids))
                ]
            ).astype(float)
            all_data = all_data[np.isfinite(all_data)]
            plot_range = [
                (
                    -np.percentile(-all_data[all_data < 0], range_prctile)
                    if np.any(all_data < 0)
                    else 0.0
                ),
                (
                    np.percentile(all_data[all_data > 0], range_prctile)
                    if np.any(all_data > 0)
                    else 0.0
                ),
            ]  # Common plot range
            diff_data = get_flattened_data(res_dicts[-1][res_sel]["data"])
            plot_range_diff = max(
                (
                    np.percentile(diff_data[diff_data > 0], range_prctile)
                    if np.any(diff_data > 0)
                    else 0.0
                ),
                (
                    np.percentile(-diff_data[diff_data < 0], range_prctile)
                    if np.any(diff_data < 0)
                    else 0.0
                ),
            )  # Diff plot range
            if plot_range_diff == 0.0:
                plot_range_diff = 1.0  # Arbitrary range needed for symmetric plotting around 0.0 in case of zero difference
            plot_range_diff = [-plot_range_diff, plot_range_diff]  # Symmetric plot range

            # Create figure
            plt.figure(figsize=plot_dict.get("fig_size", None))
            num_subplots = len(res_dicts)
            for sidx in range(num_subplots):
                plt.subplot(1, num_subplots, sidx + 1)
                if sidx < len(circuit_ids):  # Separate results plots for each circuit
                    plot_args = {
                        "fig_title": circuit_names[sidx],
                        "vmin": plot_range[0],
                        "vmax": plot_range[-1],
                        "isdiff": False,
                    }
                else:  # Difference plot between circuits
                    plot_args = {
                        "fig_title": "Diff",
                        "vmin": plot_range_diff[0],
                        "vmax": plot_range_diff[-1],
                        "isdiff": True,
                    }
                plot_results(
                    res_dicts[sidx][res_sel], res_dicts[sidx]["common"], plot_args, plot_dict
                )

            plt.suptitle(res_dicts[-1][res_sel]["name"])
            plt.tight_layout()

            # Save figure
            fig_file = plot_dict.get("fig_file", {})
            out_fn = os.path.abspath(
                os.path.join(
                    out_dir,
                    f'struct_comp-{plot_dict["name"]}-{res_sel}.{fig_file.get("format", "pdf")}',
                )
            )
            print(f"INFO: Saving {out_fn}...")
            plt.gcf().savefig(out_fn, dpi=fig_file.get("dpi", None))

            if show_fig:
                plt.show()
            else:
                plt.close("all")
