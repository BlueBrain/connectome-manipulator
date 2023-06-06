import argparse
import json

import numpy as np
import pandas as pd

from connectome_manipulator.model_building.model_types import AbstractModel

parser = argparse.ArgumentParser("converts a pure JSON config into a hybrid")
parser.add_argument("json")
parser.add_argument("output_basename")
args = parser.parse_args()

with open(args.json) as fd:
    original_config = json.load(fd)


def transform_prob_pathways(dat):
    """Transforms a pathway dictionary into a DataFrame"""
    srcs = []
    dsts = []
    orders = []
    coeffs_a = []
    coeffs_b = []

    result = {}
    for src_mtype, values in dat.items():
        for dst_mtype, params in values.items():
            srcs.append(src_mtype)
            dsts.append(dst_mtype)
            match params["coeffs"]:
                case [a]:
                    orders.append(1)
                    coeffs_a.append(a)
                    coeffs_b.append(np.nan)
                case [a, b]:
                    orders.append(2)
                    coeffs_a.append(a)
                    coeffs_b.append(b)
                case _:
                    raise ValueError(f"too many components in {params}")
    return pd.DataFrame(
        {
            "src_type": srcs,
            "dst_type": dsts,
            "connprob_order": orders,
            "connprob_coeff_a": coeffs_a,
            "connprob_coeff_b": coeffs_b,
        }
    )


param_lookup = {}
shorthand_lookup = {}
idx = 0


def transform_function(fct_dict):
    """Will modify `fct_dict` by removing pathway information and return it separately as a DataFrame"""

    global idx
    print(idx)
    idx += 1

    # Construct DataFrame based on the `ConnProbModel`
    data = fct_dict["kwargs"]["prob_model_spec"].pop("pathway_specs", {})
    df = transform_prob_pathways(data)
    fct_dict["model_pathways"] = f"{args.output_basename}.parquet"

    # Delay model coefficients need to be split
    if coeffs := fct_dict["kwargs"]["delay_model_spec"].pop("delay_mean_coefs"):
        fct_dict["kwargs"]["delay_model_spec"]["delay_mean_coeff_a"] = coeffs[0]
        fct_dict["kwargs"]["delay_model_spec"]["delay_mean_coeff_b"] = coeffs[1]

    # Simplify config: `kwargs` will be transformed into `model_config`
    kwargs = fct_dict.pop("kwargs")
    models = fct_dict.setdefault("model_config", {})
    defaults = {}

    for k, v in kwargs.items():
        if k.endswith("_spec"):
            if "pathway_specs" in v:
                raise NotImplementedError("More Work Needed!")
            if k not in param_lookup:
                imodel = AbstractModel.init_model(v)
                param_lookup[k] = imodel.param_names
                shorthand_lookup[k] = imodel.shorthand
            for p in param_lookup[k]:
                if val := v.pop(p, None):
                    defaults[f"{shorthand_lookup[k]}_{p}"] = val
            models[k] = v
        else:
            fct_dict[k] = v

    if defaults:
        present = df.columns
        for col in defaults:
            if col not in present:
                df[col] = np.nan
        for col in present:
            if col not in defaults:
                if col in ["src_type", "dst_type"]:
                    defaults[col] = "*"
                else:
                    defaults[col] = np.nan
        df = pd.concat([pd.DataFrame([defaults]), df], ignore_index=True)

    if src := fct_dict.pop("sel_src", None):
        df["src_hemisphere"] = src["hemisphere"]
        df["src_region"] = src["region"]
    if dst := fct_dict.pop("sel_dest", None):
        df["dst_hemisphere"] = dst["hemisphere"]
        df["dst_region"] = dst["region"]
    return df


df = pd.concat([transform_function(d) for d in original_config["manip"]["fcts"]])

original_config["manip"]["fcts"] = [
    json.loads(cfg) for cfg in set(json.dumps(cfg) for cfg in original_config["manip"]["fcts"])
]

with open(f"{args.output_basename}.json", "w") as fd:
    json.dump(original_config, fd)

# df.to_csv("test.csv", index=False)
df["src_type"] = df["src_type"].astype("category")
df["dst_type"] = df["dst_type"].astype("category")
df["src_hemisphere"] = df["src_hemisphere"].astype("category")
df["dst_hemisphere"] = df["dst_hemisphere"].astype("category")
df["src_region"] = df["src_region"].astype("category")
df["dst_region"] = df["dst_region"].astype("category")

df = (
    df
    .set_index(["src_hemisphere", "src_region", "dst_hemisphere", "dst_region"])
    .sort_index()
)

df.to_parquet(f"{args.output_basename}.parquet")
