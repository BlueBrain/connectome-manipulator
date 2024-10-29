# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL

"""Definition and mapping of model types to classes"""

from abc import ABCMeta, abstractmethod
import os
import sys

import json
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.interpolate import interpn
from scipy.optimize import fsolve
from scipy.sparse import csc_matrix
from scipy.spatial import distance_matrix
from scipy.stats import truncnorm, norm, gamma, poisson

import connectome_manipulator
from connectome_manipulator import log

P_TH_ABS = 0.01  # Absolute probability threshold
P_TH_REL = 0.1  # Relative probability threshold

N_SYN_PER_CONN_NAME = "n_syn_per_conn"


class AbstractModel(metaclass=ABCMeta):
    """Abstract base class for different types of models."""

    ###########################################################################
    # Abstract properties/methods to be defined in specific model subclasses
    @property
    @abstractmethod
    def param_names(self):
        """Names of model parameters which are part of the model."""

    @property
    @abstractmethod
    def param_defaults(self):
        """Default parameter values as dict. Can be empty if no default values are provided."""

    @property
    @abstractmethod
    def data_names(self):
        """Names of model data frames which are part of the model."""

    @property
    @abstractmethod
    def input_names(self):
        """Names of model inputs which are part of the model."""

    @abstractmethod
    def get_model_output(self, **kwargs):
        """Abstract method for returning model output given its model inputs."""

    @abstractmethod
    def __str__(self):
        """Abstract method for returning a model string describing the model."""

    #
    ###########################################################################

    @staticmethod
    def init_model(model_spec):
        """Model creation from file (within dict) or dict."""
        log.log_assert(
            isinstance(model_spec, dict), "Model specification must be provided as dict!"
        )
        if "file" in model_spec:  # Init from file
            log.log_assert(
                len(model_spec) == 1, "Inconsistent model specification!"
            )  # If file provided, no other entries allowed
            return AbstractModel.model_from_file(model_spec["file"])
        else:  # Init directly from dict
            return AbstractModel.model_from_dict(model_spec)

    @staticmethod
    def model_from_file(model_file):
        """Wrapper function to load model object from file."""
        log.log_assert(os.path.exists(model_file), f'Model file "{model_file}" not found!')
        log.log_assert(
            os.path.splitext(model_file)[1] == ".json",
            'Model file must be of type ".json"!',
        )
        with open(model_file, "r") as f:
            model_dict = json.loads(f.read())
        data_keys = model_dict.pop("data_keys", [])
        data_dict = AbstractModel.load_data_dict(model_file, data_keys)

        return AbstractModel.model_from_dict(model_dict, data_dict)

    @staticmethod
    def load_data_dict(model_file, data_keys):
        """Load supplementary model data (if any) from .h5 data file. [same name and folder as .json file]"""
        if len(data_keys) > 0:
            data_file = os.path.splitext(model_file)[0] + ".h5"
            log.log_assert(os.path.exists(data_file), f'Data file "{data_file}" missing!')
            data_dict = {key: pd.read_hdf(data_file, key) for key in data_keys}
        else:
            data_dict = {}
        return data_dict

    @staticmethod
    def model_from_dict(model_dict, data_dict=None):
        """Wrapper function to create model object from dict."""
        log.log_assert("model" in model_dict, "Model type not found!")
        if data_dict is None:
            data_dict = {}

        model_dict = model_dict.copy()
        model_type = model_dict.pop("model")
        model_class = getattr(sys.modules[__class__.__module__], model_type)  # Get model subclass

        model = model_class(**model_dict, **data_dict)  # Initialize model object

        return model

    def __init__(self, **kwargs):
        """Model initialization from kwargs."""
        self.init_params(kwargs)
        self.init_data(kwargs)

        unused_params = [
            k for k in kwargs if not k.startswith("__")
        ]  # Unused paramters, excluding meta data ('__<name>') that may be included in file
        if len(unused_params) > 0:
            log.warning(f"Unused parameter(s): {set(unused_params)}!")

    def init_params(self, model_dict):
        """Initialize model parameters from dict (removing used keys from dict)."""
        log.log_assert(
            all(p in model_dict or p in self.param_defaults for p in self.param_names),
            f"Missing parameters for model initialization! Must contain initialization for {set(self.param_names) - set(self.param_defaults)}.",
        )
        for p in self.param_names:
            if p in model_dict:
                val = model_dict.pop(p)
            else:  # Use value from defaults
                val = self.param_defaults[p]
            setattr(self, p, val)

    def init_data(self, data_dict):
        """Initialize data frames with supplementary model data from dict (removing used keys from dict)."""
        log.log_assert(
            all(d in data_dict for d in self.data_names),
            f"Missing data for model initialization! Must contain initialization for {set(self.data_names)}.",
        )
        log.log_assert(
            np.all(isinstance(data_dict[d], pd.DataFrame) for d in self.data_names),
            "Model data must be Pandas dataframes!",
        )
        for d in self.data_names:
            setattr(self, d, data_dict.pop(d))

    def apply(self, **kwargs):
        """Main method for applying model, i.e., returning model output given its model inputs.

        [Calls get_model_output() which must be implemented in specific model subclass!]
        """
        #         log.log_assert(
        #             all(inp in kwargs for inp in self.input_names),
        #             f"Missing model inputs! Must contain input values for {set(self.input_names)}.",
        #         )
        inp_dict = {inp: kwargs.pop(inp) for inp in self.input_names}
        if len(kwargs) > 0:
            log.debug(f"Unused input(s): {set(kwargs.keys())}!")
        return self.get_model_output(**inp_dict)

    def get_param_dict(self):
        """Return model parameters as dict."""
        return {p: getattr(self, p) for p in self.param_names}

    def get_data_dict(self):
        """Return data frames with supplementary model data as dict."""
        return {d: getattr(self, d) for d in self.data_names}

    def save_model(self, model_path, model_name):
        """Save model to file: Model dict as .json, model data (if any) as supplementary .h5 data file."""
        model_dict = self.get_param_dict()
        model_file = os.path.join(model_path, model_name + ".json")

        # Save supplementary model data (if any) to .h5 data file
        data_dict = self.get_data_dict()
        log.log_assert(
            np.all(isinstance(v, pd.DataFrame) for k, v in data_dict.items()),
            "Model data must be Pandas dataframes!",
        )
        data_file = os.path.splitext(model_file)[0] + ".h5"
        for idx, (key, df) in enumerate(data_dict.items()):
            df.to_hdf(data_file, key, mode="w" if idx == 0 else "a")
        model_dict["data_keys"] = list(data_dict.keys())

        # Save model dict to .json file
        model_dict["model"] = self.__class__.__name__
        model_dict["__version_info__"] = {
            "connectome_manipulator": connectome_manipulator.__version__,
            "python": sys.version,
            "pandas": pd.__version__,
        }
        with open(model_file, "w") as f:
            f.write(json.dumps(model_dict, indent=2))


# MODEL TEMPLATE #
# class TemplateModel(AbstractModel):
#     """ <Template> model:
#         -Model details...
#     """
#
#     # Names of model inputs, parameters and data frames which are part of this model
#     param_names = [...]
#     param_defaults = {...}
#     data_names = [...]
#     input_names = [...]
#
#     def __init__(self, **kwargs):
#         """Model initialization."""
#         super().__init__(**kwargs)
#
#         # Check parameters
#         log.log_assert(...)
#
#     # <Additional access methods, if needed>
#     ...
#
#     def get_model_output(self, **kwargs):
#         """Description..."""
#         # MUST BE IMPLEMENTED
#         return ...
#
#     def __str__(self):
#         """Return model string describing the model."""
#         model_str = f'{self.__class__.__name__}\n'
#         model_str = model_str + ...
#         return model_str


class PathwayModel(AbstractModel, metaclass=ABCMeta):
    """Abstract model base class for storing model properties per pathway (i.e., for pairs of m-types):

    - Different property values for specific pathways
    - Default property values for any other pathways or properties not specified
    - Actual functionaliy (i.e., how to use these properties) to be implemented in derived class
    """

    # Names of base model inputs and parameters that are part of this model
    pathway_input_names = ["src_type", "tgt_type"]

    def init_params(self, _):
        """Overwritten to avoid setting class variables."""

    def get_param_dict(self):
        """Return model default parameters as dict."""
        return {p: getattr(self, p)[-1, -1] for p in self.param_names}

    @property
    def shorthand(self):
        """Pathway column prefix for this model."""
        return self.__class__.__name__.replace("Model", "").lower()

    def __init__(self, src_type_map=None, tgt_type_map=None, pathway_specs=None, **kwargs):
        """Model initialization."""
        self.property_names = self.param_names
        self.input_names = self.pathway_input_names + self.input_names

        if pathway_specs is not None and (src_type_map is None or tgt_type_map is None):
            raise ValueError("Need to specify both type maps when using pathway specs.")

        self.default_enforce = False
        if src_type_map:
            self.src_type_map = src_type_map
        else:
            self.default_enforce = True
            self.src_type_map = {}
        if tgt_type_map:
            self.tgt_type_map = tgt_type_map
        else:
            self.default_enforce = True
            self.tgt_type_map = {}

        self.default_types = (len(self.src_type_map), len(self.tgt_type_map))
        shape = tuple(s + 1 for s in self.default_types)
        for param in self.param_names:
            if param in kwargs:
                default = kwargs.pop(param)
            else:
                default = self.param_defaults[param]
            matrix = np.full(shape, default)
            colname = "_".join([self.shorthand, param])
            if pathway_specs is not None and colname in pathway_specs.columns:
                col = pathway_specs[colname]
                if ("*", "*") in col:
                    if not np.isnan(default := col.pop(("*", "*"))):
                        matrix[:] = default
                for (src, dst), val in col.items():
                    if not np.isnan(val):
                        i = self.src_type_map[src]
                        j = self.tgt_type_map[dst]
                        matrix[i][j] = val
            setattr(self, param, matrix)

        super().__init__(**kwargs)

    def apply(self, **kwargs):
        """Apply the model.

        Makes sure that the source and destination type values are set according to the
        inputs to the model: if no pathways have been given at construction time, the
        application will always use the default values.
        """
        for way, default in zip(("src", "tgt"), self.default_types):
            way_type = f"{way}_type"
            way_pos = f"{way}_pos"
            if self.default_enforce or way_type not in kwargs:
                log.debug(f"Using default values for {way_type} in {self}")
                if way_type in kwargs:
                    kwargs[way_type] = np.full_like(kwargs[way_type], default)
                elif way_pos in kwargs:
                    kwargs[way_type] = np.full((kwargs[way_pos].shape[0],), default)
                else:
                    kwargs[way_type] = [default]
            elif (types := kwargs.get(way_type, None)) is not None:
                if np.isscalar(types):
                    if way_pos in kwargs:
                        types = np.full((kwargs[way_pos].shape[0],), types)
                    else:
                        types = np.array([types])
                    kwargs[way_type] = types
                kind = np.array(types).dtype.kind
                if kind in ("i", "u"):
                    # No mapping needed for integer indices
                    pass
                elif kind == "U":
                    mapping = getattr(self, f"{way}_type_map")
                    kwargs[way_type] = np.array([mapping[t] for t in types])
                else:
                    raise ValueError(f"Cannot have types '{kind}' for '{way_type}'")
        return super().apply(**kwargs)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str += f"  Model properties: {self.property_names}\n"
        if self.default_types[0] > 0 or self.default_types[1] > 0:
            counts = "×".join([str(cnt) for cnt in self.default_types if cnt > 0])
            model_str += f"  Property values for {counts} pathways\n"
        model_str += f"  Default: {self.get_param_dict()}"
        return model_str


class NSynConnModel(PathwayModel):
    """Model for number of synapses per connection for pairs of m-types [generative model]:

    - Synapses per connection drawn from gamma distribution with given mean/std
    - Integer number of synapses larger or equal to one will be returned
    - Different distribution attributes for specific pathways
    - Default distribution attributes for any other pathways not specified
    """

    # Names of model inputs, parameters and data frames which are part of this model
    # (other than the ones inherited from PathwayModel class)
    param_names = ["mean", "std"]
    param_defaults = {"mean": 3.0, "std": 1.5}
    data_names = []
    input_names = []

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        log.log_assert(np.all(self.mean > 0.0), "Mean must be larger than zero!")
        log.log_assert(np.all(self.std >= 0.0), "Std cannot be negative!")

    def get_model_output(self, src_type, tgt_type):  # pylint: disable=arguments-differ
        """Draw #syn/conn value for one connection between src_type and tgt_type [seeded through numpy]."""
        # Get distribution attribute values
        distr_mean = self.mean[src_type, tgt_type]
        distr_std = self.std[src_type, tgt_type]

        nsyn = distr_mean
        # Draw number of synapses
        if np.any(sel := distr_std > 0.0):
            nsyn[sel] = np.random.gamma(
                shape=distr_mean[sel] ** 2 / distr_std[sel] ** 2,
                scale=distr_std[sel] ** 2 / distr_mean[sel],
            )

        # Convert type
        nsyn = np.round(np.maximum(1, nsyn)).astype(int)

        return nsyn


class LinDelayModel(PathwayModel):
    """Linear distance-dependent delay model for pairs of m-types [generative model]:

    - Delay mean: delay_mean_coeff_b * distance + delay_mean_coeff_a (linear)
    - Delay std: delay_std (constant)
    - Delay min: delay_min (constant)
    - Different delay attributes for specific pathways
    - Default delay attributes for any other pathways not specified
    """

    # Names of model inputs, parameters and data frames which are part of this model
    # (other than the ones inherited from PathwayModel class)
    param_names = ["delay_mean_coeff_a", "delay_mean_coeff_b", "delay_std", "delay_min"]
    param_defaults = {
        "delay_mean_coeff_a": 0.75,
        "delay_mean_coeff_b": 0.003,
        "delay_std": 0.5,
        "delay_min": 0.2,
    }
    data_names = []
    input_names = ["distance"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        log.log_assert(np.all(self.delay_std >= 0.0), "Delay std cannot be negative!")
        log.log_assert(np.all(self.delay_min >= 0.0), "Delay min cannot be negative!")

    def get_mean(self, src_type, tgt_type, distance):
        """Returns mean delay as function of distance."""
        return (
            self.delay_mean_coeff_b[src_type, tgt_type] * np.array(distance)
            + self.delay_mean_coeff_a[src_type, tgt_type]
        )

    def get_std(self, src_type, tgt_type, distance):
        """Returns delay std as function of distance."""
        return np.full_like(
            distance, self.delay_std[src_type, tgt_type], dtype=self.delay_std.dtype
        )

    def get_min(self, src_type, tgt_type, distance):
        """Returns min delay as function of distance."""
        return np.full_like(
            distance, self.delay_min[src_type, tgt_type], dtype=self.delay_min.dtype
        )

    def get_model_output(self, src_type, tgt_type, distance):  # pylint: disable=arguments-differ
        """Draw distance-dependent delay values from truncated normal distribution [seeded through numpy]."""
        d_mean = self.get_mean(src_type, tgt_type, distance)
        d_std = self.get_std(src_type, tgt_type, distance)
        d_min = self.get_min(src_type, tgt_type, distance)
        if all(d_std > 0.0):
            return truncnorm.rvs(a=(d_min - d_mean) / d_std, b=np.inf, loc=d_mean, scale=d_std)
        else:
            return np.maximum(d_mean, d_min)  # Deterministic


class PosMapModel(AbstractModel):
    """Position mapping model, mapping one coordinate system to another for a given set of neurons:

    - Mapped neuron position: pos_table.loc[gids] (lookup-table)
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = []
    param_defaults = {}
    data_names = ["pos_table"]
    input_names = ["gids"]

    def get_gids(self):
        """Return GIDs that are mapped within this model."""
        return self.pos_table.index.values

    def get_coord_names(self):
        """Return coordinate names of this model."""
        return list(self.pos_table.columns)

    def get_model_output(self, gids):  # pylint: disable=arguments-differ
        """Return (mapped) neuron positions for a given set of GIDs."""
        return self.pos_table.loc[gids].to_numpy()

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  Size: {len(self.get_gids())} GIDs ({np.min(self.get_gids())}..{np.max(self.get_gids())})\n"
        )
        model_str = (
            model_str
            + f'  Outputs: {len(self.get_coord_names())} ({", ".join(self.get_coord_names())})\n'
        )
        model_str = (
            model_str
            + "  Range: "
            + ", ".join(
                [
                    f"{k}: {self.pos_table[k].min():.1f}..{self.pos_table[k].max():.1f}"
                    for k in self.get_coord_names()
                ]
            )
        )
        return model_str


class ConnProbModel(PathwayModel):
    """Generic connection probability model:

    - Implements both Erdos-Renyi and distance-dependent models
    - Returns connection probability for given source/target neuron positions and m-types
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["order", "coeff_a", "coeff_b"]
    param_defaults = {"order": 1, "coeff_a": 0.0, "coeff_b": np.nan}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def get_model_output(
        self, src_type, tgt_type, src_pos, tgt_pos
    ):  # pylint: disable=arguments-differ
        """Return pathway-specific connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        if np.isscalar(src_type):
            src_type = [src_type] * src_pos.shape[0]
        if np.isscalar(tgt_type):
            tgt_type = [tgt_type] * tgt_pos.shape[0]

        p_mat = np.zeros((len(src_type), len(tgt_type)))
        for ti, tt in enumerate(tgt_type):
            # tt is the first index in the Pandas DataFrame
            order = self.order[src_type, tt]
            coeff_a = self.coeff_a[src_type, tt]
            p_mat[:, ti] = coeff_a

            # override higher order values that are more expensive to calculate
            sel = order == 2
            if np.any(sel):
                coeff_a = coeff_a[sel]
                coeff_b = self.coeff_b[src_type, tt][sel]
                distance = np.sqrt(np.sum((src_pos[sel] - tgt_pos[ti]) ** 2, axis=1))
                p_mat[:, ti][sel] = coeff_a * np.exp(-coeff_b * distance)
        return p_mat


class ConnPropsModel(AbstractModel):
    """Connection/synapse properties model for pairs of m-types [generative model]:

    - Connection/synapse property values drawn from given distributions

    NOTE: 'shared_within' flag is used to indicate that same property values are used
          for all synapses within the same connection. Otherwise, property values are
          drawn for all synapses independently from same distribution.

    NOTE: Correlations between properties can be specified using a 'prop_cov' dict with
          'props' (list of correlated property names) and 'cov' (covariance matrices
          by pathway)
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["src_types", "tgt_types", "prop_stats", "prop_cov"]
    param_defaults = {"prop_cov": {}}
    data_names = []
    input_names = ["src_type", "tgt_type", "n_syn"]
    # Notes:
    # n_syn is optional; it must be provided if no N_SYN_PER_CONN_NAME property is specified
    # If N_SYN_PER_CONN_NAME property is specified but n_syn is provided, n_syn will be used

    # Required attributes for given distributions
    distribution_attributes = {
        "constant": ["mean"],
        "normal": ["mean", "std"],
        "truncnorm": ["norm_loc", "norm_scale", "min", "max"],
        "gamma": ["mean", "std"],
        "poisson": ["mean"],
        "ztpoisson": ["mean"],
        "discrete": ["val", "p"],
        "zero": [],
    }
    # Notes:
    # constant: "mean" corresponds to the constant value
    # truncnorm: "norm_loc"/"norm_scale" are center location (mean) and scale (standard deviation) of the underlying (non-truncated) normal distribution
    # ztpoisson: Zero-truncated poisson distribution
    # discrete: "val"/"p" are discrete values and probabilities of the discrete distribution
    # zero: Empty distribution always returning zero, to model unused parameters

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        def dict_conv(data):
            """Recursively convert numpy to basic data types, to have a clean JSON file"""
            if isinstance(data, dict):
                return {k: dict_conv(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [dict_conv(d) for d in data]
            elif hasattr(data, "tolist"):  # Convert numpy types
                return data.tolist()
            else:
                return data

        # Check & convert parameters
        # pylint: disable=access-member-before-definition
        log.log_assert(isinstance(self.prop_stats, dict), '"prop_stats" dictionary required!')
        self.prop_stats = dict_conv(self.prop_stats)  # Convert dict to basic data types
        self.prop_names = list(self.prop_stats.keys())
        if N_SYN_PER_CONN_NAME in self.prop_names:
            self.has_nsynconn = True
        else:
            self.has_nsynconn = False
        log.log_assert(
            all(isinstance(self.prop_stats[p], dict) for p in self.prop_names),
            "Property statistics dictionary required!",
        )
        log.log_assert(
            all(
                all(np.isin(self.src_types, list(self.prop_stats[p].keys())))
                for p in self.prop_names
            ),
            f"Source type statistics missing! self.src_types:{self.src_types} not in {self.prop_names}",
        )
        log.log_assert(
            all(
                all(isinstance(self.prop_stats[p][src], dict) for p in self.prop_names)
                for src in self.src_types
            ),
            "Property statistics dictionary required!",
        )
        log.log_assert(
            all(
                all(
                    all(np.isin(self.tgt_types, list(self.prop_stats[p][src].keys())))
                    for p in self.prop_names
                )
                for src in self.src_types
            ),
            "Target type statistics missing!",
        )
        log.log_assert(
            all(
                all(
                    all("type" in self.prop_stats[p][src][tgt].keys() for p in self.prop_names)
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            "Distribution type missing!",
        )
        log.log_assert(
            all(
                all(
                    all(
                        all(
                            np.isin(
                                self.distribution_attributes[self.prop_stats[p][src][tgt]["type"]],
                                list(self.prop_stats[p][src][tgt].keys()),
                            )
                        )
                        for p in self.prop_names
                    )
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            f"Distribution attributes missing (required: {self.distribution_attributes})!",
        )
        log.log_assert(
            all(
                all(
                    all(
                        all(
                            len(self.prop_stats[p][src][tgt][a]) > 0
                            for a in self.distribution_attributes[
                                self.prop_stats[p][src][tgt]["type"]
                            ]
                            if hasattr(self.prop_stats[p][src][tgt][a], "__iter__")
                        )
                        for p in self.prop_names
                    )
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            f"Distribution attribute(s) empty (required: {self.distribution_attributes})!",
        )
        log.log_assert(
            all(
                all(
                    all(
                        np.isclose(np.sum(self.prop_stats[p][src][tgt]["p"]), 1.0)
                        for p in self.prop_names
                        if "p" in self.prop_stats[p][src][tgt].keys()
                    )
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            'Probability attribute "p" does not sum to 1.0!',
        )
        log.log_assert(
            all(
                all(
                    all(
                        len(self.prop_stats[p][src][tgt]["p"])
                        == len(self.prop_stats[p][src][tgt]["val"])
                        for p in self.prop_names
                        if np.all(np.isin(["p", "val"], list(self.prop_stats[p][src][tgt].keys())))
                    )
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            'Probability attribute "p" does not match length of corresponding "val"!',
        )
        log.log_assert(
            all(
                all(
                    all(
                        self.prop_stats[p][src][tgt]["lower_bound"]
                        <= self.prop_stats[p][src][tgt]["upper_bound"]
                        for p in self.prop_names
                        if "lower_bound" in self.prop_stats[p][src][tgt].keys()
                        and "upper_bound" in self.prop_stats[p][src][tgt].keys()
                    )
                    for src in self.src_types
                )
                for tgt in self.tgt_types
            ),
            "Data bounds error!",
        )
        log.log_assert(isinstance(self.prop_cov, dict), '"prop_cov" dictionary required!')
        self.prop_cov = dict_conv(self.prop_cov)  # Convert dict to basic data types
        if len(self.prop_cov) > 0:
            log.log_assert(
                "props" in self.prop_cov and "cov" in self.prop_cov,
                'Property COV dict must contain "props" and "cov" keys!',
            )
            log.log_assert(len(self.prop_cov["props"]) > 0, 'Property COV "props" names missing!')
            log.log_assert(
                all(np.isin(self.prop_cov["props"], self.prop_names)),
                'Property COV "props" name error!',
            )
            log.log_assert(
                N_SYN_PER_CONN_NAME not in self.prop_cov["props"],
                f'Property COV for "{N_SYN_PER_CONN_NAME}" not supported!',
            )
            for src in self.src_types:
                log.log_assert(
                    src in self.prop_cov["cov"],
                    f'Property COV source mtype "{src}" missing!',
                )
                for tgt in self.tgt_types:
                    log.log_assert(
                        tgt in self.prop_cov["cov"][src],
                        f'Property COV target mtype "{tgt}" for source "{src}" missing!',
                    )
                    log.log_assert(
                        all(
                            np.array(np.array(self.prop_cov["cov"][src][tgt]).shape)
                            == len(self.prop_cov["props"])
                        ),
                        "Propert COV matrix size mismatch!",
                    )
                    log.log_assert(
                        np.all(np.diag(self.prop_cov["cov"][src][tgt]) == 1.0),
                        "Property COV matrices must contain ones in the diagonal!",
                    )
                    log.log_assert(
                        np.array_equal(
                            self.prop_cov["cov"][src][tgt],
                            np.array(self.prop_cov["cov"][src][tgt]).T,
                        ),
                        "Property COV matrices must be symmetric!",
                    )
            self.prop_cov_names = self.prop_cov["props"]
            self.prop_cov_mat = self.prop_cov["cov"]
        else:
            self.prop_cov_names = []
            self.prop_cov_mat = None

    def get_prop_names(self):
        """Return list of connection/synapse property names."""
        return self.prop_names

    def get_prop_cov_names(self):
        """Return list of correlated property names."""
        return self.prop_cov_names

    def get_prop_cov_mat(self, src_type, tgt_type):
        """Return covariance matrix of correlated properties."""
        return np.array(self.prop_cov_mat[src_type][tgt_type])

    def get_src_types(self):
        """Return list source (pre-synaptic) m-types."""
        return self.src_types

    def get_tgt_types(self):
        """Return list target (post-synaptic) m-types."""
        return self.tgt_types

    def get_distr_props(self, prop_name, src_type, tgt_type):
        """Return distribution type & properties (mean, std, ...)."""
        return self.prop_stats[prop_name][src_type][tgt_type]

    @staticmethod
    def zero_truncated_poisson(lam, size=1):
        """Draw value(s) from zero-truncated poisson distribution."""
        u = np.random.uniform(np.exp(-lam), 1, size=size)
        t = -np.log(u)
        return 1 + np.random.poisson(lam - t)

    @staticmethod
    def compute_ztpoisson_lambda(mean):
        """Compute lambda of zero-truncated poission distribution corresponding to given mean (numerically)."""

        def fct(lam, mn):
            return lam / (1 - np.exp(-lam)) - mn

        lam = fsolve(fct, mean, mean)[0]
        lam = np.maximum(lam, 0.0)
        return lam

    @staticmethod
    def draw_from_distribution(distr_spec, size=1):
        """Draw value(s) from given distribution"""
        distr_type = distr_spec.get("type")
        if distr_type == "constant":
            distr_val = distr_spec.get("mean")
            log.log_assert(
                distr_val is not None,
                "Distribution parameter missing (required: mean)!",
            )
            drawn_values = np.full(size, distr_val)
        elif distr_type == "normal":
            distr_mean = distr_spec.get("mean")
            distr_std = distr_spec.get("std")
            log.log_assert(
                distr_mean is not None and distr_std is not None,
                "Distribution parameter missing (required: mean/std)!",
            )
            drawn_values = np.random.normal(loc=distr_mean, scale=distr_std, size=size)
        elif distr_type == "truncnorm":
            distr_loc = distr_spec.get("norm_loc")
            distr_scale = distr_spec.get("norm_scale")
            distr_min = distr_spec.get("min")
            distr_max = distr_spec.get("max")
            log.log_assert(
                distr_loc is not None
                and distr_scale is not None
                and distr_min is not None
                and distr_max is not None,
                "Distribution parameters missing (required: norm_loc/norm_scale/min/max)!",
            )
            log.log_assert(distr_min <= distr_max, "Range error (truncnorm)!")
            if distr_scale > 0.0:
                drawn_values = truncnorm(
                    a=(distr_min - distr_loc) / distr_scale,
                    b=(distr_max - distr_loc) / distr_scale,
                    loc=distr_loc,
                    scale=distr_scale,
                ).rvs(size=size)
            else:
                drawn_values = np.clip(np.full(size, distr_loc), distr_min, distr_max)

        elif distr_type == "gamma":
            distr_mean = distr_spec.get("mean")
            distr_std = distr_spec.get("std")
            log.log_assert(
                distr_mean is not None and distr_std is not None,
                "Distribution parameter missing (required: mean/std)!",
            )
            log.log_assert(distr_mean > 0.0 and distr_std >= 0.0, "Range error (gamma)!")
            if distr_std > 0.0:
                drawn_values = np.random.gamma(
                    shape=distr_mean**2 / distr_std**2,
                    scale=distr_std**2 / distr_mean,
                    size=size,
                )
            else:
                drawn_values = np.full(size, distr_mean)
        elif distr_type == "poisson":
            distr_mean = distr_spec.get("mean")
            log.log_assert(
                distr_mean is not None, "Distribution parameter missing (required: mean)!"
            )
            log.log_assert(distr_mean >= 0.0, "Range error (poisson)!")
            drawn_values = np.random.poisson(lam=distr_mean, size=size)
        elif distr_type == "ztpoisson":
            distr_mean = distr_spec.get("mean")
            log.log_assert(
                distr_mean is not None, "Distribution parameter missing (required: mean)!"
            )
            log.log_assert(distr_mean >= 1.0, "Range error (zero-truncated poisson)!")

            # Determine lambda corresponding to given mean...
            lam = ConnPropsModel.compute_ztpoisson_lambda(distr_mean)
            # ...and draw values from zero-truncated poisson distribition
            drawn_values = ConnPropsModel.zero_truncated_poisson(lam=lam, size=size)
        elif distr_type == "discrete":
            distr_val = distr_spec.get("val")
            distr_p = distr_spec.get("p")
            log.log_assert(
                distr_val is not None
                and distr_p is not None
                and not np.isscalar(distr_val)
                and not np.isscalar(distr_p)
                and len(distr_val) == len(distr_p),
                "Distribution parameters error or missing (required: list-like val/p of same length)!",
            )
            drawn_values = np.random.choice(distr_val, size=size, p=distr_p)
        elif distr_type == "zero":
            drawn_values = np.full(size, 0.0)
        else:
            log.log_assert(False, f'Distribution type "{distr_type}" not supported!')
        return drawn_values  # pylint: disable=E0606

    def draw(self, prop_name, src_type, tgt_type, size=1):
        """Draw value(s) for given property name of a single connection

        (or multiple connections, if prop_name==N_SYN_PER_CONN_NAME)
        """
        stats_dict = self.prop_stats.get(prop_name)[src_type][tgt_type]
        if prop_name == N_SYN_PER_CONN_NAME:  # Draw <size> N_SYN_PER_CONN_NAME value(s)
            drawn_values = np.maximum(
                np.round(self.draw_from_distribution(stats_dict, size)).astype(int),
                1,
            )  # At least one synapse/connection, otherwise no connection!!
        else:
            shared_within = stats_dict.get("shared_within", True)
            if shared_within:  # Same property value for all synapses within connection
                val = self.draw_from_distribution(stats_dict, 1)  # Draw a single value...
                drawn_values = np.full(size, val)  # ...and reuse in all synapses
            else:  # Redraw property values independently for all synapses within connection
                drawn_values = self.draw_from_distribution(stats_dict, size)

        # Apply upper/lower bounds (optional)
        lower_bound = stats_dict.get("lower_bound")
        upper_bound = stats_dict.get("upper_bound")
        if lower_bound is not None:
            drawn_values = np.maximum(drawn_values, lower_bound)
        if upper_bound is not None:
            drawn_values = np.minimum(drawn_values, upper_bound)

        # Set data type (optional)
        data_type = stats_dict.get("dtype")
        if data_type is not None:
            if data_type == "int":
                drawn_values = np.round(drawn_values)
            drawn_values = drawn_values.astype(data_type)

        return drawn_values

    @staticmethod
    def remap_distribution(values, distr_spec):
        """Re-map values from standard normal disrtibution to another distributions."""
        sf_values = 1 - norm.cdf(values)
        remapped_values = None
        distr_type = distr_spec.get("type")
        if distr_type == "normal":
            distr_mean = distr_spec.get("mean")
            distr_std = distr_spec.get("std")
            remapped_values = norm.isf(sf_values, loc=distr_mean, scale=distr_std)
        elif distr_type == "truncnorm":
            distr_loc = distr_spec.get("norm_loc")
            distr_scale = distr_spec.get("norm_scale")
            distr_min = distr_spec.get("min")
            distr_max = distr_spec.get("max")
            log.log_assert(
                distr_scale > 0.0, 'Truncnorm remapping error: "norm_scale" cannot be zero!'
            )
            a = (distr_min - distr_loc) / distr_scale
            b = (distr_max - distr_loc) / distr_scale
            remapped_values = truncnorm.isf(sf_values, a=a, b=b, loc=distr_loc, scale=distr_scale)
        elif distr_type == "gamma":
            distr_mean = distr_spec.get("mean")
            distr_std = distr_spec.get("std")
            log.log_assert(
                distr_mean > 0.0 and distr_std > 0.0,
                'Gamma remapping error: "mean" and "std" cannot be zero!',
            )
            shape = distr_mean**2 / distr_std**2
            scale = distr_std**2 / distr_mean
            remapped_values = gamma.isf(sf_values, a=shape, scale=scale)
        elif distr_type == "poisson":
            distr_mean = distr_spec.get("mean")
            remapped_values = poisson.isf(sf_values, mu=distr_mean)
        else:
            log.log_assert(False, f'Remapping not supported for distribution type "{distr_type}"!')

        # Apply upper/lower bounds (optional)
        lower_bound = distr_spec.get("lower_bound")
        upper_bound = distr_spec.get("upper_bound")
        if lower_bound is not None:
            remapped_values = np.maximum(remapped_values, lower_bound)
        if upper_bound is not None:
            remapped_values = np.minimum(remapped_values, upper_bound)

        # Set data type (optional)
        data_type = distr_spec.get("dtype")
        if data_type is not None:
            if data_type == "int":
                remapped_values = np.round(remapped_values)
            remapped_values = remapped_values.astype(data_type)

        return remapped_values

    def draw_cov(self, src_type, tgt_type, size=1):
        """Draw correlated property values of a single connection.

        (As in Chindemi et al. (2022) "A calcium-based plasticity model for
        predicting long-term potentiation and depression in the neocortex")
        """
        if self.prop_cov_mat is None:
            log.warning("No correlated properties!")
            return np.zeros((size, 0))

        stats_dicts = [self.prop_stats.get(p)[src_type][tgt_type] for p in self.prop_cov_names]
        shared_within = [_sdict.get("shared_within", True) for _sdict in stats_dicts]
        log.log_assert(
            np.all(np.array(shared_within) == shared_within[0]),
            'Inconsistent "shared_within" value among correlated properties!',
        )
        shared_within = shared_within[0]

        # Draw from multivariate Gaussian
        if shared_within:  # Same property value for all synapses within connection
            val = np.random.multivariate_normal(
                np.zeros(len(self.prop_cov_names)), self.prop_cov_mat[src_type][tgt_type], size=1
            )
            drawn_values = np.repeat(val, size, axis=0)  # Reuse same value in all synapses
        else:  # Redraw property values independently for all synapses within connection
            drawn_values = np.random.multivariate_normal(
                np.zeros(len(self.prop_cov_names)), self.prop_cov_mat[src_type][tgt_type], size=size
            )

        # Re-map Gaussian to respective marginal distributions
        remapped_values = [
            self.remap_distribution(drawn_values[:, [_i]], _sdict)
            for _i, _sdict in enumerate(stats_dicts)
        ]

        return remapped_values

    def get_model_output(self, src_type, tgt_type, n_syn=None):  # pylint: disable=arguments-differ
        """Draw property values for one connection between src_type and tgt_type, returning a dataframe [seeded through numpy]."""
        syn_props = [
            p for p in self.prop_names if p != N_SYN_PER_CONN_NAME and p not in self.prop_cov_names
        ]
        if n_syn is None:
            log.log_assert(self.has_nsynconn, f'"{N_SYN_PER_CONN_NAME}" missing')
            n_syn = self.draw(N_SYN_PER_CONN_NAME, src_type, tgt_type, 1)[0]
        else:
            log.log_assert(n_syn > 0, '"n_syn" must be at least 1!')

        df = pd.DataFrame([], index=range(n_syn), columns=syn_props)
        for p in syn_props:
            df[p] = self.draw(p, src_type, tgt_type, n_syn)

        if len(self.prop_cov_names) > 0:
            # Draw correlated property values
            corr_vals = self.draw_cov(src_type, tgt_type, n_syn)
            for i, p in enumerate(self.prop_cov_names):
                df[p] = corr_vals[i]
        return df

    def apply(self, **kwargs):
        """Apply the model. Overwrite to set default."""
        if "n_syn" not in kwargs:
            kwargs["n_syn"] = None
        return super().apply(**kwargs)

    def __str__(self):
        """Return model string describing the model."""
        distr_types = {
            p: "/".join(
                np.unique(
                    [
                        [self.prop_stats[p][src][tgt]["type"] for src in self.src_types]
                        for tgt in self.tgt_types
                    ]
                )
            )
            for p in self.prop_names
        }  # Extract distribution types
        distr_dtypes = {
            p: "/".join(
                np.unique(
                    [
                        [self.prop_stats[p][src][tgt].get("dtype", "") for src in self.src_types]
                        for tgt in self.tgt_types
                    ]
                )
            )
            for p in self.prop_names
        }  # Extract distribution data types
        model_str = f"{self.__class__.__name__}"
        model_str = model_str + f" ({'with' if self.has_nsynconn else 'w/o'} #syn/conn)\n"
        model_str = (
            model_str
            + f"  Connection/synapse property distributions between {len(self.src_types)}x{len(self.tgt_types)} M-types:\n"
        )
        model_str = (
            model_str
            + "  "
            + "; ".join(
                [
                    f"{p} <{distr_types[p]}"
                    + (f" ({distr_dtypes[p]})" if distr_dtypes[p] else "")
                    + ">"
                    for p in self.prop_names
                ]
            )
        )
        if len(self.prop_cov_names) > 0:
            model_str = model_str + f"\n  Correlated properties: {', '.join(self.prop_cov_names)}"
        return model_str


class ConnProb1stOrderModel(AbstractModel):
    """1st order connection probability model (Erdos-Renyi):

    - Returns (constant) connection probability for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["p_conn"]
    param_defaults = {"p_conn": 0.0}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        if np.all(np.isnan([getattr(self, p) for p in self.param_names])):
            log.warning("Empty/invalid model!")
        else:
            log.log_assert(
                0.0 <= self.p_conn <= 1.0, "Connection probability must be between 0 and 1!"
            )

    def get_conn_prob(self):
        """Return (constant) connection probability."""
        return self.p_conn

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (constant) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        return np.full((src_pos.shape[0], tgt_pos.shape[0]), self.get_conn_prob())

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = model_str + f"  p_conn() = {self.p_conn:.3f} (constant)"
        return model_str


class ConnProb2ndOrderExpModel(AbstractModel):
    """2nd order connection probability model (exponential distance-dependent):

    - Returns (distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["scale", "exponent"]
    param_defaults = {"scale": 0.0, "exponent": 0.0}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        if np.all(np.isnan([getattr(self, p) for p in self.param_names])):
            log.warning("Empty/invalid model!")
        else:
            log.log_assert(0.0 <= self.scale <= 1.0, '"Scale" must be between 0 and 1!')
            log.log_assert(self.exponent >= 0.0, '"Exponent" must be non-negative!')

    @staticmethod
    def exp_fct(distance, scale, exponent):
        """Distance-dependent exponential probability function."""
        return scale * np.exp(-exponent * np.array(distance))

    def get_conn_prob(self, distance):
        """Return (distance-dependent) connection probability."""
        return self.exp_fct(distance, self.scale, self.exponent)

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan  # Exclude autaptic connections
        return dist_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        return self.get_conn_prob(dist_mat)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = model_str + f"  p_conn(d) = {self.scale:.3f} * exp(-{self.exponent:.3f} * d)\n"
        model_str = model_str + "  d...distance"
        return model_str


class ConnProb2ndOrderComplexExpModel(AbstractModel):
    """2nd order connection probability model (complex exponential distance-dependent),

    based on a complex (proximal) exponential and a simple (distal) exponential function:
    - Returns (distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["prox_scale", "prox_exp", "prox_exp_pow", "dist_scale", "dist_exp"]
    param_defaults = {}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        if np.all(np.isnan([getattr(self, p) for p in self.param_names])):
            log.warning("Empty/invalid model!")
        else:
            log.log_assert(0.0 <= self.prox_scale <= 1.0, '"prox_scale" must be between 0 and 1!')
            log.log_assert(self.prox_exp >= 0.0, '"prox_exp" must be non-negative!')
            log.log_assert(self.prox_exp_pow >= 0.0, '"prox_exp_pow" must be non-negative!')
            log.log_assert(0.0 <= self.dist_scale <= 1.0, '"dist_scale" must be between 0 and 1!')
            log.log_assert(self.dist_exp >= 0.0, '"dist_exp" must be non-negative!')
            test_distance = 1000.0
            if self.exp_fct(test_distance, 1.0, self.prox_exp, self.prox_exp_pow) >= self.exp_fct(
                test_distance, 1.0, self.dist_exp, 1.0
            ):
                log.warning(
                    f"Proximal exponential decays slower than distal exponential ({self.get_param_dict()})!"
                )

    @staticmethod
    def exp_fct(distance, scale, exponent, exp_power=1.0):
        """Distance-dependent (complex) exponential probability function."""
        return scale * np.exp(-exponent * np.array(distance) ** exp_power)

    def get_conn_prob(self, distance):
        """Return (distance-dependent) connection probability."""
        return self.exp_fct(
            distance, self.prox_scale, self.prox_exp, self.prox_exp_pow
        ) + self.exp_fct(distance, self.dist_scale, self.dist_exp, 1.0)

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan  # Exclude autaptic connections
        return dist_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        return self.get_conn_prob(dist_mat)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn(d) = {self.prox_scale:.3f} * exp(-{self.prox_exp:.6f} * d^{self.prox_exp_pow:.3f}) + {self.dist_scale:.3f} * exp(-{self.dist_exp:.3f} * d)\n"
        )
        model_str = model_str + "  d...distance"
        return model_str


class ConnProb3rdOrderExpModel(AbstractModel):
    """3rd order connection probability model (bipolar exponential distance-dependent):

    - Returns (bipolar distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["scale_P", "scale_N", "exponent_P", "exponent_N", "bip_coord"]
    param_defaults = {}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        if np.all(
            np.isnan([getattr(self, p) for p in np.setdiff1d(self.param_names, "bip_coord")])
        ):
            log.warning("Empty/invalid model!")
        else:
            log.log_assert(
                0.0 <= self.scale_P <= 1.0 and 0.0 <= self.scale_N <= 1.0,
                '"Scale" must be between 0 and 1!',
            )
            log.log_assert(
                self.exponent_P >= 0.0 and self.exponent_N >= 0.0,
                '"Exponent" must be non-negative!',
            )
            log.log_assert(
                isinstance(self.bip_coord, int) and 0 <= self.bip_coord <= 2,
                'Bipolar coordinate "bip_coord" out of range!',
            )

    @staticmethod
    def exp_fct(distance, scale, exponent):
        """Distance-dependent exponential probability function."""
        return scale * np.exp(-exponent * np.array(distance))

    def get_conn_prob(self, distance, bip):
        """Return (bipolar distance-dependent) connection probability."""
        p_conn_N = self.exp_fct(distance, self.scale_N, self.exponent_N)
        p_conn_P = self.exp_fct(distance, self.scale_P, self.exponent_P)
        p_conn = np.select(
            [np.array(bip) < 0.0, np.array(bip) > 0.0],
            [p_conn_N, p_conn_P],
            default=0.5 * (p_conn_N + p_conn_P),
        )
        return p_conn

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan  # Exclude autaptic connections
        return dist_mat

    @staticmethod
    def compute_bip_matrix(src_pos, tgt_pos, bip_coord):
        """Computes bipolar matrix between pairs of neurons along specified coordinate axis (default: 2..z-axis),

        defined as sign of target (POST-synaptic) minus source (PRE-synaptic) coordinate value
        (i.e., POST-synaptic neuron below (delta < 0) or above (delta > 0) PRE-synaptic neuron assuming
         axis values increasing from lower to upper layers)
        """
        bip_mat = np.sign(
            np.diff(
                np.meshgrid(src_pos[:, bip_coord], tgt_pos[:, bip_coord], indexing="ij"), axis=0
            )[0, :, :]
        )  # Bipolar distinction based on difference in specified coordinate
        return bip_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (bipolar distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        bip_mat = self.compute_bip_matrix(src_pos, tgt_pos, self.bip_coord)
        return self.get_conn_prob(dist_mat, bip_mat)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn(d, delta) = {self.scale_N:.3f} * exp(-{self.exponent_N:.3f} * d) if delta < 0\n"
        )
        model_str = (
            model_str
            + f"                     {self.scale_P:.3f} * exp(-{self.exponent_P:.3f} * d) if delta > 0\n"
        )
        model_str = model_str + "                     AVERAGE OF BOTH MODELS  if delta == 0\n"
        model_str = (
            model_str
            + f"  d...distance, delta...difference (tgt minus src) in coordinate {self.bip_coord}"
        )
        return model_str


class ConnProb3rdOrderComplexExpModel(AbstractModel):
    """3rd order connection probability model (bipolar complex exponential distance-dependent),

    based on a complex (proximal) exponential and a simple (distal) exponential function
    - Returns (bipolar distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = [
        "prox_scale_P",
        "prox_scale_N",
        "prox_exp_P",
        "prox_exp_N",
        "prox_exp_pow_P",
        "prox_exp_pow_N",
        "dist_scale_P",
        "dist_scale_N",
        "dist_exp_P",
        "dist_exp_N",
        "bip_coord",
    ]
    param_defaults = {}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        if np.all(
            np.isnan([getattr(self, p) for p in np.setdiff1d(self.param_names, "bip_coord")])
        ):
            log.warning("Empty/invalid model!")
        else:
            log.log_assert(
                0.0 <= self.prox_scale_P <= 1.0 and 0.0 <= self.prox_scale_N <= 1.0,
                '"prox_scale_P/N" must be between 0 and 1!',
            )
            log.log_assert(
                self.prox_exp_P >= 0.0 and self.prox_exp_N >= 0.0,
                '"prox_exp_P/N" must be non-negative!',
            )
            log.log_assert(
                self.prox_exp_pow_P >= 0.0 and self.prox_exp_pow_N >= 0.0,
                '"prox_exp_pow_P/N" must be non-negative!',
            )
            log.log_assert(
                0.0 <= self.dist_scale_P <= 1.0 and 0.0 <= self.dist_scale_N <= 1.0,
                '"dist_scale_P/N" must be between 0 and 1!',
            )
            log.log_assert(
                self.dist_exp_P >= 0.0 and self.dist_exp_N >= 0.0,
                '"dist_exp_P/N" must be non-negative!',
            )
            log.log_assert(
                isinstance(self.bip_coord, int) and 0 <= self.bip_coord <= 2,
                'Bipolar coordinate "bip_coord" out of range!',
            )
            test_distance = 1000.0
            if self.exp_fct(
                test_distance, 1.0, self.prox_exp_P, self.prox_exp_pow_P
            ) >= self.exp_fct(test_distance, 1.0, self.dist_exp_P, 1.0):
                log.warning(
                    f"Proximal (P) exponential decays slower than distal (P) exponential ({self.get_param_dict()})!"
                )
            if self.exp_fct(
                test_distance, 1.0, self.prox_exp_N, self.prox_exp_pow_N
            ) >= self.exp_fct(test_distance, 1.0, self.dist_exp_N, 1.0):
                log.warning(
                    f"Proximal (N) exponential decays slower than distal (N) exponential ({self.get_param_dict()})!"
                )

    @staticmethod
    def exp_fct(distance, scale, exponent, exp_power=1.0):
        """Distance-dependent (complex) exponential probability function."""
        return scale * np.exp(-exponent * np.array(distance) ** exp_power)

    def get_conn_prob(self, distance, bip):
        """Return (bipolar distance-dependent) connection probability."""
        p_conn_N = self.exp_fct(
            distance, self.prox_scale_N, self.prox_exp_N, self.prox_exp_pow_N
        ) + self.exp_fct(distance, self.dist_scale_N, self.dist_exp_N, 1.0)
        p_conn_P = self.exp_fct(
            distance, self.prox_scale_P, self.prox_exp_P, self.prox_exp_pow_P
        ) + self.exp_fct(distance, self.dist_scale_P, self.dist_exp_P, 1.0)
        p_conn = np.select(
            [np.array(bip) < 0.0, np.array(bip) > 0.0],
            [p_conn_N, p_conn_P],
            default=0.5 * (p_conn_N + p_conn_P),
        )
        return p_conn

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan  # Exclude autaptic connections
        return dist_mat

    @staticmethod
    def compute_bip_matrix(src_pos, tgt_pos, bip_coord):
        """Computes bipolar matrix between pairs of neurons along specified coordinate axis (default: 2..z-axis),

        defined as sign of target (POST-synaptic) minus source (PRE-synaptic) coordinate value
        (i.e., POST-synaptic neuron below (delta < 0) or above (delta > 0) PRE-synaptic neuron assuming
         axis values increasing from lower to upper layers)
        """
        bip_mat = np.sign(
            np.diff(
                np.meshgrid(src_pos[:, bip_coord], tgt_pos[:, bip_coord], indexing="ij"), axis=0
            )[0, :, :]
        )  # Bipolar distinction based on difference in specified coordinate
        return bip_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (bipolar distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        bip_mat = self.compute_bip_matrix(src_pos, tgt_pos, self.bip_coord)
        return self.get_conn_prob(dist_mat, bip_mat)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn(d, delta) = {self.prox_scale_N:.3f} * exp(-{self.prox_exp_N:.6f} * d^{self.prox_exp_pow_N:.3f}) + {self.dist_scale_N:.3f} * exp(-{self.dist_exp_N:.3f} * d) if delta < 0\n"
        )
        model_str = (
            model_str
            + f"                     {self.prox_scale_P:.3f} * exp(-{self.prox_exp_P:.6f} * d^{self.prox_exp_pow_P:.3f}) + {self.dist_scale_P:.3f} * exp(-{self.dist_exp_P:.3f} * d) if delta > 0\n"
        )
        model_str = model_str + "                     AVERAGE OF BOTH MODELS  if delta == 0\n"
        model_str = (
            model_str
            + f"  d...distance, delta...difference (tgt minus src) in coordinate {self.bip_coord}"
        )
        return model_str


class ConnProb4thOrderLinInterpnModel(AbstractModel):
    """4th order connection probability model (offset-dependent, linearly interpolated):

    - Returns (offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = []
    param_defaults = {}
    data_names = ["p_conn_table"]
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            len(self.p_conn_table.index.levels) == 3,
            "Data frame with 3 index levels (dx, dy, dz) required!",
        )
        log.log_assert(
            self.p_conn_table.shape[1] == 1,
            "Data frame with 1 column (conn. prob.) required!",
        )

        self._data_points = [
            list(lev_pos) for lev_pos in self.p_conn_table.index.levels
        ]  # Extract data offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = (
            np.array(self.p_data.shape) > 1
        )  # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

    @property
    def data_points(self):
        """Return data offsets."""
        return self._data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, dx, dy, dz):
        """Return (offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self._data_points) if self.data_dim_sel[idx]]
        inp_sel = [
            val
            for idx, val in enumerate([np.array(dx), np.array(dy), np.array(dz)])
            if self.data_dim_sel[idx]
        ]

        # p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        # [BUG]: Extrapolation artifacts under some circumstances [ACCS-37]

        # [FIX]: Don't use extrapolation, but set to zero instead
        #      + check that probability values at borders are sufficiently small, so that no (big) jumps at borders
        #        (otherwise, sampling range should be increased)
        p_conn = np.minimum(
            np.maximum(
                interpn(
                    data_sel,
                    np.squeeze(self.p_data),
                    np.array(inp_sel).T,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                ),
                0.0,
            ),
            1.0,
        ).T

        # Check max. probability at boders (at dimensions that are actually interpolated)
        p_border = 0.0
        for dim in np.where(self.data_dim_sel)[0]:
            p_border = np.maximum(
                p_border, np.max(np.max(self.p_data, dim)[[0, -1]])
            )  # Take first/last element per dimension
        if p_border > P_TH_ABS or p_border / np.max(self.p_data) > P_TH_REL:
            log.warning(
                f"Probability at border should be close to zero (p_abs={p_border:.2f} (th_abs={P_TH_ABS:.2f}); p_rel={p_border / np.max(self.p_data):.2f} (th_rel={P_TH_REL:.2f})). Consider smoothing and/or increasing max. sampling range!"
            )

        return p_conn

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dx/dy/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in y coordinate
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in z coordinate
        return dx_mat, dy_mat, dz_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        dx_mat, dy_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
        return self.get_conn_prob(dx_mat, dy_mat, dz_mat)

    def __str__(self):
        """Return model string describing the model."""
        inp_names = np.array(["dx", "dy", "dz"])
        inp_str = ", ".join(inp_names[self.data_dim_sel])
        range_str = ", ".join(
            [
                f"{inp_names[i]}({len(self.p_conn_table.index.levels[i])}): {np.min(self.p_conn_table.index.levels[i]):.2f}..{np.max(self.p_conn_table.index.levels[i]):.2f}"
                for i in range(self.p_conn_table.index.nlevels)
                if self.data_dim_sel[i]
            ]
        )
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n"
        )
        model_str = model_str + "  dx/dy/dz...position offset (tgt minus src) in x/y/z dimension"
        return model_str


class ConnProb4thOrderLinInterpnReducedModel(AbstractModel):
    """Reduced 4th order connection probability model (offset-dependent in radial/axial direction, linearly interpolated):

    - Returns (offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["axial_coord"]
    param_defaults = {}
    data_names = ["p_conn_table"]
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            isinstance(self.axial_coord, int) and 0 <= self.axial_coord <= 2,
            'Axial coordinate "axial_coord" out of range!',
        )
        log.log_assert(
            len(self.p_conn_table.index.levels) == 2,
            "Data frame with 2 index levels (dr, dz) required!",
        )
        log.log_assert(
            self.p_conn_table.shape[1] == 1,
            "Data frame with 1 column (conn. prob.) required!",
        )

        self._data_points = [
            list(lev_pos) for lev_pos in self.p_conn_table.index.levels
        ]  # Extract data offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = (
            np.array(self.p_data.shape) > 1
        )  # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

        # Mirror around dr == 0, so that smooth interpolation for dr towards 0
        if self.data_dim_sel[0]:
            self._data_points[0] = [-d for d in self._data_points[0][::-1]] + self._data_points[0]
            self.p_data = np.vstack([self.p_data[::-1, :], self.p_data])

    @property
    def data_points(self):
        """Return data offsets."""
        return self._data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, dr, dz):
        """Return (offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self._data_points) if self.data_dim_sel[idx]]
        inp_sel = [
            val for idx, val in enumerate([np.array(dr), np.array(dz)]) if self.data_dim_sel[idx]
        ]

        # p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        # [BUG]: Extrapolation artifacts under some circumstances [ACCS-37]

        # [FIX]: Don't use extrapolation, but set to zero instead
        #      + check that probability values at borders are sufficiently small, so that no (big) jumps at borders
        #        (otherwise, sampling range should be increased)
        p_conn = np.minimum(
            np.maximum(
                interpn(
                    data_sel,
                    np.squeeze(self.p_data),
                    np.array(inp_sel).T,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                ),
                0.0,
            ),
            1.0,
        ).T

        # Check max. probability at boders (at dimensions that are actually interpolated)
        p_border = 0.0
        for dim in np.where(self.data_dim_sel)[0]:
            p_border = np.maximum(
                p_border, np.max(np.max(self.p_data, dim)[[0, -1]])
            )  # Take first/last element per dimension
        if p_border > P_TH_ABS or p_border / np.max(self.p_data) > P_TH_REL:
            log.warning(
                f"Probability at border should be close to zero (p_abs={p_border:.2f} (th_abs={P_TH_ABS:.2f}); p_rel={p_border / np.max(self.p_data):.2f} (th_rel={P_TH_REL:.2f})). Consider smoothing and/or increasing max. sampling range!"
            )

        return p_conn

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos, axial_coord):
        """Computes radial/axial offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        d_matrices = [
            np.diff(np.meshgrid(src_pos[:, i], tgt_pos[:, i], indexing="ij"), axis=0)[0, :, :]
            for i in range(3)
        ]  # Relative differences in x/y/z coordinates

        radial_coords = list(set(range(3)) - {axial_coord})  # Radial coordinates
        dr_mat = np.sqrt(
            d_matrices[radial_coords[0]] ** 2 + d_matrices[radial_coords[1]] ** 2
        )  # Relative offset in radial plane (Euclidean distance)
        dz_mat = d_matrices[axial_coord]  # Axial offset

        return dr_mat, dz_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        dr_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos, self.axial_coord)
        return self.get_conn_prob(dr_mat, dz_mat)

    def __str__(self):
        """Return model string describing the model."""
        inp_names = np.array(["dr", "dz"])
        inp_str = ", ".join(inp_names[self.data_dim_sel])
        range_str = ", ".join(
            [
                f"{inp_names[i]}({len(self.p_conn_table.index.levels[i])}): {np.min(self.p_conn_table.index.levels[i]):.2f}..{np.max(self.p_conn_table.index.levels[i]):.2f}"
                for i in range(self.p_conn_table.index.nlevels)
                if self.data_dim_sel[i]
            ]
        )
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n"
        )
        model_str = (
            model_str
            + f"  dr/dz...radial/axial position offset (tgt minus src), with axial coordinate {self.axial_coord}"
        )
        return model_str


class ConnProb5thOrderLinInterpnModel(AbstractModel):
    """5th order connection probability model (position- & offset-dependent, linearly interpolated):

    - Returns (position- & offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = []
    param_defaults = {}
    data_names = ["p_conn_table"]
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            len(self.p_conn_table.index.levels) == 6,
            "Data frame with 6 index levels (x, y, z, dx, dy, dz) required!",
        )
        log.log_assert(
            self.p_conn_table.shape[1] == 1,
            "Data frame with 1 column (conn. prob.) required!",
        )

        self._data_points = [
            list(lev_pos) for lev_pos in self.p_conn_table.index.levels
        ]  # Extract data positions & offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = (
            np.array(self.p_data.shape) > 1
        )  # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

    @property
    def data_points(self):
        """Return data offsets."""
        return self._data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, x, y, z, dx, dy, dz):
        """Return (position- & offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self._data_points) if self.data_dim_sel[idx]]
        inp_sel = [
            val
            for idx, val in enumerate(
                [np.array(x), np.array(y), np.array(z), np.array(dx), np.array(dy), np.array(dz)]
            )
            if self.data_dim_sel[idx]
        ]

        # p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        # [BUG]: Extrapolation artifacts under some circumstances [ACCS-37]

        # [FIX]: Don't use extrapolation, but set to zero instead
        #      + check that probability values at borders are sufficiently small, so that no (big) jumps at borders
        #        (otherwise, sampling range should be increased)
        p_conn = np.minimum(
            np.maximum(
                interpn(
                    data_sel,
                    np.squeeze(self.p_data),
                    np.array(inp_sel).T,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                ),
                0.0,
            ),
            1.0,
        ).T

        # Check max. probability at boders (at dimensions that are actually interpolated)
        p_border = 0.0
        for dim in np.where(self.data_dim_sel)[0]:
            p_border = np.maximum(
                p_border, np.max(np.max(self.p_data, dim)[[0, -1]])
            )  # Take first/last element per dimension
        if p_border > P_TH_ABS or p_border / np.max(self.p_data) > P_TH_REL:
            log.warning(
                f"Probability at border should be close to zero (p_abs={p_border:.2f} (th_abs={P_TH_ABS:.2f}); p_rel={p_border / np.max(self.p_data):.2f} (th_rel={P_TH_REL:.2f})). Consider smoothing and/or increasing max. sampling range!"
            )

        return p_conn

    @staticmethod
    def compute_position_matrices(src_pos, tgt_pos):
        """Computes x/y/z position matrices of src/PRE neurons (src/PRE neuron positions repeated over tgt/POST neuron number)."""
        x_mat, y_mat, z_mat = [
            np.tile(src_pos[:, i : i + 1], [1, tgt_pos.shape[0]]) for i in range(src_pos.shape[1])
        ]
        return x_mat, y_mat, z_mat

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dx/dy/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in y coordinate
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in z coordinate
        return dx_mat, dy_mat, dz_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (position- & offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        x_mat, y_mat, z_mat = self.compute_position_matrices(src_pos, tgt_pos)
        dx_mat, dy_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
        return self.get_conn_prob(x_mat, y_mat, z_mat, dx_mat, dy_mat, dz_mat)

    def __str__(self):
        """Return model string describing the model."""
        inp_names = np.array(["x", "y", "z", "dx", "dy", "dz"])
        inp_str = ", ".join(inp_names[self.data_dim_sel])
        range_str = ", ".join(
            [
                f"{inp_names[i]}({len(self.p_conn_table.index.levels[i])}): {np.min(self.p_conn_table.index.levels[i]):.2f}..{np.max(self.p_conn_table.index.levels[i]):.2f}"
                for i in range(self.p_conn_table.index.nlevels)
                if self.data_dim_sel[i]
            ]
        )
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n"
        )
        model_str = (
            model_str
            + "  x/y/z...src position, dx/dy/dz...position offset (tgt minus src) in x/y/z dimension"
        )
        return model_str


class ConnProb5thOrderLinInterpnReducedModel(AbstractModel):
    """Reduced 5th order connection probability model (position-dependent in axial direction & offset-dependent in radial/axial direction, linearly interpolated):

    - Returns (position- & offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["axial_coord"]
    param_defaults = {}
    data_names = ["p_conn_table"]
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            isinstance(self.axial_coord, int) and 0 <= self.axial_coord <= 2,
            'Axial coordinate "axial_coord" out of range!',
        )
        log.log_assert(
            len(self.p_conn_table.index.levels) == 3,
            "Data frame with 3 index levels (z, dr, dz) required!",
        )
        log.log_assert(
            self.p_conn_table.shape[1] == 1,
            "Data frame with 1 column (conn. prob.) required!",
        )

        self._data_points = [
            list(lev_pos) for lev_pos in self.p_conn_table.index.levels
        ]  # Extract data positions & offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = (
            np.array(self.p_data.shape) > 1
        )  # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

        # Mirror around dr == 0, so that smooth interpolation for dr towards 0
        if self.data_dim_sel[1]:
            self._data_points[1] = [-d for d in self._data_points[1][::-1]] + self._data_points[1]
            self.p_data = np.hstack([self.p_data[:, ::-1, :], self.p_data])

    @property
    def data_points(self):
        """Return data offsets."""
        return self._data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, z, dr, dz):
        """Return (position- & offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self._data_points) if self.data_dim_sel[idx]]
        inp_sel = [
            val
            for idx, val in enumerate([np.array(z), np.array(dr), np.array(dz)])
            if self.data_dim_sel[idx]
        ]

        # p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        # [BUG]: Extrapolation artifacts under some circumstances [ACCS-37]

        # [FIX]: Don't use extrapolation, but set to zero instead
        #      + check that probability values at borders are sufficiently small, so that no (big) jumps at borders
        #        (otherwise, sampling range should be increased)
        p_conn = np.minimum(
            np.maximum(
                interpn(
                    data_sel,
                    np.squeeze(self.p_data),
                    np.array(inp_sel).T,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                ),
                0.0,
            ),
            1.0,
        ).T

        # Check max. probability at boders (at dimensions that are actually interpolated)
        p_border = 0.0
        for dim in np.where(self.data_dim_sel)[0]:
            p_border = np.maximum(
                p_border, np.max(np.max(self.p_data, dim)[[0, -1]])
            )  # Take first/last element per dimension
        if p_border > P_TH_ABS or p_border / np.max(self.p_data) > P_TH_REL:
            log.warning(
                f"Probability at border should be close to zero (p_abs={p_border:.2f} (th_abs={P_TH_ABS:.2f}); p_rel={p_border / np.max(self.p_data):.2f} (th_rel={P_TH_REL:.2f})). Consider smoothing and/or increasing max. sampling range!"
            )

        return p_conn

    @staticmethod
    def compute_position_matrix(src_pos, tgt_pos, axial_coord):
        """Computes axial position matrix of src/PRE neurons (src/PRE neuron positions repeated over tgt/POST neuron number)."""
        z_mat = np.tile(src_pos[:, [axial_coord]], [1, tgt_pos.shape[0]])  # Axial position
        return z_mat

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos, axial_coord):
        """Computes radial/axial offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        d_matrices = [
            np.diff(np.meshgrid(src_pos[:, i], tgt_pos[:, i], indexing="ij"), axis=0)[0, :, :]
            for i in range(3)
        ]  # Relative differences in x/y/z coordinates

        radial_coords = list(set(range(3)) - {axial_coord})  # Radial coordinates
        dr_mat = np.sqrt(
            d_matrices[radial_coords[0]] ** 2 + d_matrices[radial_coords[1]] ** 2
        )  # Relative offset in radial plane (Euclidean distance)
        dz_mat = d_matrices[axial_coord]  # Axial offset

        return dr_mat, dz_mat

    def get_model_output(self, src_pos, tgt_pos):  # pylint: disable=arguments-differ
        """Return (position- & offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        z_mat = self.compute_position_matrix(src_pos, tgt_pos, self.axial_coord)
        dr_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos, self.axial_coord)
        return self.get_conn_prob(z_mat, dr_mat, dz_mat)

    def __str__(self):
        """Return model string describing the model."""
        inp_names = np.array(["z", "dr", "dz"])
        inp_str = ", ".join(inp_names[self.data_dim_sel])
        range_str = ", ".join(
            [
                f"{inp_names[i]}({len(self.p_conn_table.index.levels[i])}): {np.min(self.p_conn_table.index.levels[i]):.2f}..{np.max(self.p_conn_table.index.levels[i]):.2f}"
                for i in range(self.p_conn_table.index.nlevels)
                if self.data_dim_sel[i]
            ]
        )
        model_str = f"{self.__class__.__name__}\n"
        model_str = (
            model_str
            + f"  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n"
        )
        model_str = (
            model_str
            + f"  z...axial src position, dr/dz...radial/axial position offset (tgt minus src), with axial coordinate {self.axial_coord}"
        )
        return model_str


class LookupTableModel(AbstractModel):
    """Generic model to access any (sparse) information based on source and target neuron IDs.

    This model can be used for adjacency matrices (bool), deterministic connection
    probabilities (float), specific #syn/conn (int), etc.
    - Data are internally stored as sparse matrix in CSC format
      (initialized from data frame with 'row_ind', 'col_ind', and 'value' columns)
    - Size of that matrix matches selected src/dest neuron selections
      (initialized from data frames with 'src_node_ids' and 'tgt_node_ids')
    - Returns (deterministic) value (always float) for given source/target neuron IDs
      (or 0.0 if no value is stored)
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = []
    param_defaults = {}
    data_names = ["src_nodes_table", "tgt_nodes_table", "lookup_table"]
    input_names = ["src_nid", "tgt_nid"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            self.src_nodes_table.shape[1] == 1,
            "Data frame with 1 column (src_node_ids) required!",
        )
        self.src_node_ids = self.src_nodes_table.to_numpy().flatten()
        self.src_idx_table = pd.Series(
            self.src_nodes_table.index, index=self.src_node_ids, name="src_index"
        )

        log.log_assert(
            self.tgt_nodes_table.shape[1] == 1,
            "Data frame with 1 column (tgt_node_ids) required!",
        )
        self.tgt_node_ids = self.tgt_nodes_table.to_numpy().flatten()
        self.tgt_idx_table = pd.Series(
            self.tgt_nodes_table.index, index=self.tgt_node_ids, name="tgt_index"
        )

        log.log_assert(
            self.lookup_table.shape[1] == 3,
            "Data frame with 3 columns (row_ind, col_ind, value) required!",
        )
        self.lut_mat = csc_matrix(
            (
                self.lookup_table["value"],
                (self.lookup_table["row_ind"], self.lookup_table["col_ind"]),
            ),
            shape=(len(self.src_nodes_table), len(self.tgt_nodes_table)),
        )

    def get_lookup_table(self):
        """Return underlying (sparse) LUT matrix."""
        return self.lut_mat

    def get_src_nids(self):
        """Return source node IDs stored in this model."""
        return self.src_node_ids

    def get_tgt_nids(self):
        """Return target node IDs stored in this model."""
        return self.tgt_node_ids

    def get_model_output(self, src_nid, tgt_nid):  # pylint: disable=arguments-differ
        """Return LUT values of size <#src x #tgt> as given by underlying LUT matrix."""
        src_sel = self.src_idx_table.loc[src_nid].values
        tgt_sel = self.tgt_idx_table.loc[tgt_nid].values
        mat_sel = self.lut_mat[:, tgt_sel][src_sel, :].toarray()
        return mat_sel

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}"
        model_str = model_str + "\n  " + self.lut_mat.__repr__()
        if len(self.lut_mat.data) > 0:
            range_str = f"{self.lut_mat.data.min()}"
            if self.lut_mat.data.max() > self.lut_mat.data.min():
                range_str = range_str + f"..{self.lut_mat.data.max()}"
        else:
            range_str = "None"
        model_str = model_str + "\n  Value range: " + range_str
        model_str = model_str + f" (dtype: {self.lut_mat.dtype})"
        return model_str

    @staticmethod
    def init_from_sparse_matrix(matrix, src_node_ids, tgt_node_ids):
        """Model initialization from sparse matrix."""
        log.log_assert(sps.issparse(matrix), "Matrix must be in sparse format!")
        log.log_assert(matrix.shape[0] == len(src_node_ids), "Source node IDs mismatch!")
        log.log_assert(matrix.shape[1] == len(tgt_node_ids), "Target node IDs mismatch!")

        src_nodes_table = pd.DataFrame(src_node_ids, columns=["src_node_ids"])
        tgt_nodes_table = pd.DataFrame(tgt_node_ids, columns=["tgt_node_ids"])
        matrix = matrix.tocoo()  # Convert to COO, for easy access to row/col and data!!
        lookup_table = pd.DataFrame(
            {"row_ind": matrix.row, "col_ind": matrix.col, "value": matrix.data}
        )

        return LookupTableModel(
            src_nodes_table=src_nodes_table,
            tgt_nodes_table=tgt_nodes_table,
            lookup_table=lookup_table,
        )


class ConnProbAdjModel(AbstractModel):
    """Deterministic connection probability model, defined by an adjacency matrix (internally stored as LookupTableModel):

    - Adjacency matrix represented as boolean matrix in sparse CSC format
      (initialized from data frame with 'row_ind' and 'col_ind' columns)
    - Size of adjacency matrix matches selected src/dest neuron selections
      (initialized from data frames with 'src_node_ids' and 'tgt_node_ids')
    - Returns deterministic connection probability (0.0 or 1.0) for given source/target neuron IDs
    - Can optionally store inverted connectivity (i.e., True...no connection, False...connection)
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["inverted"]
    param_defaults = {"inverted": False}
    data_names = ["src_nodes_table", "tgt_nodes_table", "adj_table"]
    input_names = ["src_nid", "tgt_nid"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert(
            isinstance(self.inverted, bool),
            "Inverted flag must be boolean!",
        )

        # Init internal LUT
        lookup_table = self.adj_table.copy()
        lookup_table["value"] = [True] * lookup_table.shape[0]
        self.lut = LookupTableModel(
            src_nodes_table=self.src_nodes_table,
            tgt_nodes_table=self.tgt_nodes_table,
            lookup_table=lookup_table,
        )

    def get_adj_matrix(self):
        """Return adjacency matrix."""
        return self.lut.get_lookup_table()

    def get_src_nids(self):
        """Return source node IDs stored in this model."""
        return self.lut.src_node_ids

    def get_tgt_nids(self):
        """Return target node IDs stored in this model."""
        return self.lut.tgt_node_ids

    def is_inverted(self):
        """Return if connectivity is stored inverted."""
        return self.inverted

    def get_model_output(self, src_nid, tgt_nid):  # pylint: disable=arguments-differ
        """Return deterministic connection probabilities (0.0 or 1.0) of size <#src x #tgt> as given by adjacency matrix."""
        adj_sel = self.lut.get_model_output(src_nid, tgt_nid)
        if self.inverted:
            return np.logical_not(adj_sel).astype(float)
        else:
            return adj_sel.astype(float)

    def __str__(self):
        """Return model string describing the model."""
        if self.inverted:
            inv_str = "Inverted "
        else:
            inv_str = ""
        model_str = f"{inv_str}{self.__class__.__name__}\n"
        model_str = model_str + "  " + self.lut.get_lookup_table().__repr__()
        return model_str


class PropsTableModel(AbstractModel):
    """Generic model to store any/multiple (edge) properties based on source and target neuron IDs.

    e.g., this model can be used for providing synapse positions, etc.
    - Data are initialized and internally stored as data frame with properties as columns
    - Source/target node IDs must be provided as @source_node/@target_node columns
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = []
    param_defaults = {}
    data_names = ["props_table"]
    input_names = ["src_nid", "tgt_nid", "prop_names", "num_sel"]
    # Notes: "prop_names" is optional; if not provided, all properties will be returned
    #        "num_sel" is optional; if not provided, all entries will be returned; otherwise, only the first num_sel

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        log.log_assert("@source_node" in self.props_table.columns, "@source_node column required!")
        log.log_assert("@target_node" in self.props_table.columns, "@target_node column required!")
        self.property_names = np.setdiff1d(
            self.props_table.columns, ["@source_node", "@target_node"]
        ).tolist()
        log.log_assert(len(self.property_names) > 0, "Properties missing!")
        log.log_assert(self.props_table.size > 0, "Properties table empty!")

    def get_property_names(self):
        """Return property names."""
        return self.property_names

    def get_src_nids(self):
        """Return (unique) source node IDs stored in this model."""
        return np.unique(self.props_table["@source_node"])

    def get_tgt_nids(self):
        """Return (unique) target node IDs stored in this model."""
        return np.unique(self.props_table["@target_node"])

    def get_src_tgt_counts(self):
        """Return (unique) source/target node ID pairs and counts stored in this model."""
        return np.unique(
            self.props_table[["@source_node", "@target_node"]], axis=0, return_counts=True
        )

    def get_model_output(
        self, src_nid, tgt_nid, prop_names=None, num_sel=None
    ):  # pylint: disable=arguments-differ
        """Return property values of given source/target node IDs."""
        src_sel = np.isin(self.props_table["@source_node"], src_nid)
        tgt_sel = np.isin(self.props_table["@target_node"], tgt_nid)
        if prop_names is None:
            prop_names = self.property_names
        tab = self.props_table[prop_names].loc[np.logical_and(src_sel, tgt_sel)]
        if num_sel is None:
            return tab
        else:
            log.log_assert(
                0 <= num_sel <= tab.shape[0], "Selected number of elements out of range!"
            )
            return tab[:num_sel]

    def apply(self, **kwargs):
        """Apply the model. Overwrite to set default."""
        if "prop_names" not in kwargs:
            kwargs["prop_names"] = None
        if "num_sel" not in kwargs:
            kwargs["num_sel"] = None
        return super().apply(**kwargs)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}"
        model_str = model_str + f"\n  Properties: {self.property_names}"
        model_str = model_str + f"\n  Entries: {self.props_table.shape[0]}"
        return model_str
