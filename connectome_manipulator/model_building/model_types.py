"""Definition and mapping of model types to classes"""

from abc import ABCMeta, abstractmethod
from functools import cached_property, lru_cache
import itertools
import os
import sys

import jsonpickle
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from scipy.spatial import distance_matrix
from scipy.stats import truncnorm

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
            model_dict = jsonpickle.decode(f.read())
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
            log.warning(f"Unused parameter(s): {set(kwargs.keys())}!")

    def init_params(self, model_dict):
        """Initialize model parameters from dict (removing used keys from dict)."""
        log.log_assert(
            all(p in model_dict or p in self.param_defaults for p in self.param_names),
            f"Missing parameters for model initialization! Must contain initialization for {set(self.param_names) - set(self.param_defaults)}.",
        )
        for p in self.param_names:
            if p in model_dict:
                val = np.array(
                    model_dict.pop(p)
                ).tolist()  # Convert to numpy and back to list, so that reduced to basic (non-numpy) data types
            else:  # Use value from defaults
                val = np.array(
                    self.param_defaults[p]
                ).tolist()  # Convert to numpy and back to list, so that reduced to basic (non-numpy) data types
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
            log.warning(f"Unused input(s): {set(kwargs.keys())}!")
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
            df.to_hdf(data_file, key, append=idx > 0)
        model_dict["data_keys"] = list(data_dict.keys())

        # Save model dict to .json file [using jsonpickle, so also non-serializable objects can be stored!]
        model_dict["model"] = self.__class__.__name__
        model_dict["__version_info__"] = {
            "python": sys.version,
            "jsonpickle": jsonpickle.__version__,
            "pandas": pd.__version__,
        }
        with open(model_file, "w") as f:
            f.write(jsonpickle.encode(model_dict, indent=2))


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
    pathway_param_name = "pathway_specs"
    pathway_input_names = ["src_type", "tgt_type"]

    def __init__(self, **kwargs):
        """Model initialization."""
        log.log_assert(
            self.pathway_param_name not in self.param_names,
            f'"{self.pathway_param_name}" must not be part of model parameter names!',
        )
        log.log_assert(
            not any(p in self.input_names for p in self.pathway_input_names),
            f"Model inputs {self.pathway_input_names} already part of base model!",
        )
        self.property_names = self.param_names
        self.param_names = [self.pathway_param_name] + self.param_names
        self.input_names = self.pathway_input_names + self.input_names
        if self.pathway_param_name not in kwargs:
            kwargs[self.pathway_param_name] = {}  # Add empty dict for initialization
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
        log.log_assert(isinstance(self.pathway_specs, dict), '"pathway_specs" dictionary required!')
        for src, tgt_dict in self.pathway_specs.items():
            log.log_assert(
                isinstance(tgt_dict, dict) and len(tgt_dict) > 0,
                "Pathway target dictionary missing or empty!",
            )
            for tgt, distr_dict in tgt_dict.items():
                log.log_assert(
                    isinstance(distr_dict, dict)
                    and len(distr_dict) >= 1  # At least one property must be set
                    and all(attr in self.property_names for attr in distr_dict),
                    f"Pathway property missing or unknown (must be of: {self.property_names})!",
                )
                log.log_assert(src is not None and tgt is not None, "Invalid pathway identifier!")
        self.pathway_specs = dict_conv(self.pathway_specs)  # Convert dict to basic data types

    def has_property(self, prop_name):
        """Return if a property is part of this model."""
        return prop_name in self.property_names

    def get_properties(self):
        """Return list of model property names."""
        return self.property_names.copy()

    def has_pathway(self, src_type, tgt_type, prop_name=None):
        """Return if the given pathway (and optionally, the given property) is stored in this model."""
        types_in_spec = src_type in self.pathway_specs and tgt_type in self.pathway_specs[src_type]
        if not types_in_spec:
            return False
        if prop_name:
            return prop_name in self.pathway_specs[src_type][tgt_type]
        return True

    def get_pathways(self):
        """Return list of pathways stored in this model."""
        for k, v in self.pathway_specs.items():
            for el in v:
                yield k, el

    def has_default(self, prop_name):
        """Return if a property has a default value stored in this model."""
        log.log_assert(self.has_property(prop_name), f'Property "{prop_name}" unknown!')
        return hasattr(self, prop_name) and getattr(self, prop_name) is not None

    def get_default(self, prop_name):
        """Return default value for given property."""
        log.log_assert(self.has_property(prop_name), f'Property "{prop_name}" unknown!')
        log.log_assert(self.has_default(prop_name), f'No default value for "{prop_name}"!')
        return getattr(self, prop_name)

    @cached_property
    def default_dict(self):
        """Return dictionary with default property values, if any."""
        default_dict = {}
        for prop in self.property_names:
            if self.has_default(prop):
                default_dict[prop] = self.get_default(prop)
        return default_dict

    def get_pathway_property(self, prop_name, src_type=None, tgt_type=None):
        """Acces method returning a pathway property value for a given pair of src_type and tgt_type.

        Note: This method can be used in any derived class to access pathway properties.
        """
        try:
            return self.pathway_specs[src_type][tgt_type][prop_name]
        except KeyError:
            return self.get_default(prop_name)

    @lru_cache(maxsize=1_000_000)
    def get_pathway_dict(self, src_type=None, tgt_type=None, default_if_missing=True):
        """Access method returning a pathway properties dict for a given pair of src_type and tgt_type.

        Optionally, if a pathway or property is not specified or does not exists in the model, the default is returned.
        Note: This method is can be used in any derived class to access pathway properties.
        """
        try:
            if not default_if_missing:
                return self.pathway_specs[src_type][tgt_type]
            else:
                return self.default_dict | self.pathway_specs[src_type][tgt_type]
        except KeyError:
            if not default_if_missing:
                raise
            return self.default_dict

    def apply(self, **kwargs):
        """Apply model, but setting pathway inputs to default values (None)."""
        for inp in self.pathway_input_names:
            if inp not in kwargs:
                kwargs[inp] = None
        return super().apply(**kwargs)

    def __str__(self):
        """Return model string describing the model."""
        model_str = f"{self.__class__.__name__}\n"
        model_str = model_str + f"  Model properties: {self.property_names}\n"
        pw_list = list(self.get_pathways())
        model_str = model_str + f"  Property values for {len(pw_list)} pathways: "
        if len(pw_list) > 2:
            model_str = model_str + f"[{pw_list[0]}..{pw_list[-1]}]\n"
        else:
            model_str = model_str + f"{pw_list}\n"
        model_str = model_str + f"  Default: {self.default_dict}"
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

        # Check init values of all pathways + default
        pw_list = self.get_pathways()
        for src, tgt in itertools.chain(pw_list, [(None, None)]):
            attr_dict = self.get_pathway_dict(src, tgt, default_if_missing=True)
            if "mean" in attr_dict:
                log.log_assert(attr_dict["mean"] > 0.0, "Mean must be larger than zero!")
            if "std" in attr_dict:
                log.log_assert(attr_dict["std"] >= 0.0, "Std cannot be negative!")

    def get_model_output(self, **kwargs):
        """Draw #syn/conn value for one connection between src_type and tgt_type [seeded through numpy]."""
        # Get distribution attribute values
        attr_dict = self.get_pathway_dict(kwargs["src_type"], kwargs["tgt_type"])
        distr_mean = attr_dict["mean"]
        distr_std = attr_dict["std"]

        # Draw number of synapses
        if distr_std > 0.0:
            nsyn = np.random.gamma(
                shape=distr_mean**2 / distr_std**2,
                scale=distr_std**2 / distr_mean,
                size=1,
            )
        else:
            nsyn = distr_mean

        # Convert type
        nsyn = int(np.round(np.maximum(1, nsyn)))

        return nsyn


class LinDelayModel(PathwayModel):
    """Linear distance-dependent delay model for pairs of m-types [generative model]:

    - Delay mean: delay_mean_coefs[1] * distance + delay_mean_coefs[0] (linear)
    - Delay std: delay_std (constant)
    - Delay min: delay_min (constant)
    - Different delay attributes for specific pathways
    - Default delay attributes for any other pathways not specified
    """

    # Names of model inputs, parameters and data frames which are part of this model
    # (other than the ones inherited from PathwayModel class)
    param_names = ["delay_mean_coefs", "delay_std", "delay_min"]
    param_defaults = {"delay_mean_coefs": [0.75, 0.003], "delay_std": 0.5, "delay_min": 0.2}
    data_names = []
    input_names = ["distance"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check init values of all pathways + default
        pw_list = self.get_pathways()
        for src, tgt in itertools.chain(pw_list, [(None, None)]):
            attr_dict = self.get_pathway_dict(src, tgt, default_if_missing=True)
            if "delay_mean_coefs" in attr_dict:
                log.log_assert(
                    hasattr(attr_dict["delay_mean_coefs"], "__iter__")
                    and len(attr_dict["delay_mean_coefs"]) == 2,
                    "Two mean coefficients required for linear delay model!",
                )
            if "delay_std" in attr_dict:
                log.log_assert(attr_dict["delay_std"] >= 0.0, "Delay std cannot be negative!")
            if "delay_min" in attr_dict:
                log.log_assert(attr_dict["delay_min"] >= 0.0, "Delay min cannot be negative!")

    def get_mean(self, distance, src_type=None, tgt_type=None):
        """Get delay mean for given distance (linear)."""
        delay_mean_coefs = self.get_pathway_property("delay_mean_coefs", src_type, tgt_type)
        return delay_mean_coefs[1] * distance + delay_mean_coefs[0]

    def get_std(self, distance, src_type=None, tgt_type=None):
        """Get delay std for given distance (constant)."""
        delay_std = self.get_pathway_property("delay_std", src_type, tgt_type)
        return np.full_like(distance, delay_std, dtype=type(delay_std))

    def get_min(self, distance, src_type=None, tgt_type=None):
        """Get delay min for given distance (constant)."""
        delay_min = self.get_pathway_property("delay_min", src_type, tgt_type)
        return np.full_like(distance, delay_min, dtype=type(delay_min))

    def get_model_output(self, **kwargs):
        """Draw distance-dependent delay values from truncated normal distribution [seeded through numpy]."""
        d_mean = self.get_mean(np.array(kwargs["distance"]), kwargs["src_type"], kwargs["tgt_type"])
        d_std = self.get_std(np.array(kwargs["distance"]), kwargs["src_type"], kwargs["tgt_type"])
        d_min = self.get_min(np.array(kwargs["distance"]), kwargs["src_type"], kwargs["tgt_type"])
        if all(d_std > 0.0):
            return truncnorm(a=(d_min - d_mean) / d_std, b=np.inf, loc=d_mean, scale=d_std).rvs()
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

    def get_model_output(self, **kwargs):
        """Return (mapped) neuron positions for a given set of GIDs."""
        gids = np.array(kwargs["gids"])
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
    param_names = ["order", "coeffs"]
    param_defaults = {"order": 1, "coeffs": (0.0,)}
    data_names = []
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        # Derive default model order from coefficients (if possible)
        if "order" not in kwargs and "coeffs" in kwargs:
            if np.isscalar(kwargs["coeffs"]):
                kwargs["coeffs"] = (kwargs["coeffs"],)  # Convert scalar to tuple
            if len(kwargs["coeffs"]) == 1:
                kwargs["order"] = 1
            elif len(kwargs["coeffs"]) == 2:
                kwargs["order"] = 2
            else:
                log.log_assert(False, "Default model order cannot be determined!")

        # Derive pathway model order from coefficients (if possible)
        if "pathway_specs" in kwargs:
            for src, tgt_dict in kwargs["pathway_specs"].items():
                for tgt, attr_dict in tgt_dict.items():
                    if "order" not in attr_dict and "coeffs" in attr_dict:
                        if np.isscalar(attr_dict["coeffs"]):
                            attr_dict["coeffs"] = (attr_dict["coeffs"],)  # Convert scalar to tuple
                        if len(attr_dict["coeffs"]) == 1:
                            attr_dict["order"] = 1
                        elif len(attr_dict["coeffs"]) == 2:
                            attr_dict["order"] = 2
                        else:
                            log.log_assert(False, "Model order cannot be determined!")
        super().__init__(**kwargs)

        # Check init values of all pathways + default
        pw_list = self.get_pathways()
        for src, tgt in itertools.chain(pw_list, [(None, None)]):
            attr_dict = self.get_pathway_dict(src, tgt, default_if_missing=True)
            if np.all(np.isnan(attr_dict["coeffs"])):
                log.warning("Empty/invalid model!")
                # FIXME: should this not fail?
            if attr_dict["order"] == 1:
                log.log_assert(
                    len(attr_dict["coeffs"]) == 1, "Number of provided coefficients must be 1!"
                )
                log.log_assert(
                    0.0 <= attr_dict["coeffs"][0] <= 1.0,
                    "Connection probability must be between 0 and 1!",
                )
            elif attr_dict["order"] == 2:
                log.log_assert(
                    len(attr_dict["coeffs"]) == 2, "Number of provided coefficients must be 2"
                )
                log.log_assert(
                    0.0 <= attr_dict["coeffs"][0] <= 1.0, '"Scale" must be between 0 and 1!'
                )
                log.log_assert(attr_dict["coeffs"][1] >= 0.0, '"Exponent" must be non-negative!')
            else:
                log.log_assert(False, f"Order-{attr_dict['order']} model not implemented!")

    @staticmethod
    def exp_fct(distance, scale, exponent):
        """Distance-dependent exponential probability function."""
        return scale * np.exp(-exponent * np.array(distance))

    @staticmethod
    def prob_fct(order, coeffs, src_pos, tgt_pos):
        """Connection probability of given order."""
        if order == 1:
            return coeffs[0]
        elif order == 2:
            distance = np.sqrt(np.sum((src_pos - tgt_pos) ** 2))  # Euclidean distance
            return coeffs[0] * np.exp(-coeffs[1] * np.array(distance))
        else:
            raise ValueError("Order-{order} probability function not implemented!")

    def get_model_output(self, **kwargs):
        """Return pathway-specific connection probabilities <#ysrc x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
        src_type = kwargs[
            "src_type"
        ]  # May be scalar (or None), or array-like matching the size of src_pos
        tgt_type = kwargs[
            "tgt_type"
        ]  # May be scalar (or None), or array-like matching the size of tgt_pos
        if src_type is None or np.isscalar(src_type):
            src_type = [src_type] * src_pos.shape[0]
        if tgt_type is None or np.isscalar(tgt_type):
            tgt_type = [tgt_type] * tgt_pos.shape[0]

        p_mat = np.zeros((len(src_type), len(tgt_type)))
        for si, st in enumerate(src_type):
            for ti, tt in enumerate(tgt_type):
                props = self.get_pathway_dict(st, tt)
                p_mat[si, ti] = self.prob_fct(
                    props["order"], props["coeffs"], src_pos[si], tgt_pos[ti]
                )
        return p_mat


class ConnPropsModel(AbstractModel):
    """Connection/synapse properties model for pairs of m-types [generative model]:

    - Connection/synapse property values drawn from given distributions

    NOTE: 'std-within' larger than zero can be used to specify variability within
          connections (all properties except <N_SYN_PER_CONN_NAME>). However, the
          actual value of 'std-within' will only be used for distribution types that
          have a 'std' attribute, so that 'std' can be set to 'std-within'!
          For 'std-within' equal to zero, all values within a connection will be the same.
    """

    # Names of model inputs, parameters and data frames which are part of this model
    param_names = ["src_types", "tgt_types", "prop_stats"]
    param_defaults = {}
    data_names = []
    input_names = ["src_type", "tgt_type"]

    # Required attributes for given distributions
    distribution_attributes = {
        "constant": ["mean"],
        "normal": ["mean", "std"],
        "truncnorm": ["mean", "std", "min", "max"],
        "gamma": ["mean", "std"],
        "poisson": ["mean"],
        "discrete": ["val", "p"],
    }

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
        log.log_assert(N_SYN_PER_CONN_NAME in self.prop_names, f'"{N_SYN_PER_CONN_NAME}" missing')
        log.log_assert(
            all(isinstance(self.prop_stats[p], dict) for p in self.prop_names),
            "Property statistics dictionary required!",
        )
        log.log_assert(
            all(
                all(np.isin(self.src_types, list(self.prop_stats[p].keys())))
                for p in self.prop_names
            ),
            "Source type statistics missing!",
        )
        log.log_assert(
            all(
                [isinstance(self.prop_stats[p][src], dict) for p in self.prop_names]
                for src in self.src_types
            ),
            "Property statistics dictionary required!",
        )
        log.log_assert(
            all(
                [
                    np.all(np.isin(self.tgt_types, list(self.prop_stats[p][src].keys())))
                    for p in self.prop_names
                ]
                for src in self.src_types
            ),
            "Target type statistics missing!",
        )
        log.log_assert(
            all(
                [
                    ["type" in self.prop_stats[p][src][tgt].keys() for p in self.prop_names]
                    for src in self.src_types
                ]
                for tgt in self.tgt_types
            ),
            "Distribution type missing!",
        )
        log.log_assert(
            all(
                [
                    [
                        np.all(
                            np.isin(
                                self.distribution_attributes[self.prop_stats[p][src][tgt]["type"]],
                                list(self.prop_stats[p][src][tgt].keys()),
                            )
                        )
                        for p in self.prop_names
                    ]
                    for src in self.src_types
                ]
                for tgt in self.tgt_types
            ),
            f"Distribution attributes missing (required: {self.distribution_attributes})!",
        )
        log.log_assert(
            all(
                [
                    [
                        np.all(
                            [
                                len(self.prop_stats[p][src][tgt][a]) > 0
                                for a in self.distribution_attributes[
                                    self.prop_stats[p][src][tgt]["type"]
                                ]
                                if hasattr(self.prop_stats[p][src][tgt][a], "__iter__")
                            ]
                        )
                        for p in self.prop_names
                    ]
                    for src in self.src_types
                ]
                for tgt in self.tgt_types
            ),
            f"Distribution attribute(s) empty (required: {self.distribution_attributes})!",
        )
        log.log_assert(
            all(
                [
                    [
                        np.isclose(np.sum(self.prop_stats[p][src][tgt]["p"]), 1.0)
                        for p in self.prop_names
                        if "p" in self.prop_stats[p][src][tgt].keys()
                    ]
                    for src in self.src_types
                ]
                for tgt in self.tgt_types
            ),
            'Probability attribute "p" does not sum to 1.0!',
        )
        log.log_assert(
            all(
                [
                    [
                        len(self.prop_stats[p][src][tgt]["p"])
                        == len(self.prop_stats[p][src][tgt]["val"])
                        for p in self.prop_names
                        if np.all(np.isin(["p", "val"], list(self.prop_stats[p][src][tgt].keys())))
                    ]
                    for src in self.src_types
                ]
                for tgt in self.tgt_types
            ),
            'Probability attribute "p" does not match length of corresponding "val"!',
        )
        log.log_assert(
            np.all(
                [
                    [
                        [
                            self.prop_stats[p][src][tgt]["lower_bound"]
                            <= self.prop_stats[p][src][tgt]["upper_bound"]
                            for p in self.prop_names
                            if "lower_bound" in self.prop_stats[p][src][tgt].keys()
                            and "upper_bound" in self.prop_stats[p][src][tgt].keys()
                        ]
                        for src in self.src_types
                    ]
                    for tgt in self.tgt_types
                ]
            ),
            "Data bounds error!",
        )

    def get_prop_names(self):
        """Return list of connection/synapse property names."""
        return self.prop_names

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
    def draw_from_distribution(
        distr_type,
        distr_mean=None,
        distr_std=None,
        distr_min=None,
        distr_max=None,
        distr_val=None,
        distr_p=None,
        size=1,
    ):
        """Draw value(s) from given distribution"""
        if distr_type == "constant":
            drawn_values = np.full(size, distr_mean)
        elif distr_type == "normal":
            log.log_assert(
                distr_mean is not None and distr_std is not None,
                "Distribution parameter missing (required: mean/std)!",
            )
            drawn_values = np.random.normal(loc=distr_mean, scale=distr_std, size=size)
        elif distr_type == "truncnorm":
            log.log_assert(
                distr_mean is not None
                and distr_std is not None
                and distr_min is not None
                and distr_max is not None,
                "Distribution missing (required: mean/std/min/max)!",
            )
            log.log_assert(distr_min <= distr_max, "Range error (truncnorm)!")
            if distr_std > 0.0:
                drawn_values = truncnorm(
                    a=(distr_min - distr_mean) / distr_std,
                    b=(distr_max - distr_mean) / distr_std,
                    loc=distr_mean,
                    scale=distr_std,
                ).rvs(size=size)
            else:
                drawn_values = np.minimum(
                    np.maximum(np.full(size, distr_mean), distr_min), distr_max
                )
        elif distr_type == "gamma":
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
            log.log_assert(
                distr_mean is not None, "Distribution parameter missing (required: mean)!"
            )
            log.log_assert(distr_mean >= 0.0, "Range error (poisson)!")
            drawn_values = np.random.poisson(lam=distr_mean, size=size)
        elif distr_type == "discrete":
            drawn_values = np.random.choice(distr_val, size=size, p=distr_p)
        else:
            log.log_assert(False, f'Distribution type "{distr_type}" not supported!')
        return drawn_values

    def draw(self, prop_name, src_type, tgt_type, size=1):
        """Draw value(s) for given property name of a single connection

        (or multiple connections, if prop_name==N_SYN_PER_CONN_NAME)
        """
        stats_dict = self.prop_stats.get(prop_name)

        distr_type = stats_dict[src_type][tgt_type].get("type")
        mean_val = stats_dict[src_type][tgt_type].get("mean", np.nan)
        std_val = stats_dict[src_type][tgt_type].get("std", 0.0)
        std_within = stats_dict[src_type][tgt_type].get("std-within", 0.0)
        min_val = stats_dict[src_type][tgt_type].get("min", -np.inf)
        max_val = stats_dict[src_type][tgt_type].get("max", np.inf)
        distr_val = stats_dict[src_type][tgt_type].get("val", [])
        distr_p = stats_dict[src_type][tgt_type].get("p", [])

        if prop_name == N_SYN_PER_CONN_NAME:  # Draw <size> N_SYN_PER_CONN_NAME value(s)
            drawn_values = np.maximum(
                np.round(
                    self.draw_from_distribution(
                        distr_type, mean_val, std_val, min_val, max_val, distr_val, distr_p, size
                    )
                ).astype(int),
                1,
            )  # At least one synapse/connection, otherwise no connection!!
        else:
            conn_mean = self.draw_from_distribution(
                distr_type, mean_val, std_val, min_val, max_val, distr_val, distr_p, 1
            )  # Draw connection mean
            if std_within > 0.0 and size > 0:
                drawn_values = self.draw_from_distribution(
                    distr_type, conn_mean, std_within, min_val, max_val, distr_val, distr_p, size
                )  # Draw property values for synapses within connection
            else:
                drawn_values = np.full(size, conn_mean)  # No within-connection variability

        # Apply upper/lower bounds (optional)
        lower_bound = stats_dict[src_type][tgt_type].get("lower_bound")
        upper_bound = stats_dict[src_type][tgt_type].get("upper_bound")
        if lower_bound is not None:
            drawn_values = np.maximum(drawn_values, lower_bound)
        if upper_bound is not None:
            drawn_values = np.minimum(drawn_values, upper_bound)

        # Set data type (optional)
        data_type = stats_dict[src_type][tgt_type].get("dtype")
        if data_type is not None:
            if data_type == "int":
                drawn_values = np.round(drawn_values).astype(data_type)
            else:
                drawn_values = drawn_values.astype(data_type)

        return drawn_values

    def get_model_output(self, **kwargs):
        """Draw property values for one connection between src_type and tgt_type, returning a dataframe [seeded through numpy]."""
        syn_props = [p for p in self.prop_names if p != N_SYN_PER_CONN_NAME]
        n_syn = self.draw(N_SYN_PER_CONN_NAME, kwargs["src_type"], kwargs["tgt_type"], 1)[0]

        df = pd.DataFrame([], index=range(n_syn), columns=syn_props)
        for p in syn_props:
            df[p] = self.draw(p, kwargs["src_type"], kwargs["tgt_type"], n_syn)
        return df

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
        model_str = f"{self.__class__.__name__}\n"
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

    def get_model_output(self, **kwargs):
        """Return (constant) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
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

    def get_model_output(self, **kwargs):
        """Return (distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
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
            log.log_assert(
                self.exp_fct(test_distance, 1.0, self.prox_exp, self.prox_exp_pow)
                < self.exp_fct(test_distance, 1.0, self.dist_exp, 1.0),
                f"Proximal exponential must decay faster than distal exponential ({self.get_param_dict()})!",
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

    def get_model_output(self, **kwargs):
        """Return (distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
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
                isinstance(self.bip_coord, int) and self.bip_coord >= 0,
                'Bipolar coordinate "bip_coord" must be a non-negative integer!',
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
    def compute_bip_matrix(src_pos, tgt_pos, bip_coord=2):
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

    def get_model_output(self, **kwargs):
        """Return (bipolar distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        bip_mat = self.compute_bip_matrix(src_pos, tgt_pos, self.bip_coord)
        return self.get_conn_prob(dist_mat, bip_mat)

    def __str__(self):
        """Return model string describing the model."""
        coord_nr = self.bip_coord + 1
        coord_str = f'{coord_nr}{"st" if coord_nr == 1 else "nd" if coord_nr == 2 else "rd" if coord_nr == 3 else "th"}'
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
            + f"  d...distance, delta...difference (tgt minus src) in {coord_str} coordinate"
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
                isinstance(self.bip_coord, int) and self.bip_coord >= 0,
                'Bipolar coordinate "bip_coord" must be a non-negative integer!',
            )
            test_distance = 1000.0
            log.log_assert(
                self.exp_fct(test_distance, 1.0, self.prox_exp_P, self.prox_exp_pow_P)
                < self.exp_fct(test_distance, 1.0, self.dist_exp_P, 1.0),
                f"Proximal (P) exponential must decay faster than distal (P) exponential ({self.get_param_dict()})!",
            )
            log.log_assert(
                self.exp_fct(test_distance, 1.0, self.prox_exp_N, self.prox_exp_pow_N)
                < self.exp_fct(test_distance, 1.0, self.dist_exp_N, 1.0),
                f"Proximal (N) exponential must decay faster than distal (N) exponential ({self.get_param_dict()})!",
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
    def compute_bip_matrix(src_pos, tgt_pos, bip_coord=2):
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

    def get_model_output(self, **kwargs):
        """Return (bipolar distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        bip_mat = self.compute_bip_matrix(src_pos, tgt_pos, self.bip_coord)
        return self.get_conn_prob(dist_mat, bip_mat)

    def __str__(self):
        """Return model string describing the model."""
        coord_nr = self.bip_coord + 1
        # pylint: disable=unused-variable
        coord_str = f'{coord_nr}{"st" if coord_nr == 1 else "nd" if coord_nr == 2 else "rd" if coord_nr == 3 else "th"}'
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
            + "  d...distance, delta...difference (tgt minus src) in {coord_str} coordinate"
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

    def get_model_output(self, **kwargs):
        """Return (offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
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
    param_names = []
    param_defaults = {}
    data_names = ["p_conn_table"]
    input_names = ["src_pos", "tgt_pos"]

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
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
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dr/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in y coordinate
        dr_mat = np.sqrt(
            dx_mat**2 + dy_mat**2
        )  # Relative offset in x/y plane (Euclidean distance)
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in z coordinate
        return dr_mat, dz_mat

    def get_model_output(self, **kwargs):
        """Return (offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        dr_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
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
        model_str = model_str + "  dr/dz...radial/axial position offset (tgt minus src)"
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

    def get_model_output(self, **kwargs):
        """Return (position- & offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
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
    def compute_position_matrix(src_pos, tgt_pos):
        """Computes z position matrix of src/PRE neurons (src/PRE neuron positions repeated over tgt/POST neuron number)."""
        z_mat = np.tile(src_pos[:, 2:3], [1, tgt_pos.shape[0]])
        return z_mat

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dr/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in y coordinate
        dr_mat = np.sqrt(
            dx_mat**2 + dy_mat**2
        )  # Relative offset in x/y plane (Euclidean distance)
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing="ij"), axis=0)[
            0, :, :
        ]  # Relative difference in z coordinate
        return dr_mat, dz_mat

    def get_model_output(self, **kwargs):
        """Return (position- & offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs["src_pos"]
        tgt_pos = kwargs["tgt_pos"]
        #         log.log_assert(
        #             src_pos.shape[1] == tgt_pos.shape[1],
        #             "Dimension mismatch of source/target neuron positions!",
        #         )
        #         log.log_assert(
        #             src_pos.shape[1] == 3, "Wrong number of input dimensions (3 required)!"
        #         )
        z_mat = self.compute_position_matrix(src_pos, tgt_pos)
        dr_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
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
            + "  z...axial src position, dr/dz...radial/axial position offset (tgt minus src)"
        )
        return model_str
