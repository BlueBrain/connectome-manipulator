"""
Definition and mapping of model types to classes
"""

from abc import ABCMeta, abstractmethod
import jsonpickle
import numpy as np
import os
import pandas as pd
from scipy.stats import truncnorm
import sys

class AbstractModel(metaclass=ABCMeta):
    """Abstract base class for different types of models."""

    ###########################################################################
    # Abstract properties/methods to be defined in specific model subclasses
    @property
    @abstractmethod
    def param_names(self):
        """Names of model parameters which are part of the model."""
        pass

    @property
    @abstractmethod
    def data_names(self):
        """Names of model data frames which are part of the model."""
        pass

    @property
    @abstractmethod
    def input_names(self):
        """Names of model inputs which are part of the model."""
        pass

    @abstractmethod
    def get_model_output(self, **kwargs):
        """Abstract method for returning model output given its model inputs."""
        pass

    @abstractmethod
    def get_model_str(self):
        """Abstract method for returning a model string describing the model."""
        pass
    #
    ###########################################################################

    @staticmethod
    def model_from_file(model_file):
        """Wrapper function to load model object from file."""
        assert os.path.exists(model_file), f'ERROR: Model file "{model_file}" not found!'
        with open(model_file, 'r') as f:
            model_dict = jsonpickle.decode(f.read())
        assert 'model' in model_dict, 'ERROR: Model type not found!'

        model_type = model_dict['model']
        model_class = getattr(sys.modules[__class__.__module__], model_type) # Get model subclass
        model = model_class(model_file=model_file) # Initialize model object

        return model

    def __init__(self, **kwargs):
        """Model initialization from file or kwargs."""
        if 'model_file' in kwargs: # Load model from file [must be of same type/class]
            model_file = kwargs.pop('model_file')
            self.load_model(model_file)
        else: # Initialize directly from kwargs
            self.init_params(kwargs)
            self.init_data(kwargs)
        if len(kwargs) > 0:
            print(f'WARNING: Unused parameter(s): {set(kwargs.keys())}!')

    def init_params(self, model_dict):
        """Initialize model parameters from dict (removing used keys from dict)."""
        assert np.all([p in model_dict for p in self.param_names]), f'ERROR: Missing parameters for model initialization! Must contain initialization for {set(self.param_names)}.'
        for p in self.param_names:
            val = np.array(model_dict.pop(p)).tolist() # Convert to numpy and back to list, so that reduced to basic (non-numpy) data types
            setattr(self, p, val)

    def init_data(self, data_dict):
        """Initialize data frames with supplementary model data from dict (removing used keys from dict)."""
        assert np.all([d in data_dict for d in self.data_names]), f'ERROR: Missing data for model initialization! Must contain initialization for {set(self.data_names)}.'
        assert np.all([isinstance(data_dict[d], pd.DataFrame) for d in self.data_names]), 'ERROR: Model data must be Pandas dataframes!'
        for d in self.data_names:
            setattr(self, d, data_dict.pop(d))

    def apply(self, **kwargs):
        """Main method for applying model, i.e., returning model output given its model inputs.
           [Calls get_model_output() which must be implemented in specific model subclass!]"""
        assert np.all([inp in kwargs for inp in self.input_names]), f'ERROR: Missing model inputs! Must contain input values for {set(self.input_names)}.'
        inp_dict = {inp: kwargs.pop(inp) for inp in self.input_names}
        if len(kwargs) > 0:
            print(f'WARNING: Unused input(s): {set(kwargs.keys())}!')
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
        model_file = os.path.join(model_path, model_name + '.json')

        # Save supplementary model data (if any) to .h5 data file
        data_dict = self.get_data_dict()
        assert np.all([isinstance(v, pd.DataFrame) for k, v in data_dict.items()]), 'ERROR: Model data must be Pandas dataframes!'
        data_file = os.path.splitext(model_file)[0] + '.h5'
        for idx, (key, df) in enumerate(data_dict.items()):
            df.to_hdf(data_file, key, append=idx > 0)
        model_dict['data_keys'] = list(data_dict.keys())

        # Save model dict to .json file [using jsonpickle, so also non-serializable objects can be stored!]
        model_dict['model'] = self.__class__.__name__
        model_dict['__version_info__'] = {'python': sys.version, 'jsonpickle': jsonpickle.__version__, 'pandas': pd.__version__}
        with open(model_file, 'w') as f:
            f.write(jsonpickle.encode(model_dict, indent=2))

    def load_model(self, model_file):
        """Load model from file: Model dict from .json, model data (if any) from supplementary .h5 data file."""
        assert os.path.exists(model_file), f'ERROR: Model file "{model_file}" not found!'
        with open(model_file, 'r') as f:
            model_dict = jsonpickle.decode(f.read())

        assert 'model' in model_dict and model_dict.pop('model') == self.__class__.__name__, 'ERROR: Model type mismatch!'
        self.init_params(model_dict)
        data_keys = model_dict.pop('data_keys')
        unused_params = [k for k in model_dict.keys() if k.find('__') != 0] # Unused paramters, excluding meta data ('__<name>') that may be included in file
        if len(unused_params) > 0:
            print(f'WARNING: Unused parameter(s) in model file: {set(unused_params)}!')

        # Load supplementary model data (if any) from .h5 data file [same name and folder as .json file]
        if len(data_keys) > 0:
            data_file = os.path.splitext(model_file)[0] + '.h5'
            assert os.path.exists(data_file), f'ERROR: Data file "{data_file}" missing!'
            data_dict = {key: pd.read_hdf(data_file, key) for key in data_keys}
            self.init_data(data_dict)
            if len(data_dict) > 0:
                print(f'WARNING: Unused data frame(s) in model data file: {set(data_dict.keys())}!')


### MODEL TEMPLATE ###
# class TemplateModel(AbstractModel):
#     """ <Template> model:
#         -Model details...
#     """
# 
#     # Names of model inputs, parameters and data frames which are part if this model
#     param_names = [...]
#     data_names = [...]
#     input_names = [...]
# 
#     def __init__(self, **kwargs):
#         """Model initialization."""
#         super().__init__(**kwargs)
# 
#         # Check parameters
#         assert ...
# 
#     # <Additional access methods, if needed>
#     ...
# 
#     def get_model_output(self, **kwargs):
#         """Description..."""
#         # MUST BE IMPLEMENTED
#         return ...
# 
#     def get_model_str(self):
#         """Return model string describing the model."""
#         model_str = f'{self.__class__.__name__}\n'
#         model_str = model_str + ...
#         return model_str


class LinDelayModel(AbstractModel):
    """ Linear distance-dependent delay model [generative model]:
        -Delay mean: delay_mean_coefs[1] * distance + delay_mean_coefs[0] (linear)
        -Delay std: delay_std (constant)
        -Delay min: delay_min (constant)
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = ['delay_mean_coefs', 'delay_std', 'delay_min']
    data_names = []
    input_names = ['distance']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check paramters
        assert hasattr(self.delay_mean_coefs, '__iter__') and len(self.delay_mean_coefs) == 2, 'ERROR: Two mean coefficients required for linear delay model!'
        assert self.delay_std > 0.0, 'ERROR: Delay std must be larger than zero!'
        assert self.delay_min >= 0.0, 'ERROR: Delay min cannot be negative!'

    def get_mean(self, distance):
        """Get delay mean for given distance (linear)."""
        return self.delay_mean_coefs[1] * distance + self.delay_mean_coefs[0]

    def get_std(self, distance):
        """Get delay std for given distance (constant)."""
        return np.full_like(distance, self.delay_std, dtype=type(self.delay_std))

    def get_min(self, distance):
        """Get delay min for given distance (constant)."""
        return np.full_like(distance, self.delay_min, dtype=type(self.delay_min))

    def get_model_output(self, **kwargs):
        """Draw distance-dependent delay values from truncated normal distribution [seeded through numpy]."""
        d_mean = self.get_mean(np.array(kwargs['distance']))
        d_std = self.get_std(np.array(kwargs['distance']))
        d_min = self.get_min(np.array(kwargs['distance']))
        return truncnorm(a=(d_min - d_mean) / d_std, b=np.inf, loc=d_mean, scale=d_std).rvs()

    def get_model_str(self):
        """Return model string describing the model."""
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  Delay mean: {self.delay_mean_coefs[1]:.3f} * distance + {self.delay_mean_coefs[0]:.3f}\n'
        model_str = model_str + f'  Delay std: {self.delay_std:.3f} (constant)\n'
        model_str = model_str + f'  Delay min: {self.delay_min:.3f} (constant)'
        return model_str


class PosMapModel(AbstractModel):
    """ Position mapping model, mapping one coordinate system to another for a given set of neurons:
        -Mapped neuron position: pos_table.loc[gids] (lookup-table)
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = []
    data_names = ['pos_table']
    input_names = ['gids']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

    def get_gids(self):
        """Return GIDs that are mapped within this model."""
        return self.pos_table.index.values

    def get_model_output(self, **kwargs):
        """Return (mapped) neuron positions for a given set of GIDs."""
        gids = np.array(kwargs['gids'])
        return self.pos_table.loc[gids].to_numpy()

    def get_model_str(self):
        """Return model string describing the model."""
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  Size: {self.pos_table.shape[0]} GIDs\n'
        model_str = model_str + f'  Outputs: {self.pos_table.shape[1]} ({", ".join(self.pos_table.keys())})\n'
        model_str = model_str +  '  Range: ' + ', '.join([f'{k}: {self.pos_table[k].min():.1f}..{self.pos_table[k].max():.1f}' for k in self.pos_table.keys()])
        return model_str

    
class ConnPropsModel(AbstractModel):
    """ Connection/synapse properties model for pairs of m-types [generative model]:
        -Connection/synapse property values drawn from given distributions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = ['src_types', 'tgt_types', 'prop_names', 'prop_stats']
    data_names = []
    input_names = ['src_type', 'tgt_type']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert isinstance(self.prop_stats, dict), 'ERROR: "prop_stats" dictionary required!'
        assert 'n_syn_per_conn' in self.prop_names, 'ERROR: "n_syn_per_conn" missing'
        assert np.all(np.isin(self.prop_names, list(self.prop_stats.keys()))), 'ERROR: Property name/statistics mismatch!'
        assert np.all([isinstance(self.prop_stats[p], dict) for p in self.prop_names]), 'ERROR: Property statistics dictionary required!'
        assert np.all([np.all(np.isin(self.src_types, list(self.prop_stats[p].keys()))) for p in self.prop_names]), 'ERROR: Source type statistics missing!'
        assert np.all([[isinstance(self.prop_stats[p][src], dict) for p in self.prop_names] for src in self.src_types]), 'ERROR: Property statistics dictionary required!'
        assert np.all([[np.all(np.isin(self.tgt_types, list(self.prop_stats[p][src].keys()))) for p in self.prop_names] for src in self.src_types]), 'ERROR: Target type statistics missing!'
        required_keys = ['type', 'mean', 'std'] # Required keys to be specified for each distribution
        assert np.all([[[np.all(np.isin(required_keys, list(self.prop_stats[p][src][tgt].keys()))) for p in self.prop_names] for src in self.src_types] for tgt in self.tgt_types]), f'ERROR: Distribution attributes missing (required: {required_keys})!'

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

    def draw(self, prop_name, src_type, tgt_type, size=1):
        """Draw value(s) for given property name"""
        stats_dict = self.prop_stats.get(prop_name)
        distr_type = stats_dict[src_type][tgt_type].get('type')
        if distr_type == 'constant':
            drawn_values = np.full(size, stats_dict[src_type][tgt_type]['mean'])
        elif distr_type == 'normal':
            drawn_values = np.random.normal(loc=stats_dict[src_type][tgt_type]['mean'], scale=stats_dict[src_type][tgt_type]['std'], size=size)
        elif distr_type == 'truncnorm':
            min_val = stats_dict[src_type][tgt_type].get('min', -np.inf)
            max_val = stats_dict[src_type][tgt_type].get('max', np.inf)
            drawn_values = truncnorm(a=(min_val - stats_dict[src_type][tgt_type]['mean']) / stats_dict[src_type][tgt_type]['std'],
                                     b=(max_val - stats_dict[src_type][tgt_type]['mean']) / stats_dict[src_type][tgt_type]['std'],
                                     loc=stats_dict[src_type][tgt_type]['mean'], scale=stats_dict[src_type][tgt_type]['std']).rvs(size=size)
        elif distr_type == 'gamma':
            drawn_values = np.random.gamma(shape=stats_dict[src_type][tgt_type]['mean']**2 / stats_dict[src_type][tgt_type]['std']**2,
                                           scale=stats_dict[src_type][tgt_type]['std']**2 / stats_dict[src_type][tgt_type]['mean'], size=size)
        elif distr_type == 'poisson':
            drawn_values = np.random.poisson(lam=stats_dict[src_type][tgt_type]['mean'], size=size)
        else:
            assert False, f'ERROR: Distribution type "{distr_type}" not supported!'

        # Apply upper/lower bounds (optional)
        lower_bound = stats_dict[src_type][tgt_type].get('lower_bound')
        upper_bound = stats_dict[src_type][tgt_type].get('upper_bound')
        if lower_bound is not None:
            drawn_values = np.maximum(drawn_values, lower_bound)
        if upper_bound is not None:
            drawn_values = np.minimum(drawn_values, upper_bound)

        # Set data type (optional)
        data_type = stats_dict[src_type][tgt_type].get('dtype')
        if data_type is not None:
            if data_type == 'int':
                drawn_values = np.round(drawn_values).astype(data_type)
            else:
                drawn_values = drawn_values.astype(data_type)

        return drawn_values

    def get_model_output(self, **kwargs):
        """Draw property values for a connection between src_type and tgt_type, returning a dataframe [seeded through numpy]."""
        syn_props = [p for p in self.prop_names if p != 'n_syn_per_conn']
        n_syn = self.draw('n_syn_per_conn', kwargs['src_type'], kwargs['tgt_type'], 1)[0]
        
        df = pd.DataFrame([], index=range(n_syn), columns=syn_props)
        for p in syn_props:
            df[p] = self.draw(p, kwargs['src_type'], kwargs['tgt_type'], n_syn)
        return df

    def get_model_str(self):
        """Return model string describing the model."""
        distr_types = {p: '/'.join(np.unique([[self.prop_stats[p][src][tgt]["type"] for src in self.src_types] for tgt in self.tgt_types])) for p in self.prop_names} # Extract distribution types
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  Connection/synapse property distributions between {len(self.src_types)}x{len(self.tgt_types)} M-types:\n'
        model_str = model_str + '  ' + '; '.join([f'{p}: {distr_types[p]}' for p in self.prop_names])
        return model_str
