"""
Definition and mapping of model types to classes
"""

from abc import ABCMeta, abstractmethod
import jsonpickle
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interpn
from scipy.spatial import distance_matrix
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
    param_names = ['src_types', 'tgt_types', 'prop_stats']
    data_names = []
    input_names = ['src_type', 'tgt_type']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        def dict_conv(data):
            """ Recursively convert numpy to basic data types, to have a clean JSON file """
            if isinstance(data, dict):
                return {k: dict_conv(v) for k, v in data.items()}
            elif isinstance(data, list) or isinstance(data, tuple):
                return [dict_conv(d) for d in data]
            elif hasattr(data, 'tolist'): # Convert numpy types
                return data.tolist()
            else:
                return data

        # Check & convert parameters
        assert isinstance(self.prop_stats, dict), 'ERROR: "prop_stats" dictionary required!'
        self.prop_stats = dict_conv(self.prop_stats) # Convert dict to basic data types
        self.prop_names = self.prop_stats.keys()
        assert 'n_syn_per_conn' in self.prop_names, 'ERROR: "n_syn_per_conn" missing'
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

    @staticmethod
    def draw_from_distribution(distr_type, distr_mean, distr_std=None, distr_min=None, distr_max=None, size=1):
        """Draw value(s) from given distribution"""
        if distr_type == 'constant':
            drawn_values = np.full(size, distr_mean)
        elif distr_type == 'normal':
            assert distr_mean is not None and distr_std is not None, 'ERROR: Distribution parameter missing (required: mean/std)!'
            drawn_values = np.random.normal(loc=distr_mean, scale=distr_std, size=size)
        elif distr_type == 'truncnorm':
            assert distr_mean is not None and distr_std is not None and distr_min is not None and distr_max is not None, 'ERROR: Distribution missing (required: mean/std/min/max)!'
            drawn_values = truncnorm(a=(distr_min - distr_mean) / distr_std,
                                     b=(distr_max - distr_mean) / distr_std,
                                     loc=distr_mean, scale=distr_std).rvs(size=size)
        elif distr_type == 'gamma':
            assert distr_mean is not None and distr_std is not None, 'ERROR: Distribution parameter missing (required: mean/std)!'
            drawn_values = np.random.gamma(shape=distr_mean**2 / distr_std**2,
                                           scale=distr_std**2 / distr_mean, size=size)
        elif distr_type == 'poisson':
            assert distr_mean is not None, 'ERROR: Distribution parameter missing (required: mean)!'
            drawn_values = np.random.poisson(lam=distr_mean, size=size)
        else:
            assert False, f'ERROR: Distribution type "{distr_type}" not supported!'
        return drawn_values

    def draw(self, prop_name, src_type, tgt_type, size=1):
        """Draw value(s) for given property name of a single connection"""
        stats_dict = self.prop_stats.get(prop_name)

        distr_type = stats_dict[src_type][tgt_type].get('type')
        mean_val = stats_dict[src_type][tgt_type]['mean']
        std_val = stats_dict[src_type][tgt_type]['std']
        std_within = stats_dict[src_type][tgt_type].get('std-within', 0.0)
        min_val = stats_dict[src_type][tgt_type].get('min', -np.inf)
        max_val = stats_dict[src_type][tgt_type].get('max', np.inf)

        conn_mean = self.draw_from_distribution(distr_type, mean_val, std_val, min_val, max_val, 1) # Draw connection mean
        if std_within > 0.0 and size > 0:
            drawn_values = self.draw_from_distribution(distr_type, conn_mean, std_within, min_val, max_val, size) # Draw property values for synapses within connection
        else:
            drawn_values = np.full(size, conn_mean) # No within-connection variability

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
        """Draw property values for one connection between src_type and tgt_type, returning a dataframe [seeded through numpy]."""
        syn_props = [p for p in self.prop_names if p != 'n_syn_per_conn']
        n_syn = self.draw('n_syn_per_conn', kwargs['src_type'], kwargs['tgt_type'], 1)[0]

        df = pd.DataFrame([], index=range(n_syn), columns=syn_props)
        for p in syn_props:
            df[p] = self.draw(p, kwargs['src_type'], kwargs['tgt_type'], n_syn)
        return df

    def get_model_str(self):
        """Return model string describing the model."""
        distr_types = {p: '/'.join(np.unique([[self.prop_stats[p][src][tgt]['type'] for src in self.src_types] for tgt in self.tgt_types])) for p in self.prop_names} # Extract distribution types
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  Connection/synapse property distributions between {len(self.src_types)}x{len(self.tgt_types)} M-types:\n'
        model_str = model_str + '  ' + '; '.join([f'{p}: {distr_types[p]}' for p in self.prop_names])
        return model_str


class ConnProb1stOrderModel(AbstractModel):
    """ 1st order connection probability model (Erdos-Renyi):
        -Returns (constant) connection probability for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = ['p_conn']
    data_names = []
    input_names = ['src_pos', 'tgt_pos']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert 0.0 <= self.p_conn <= 1.0, 'ERROR: Connection probability must be between 0 and 1!'

    def get_conn_prob(self):
        """Return (constant) connection probability."""
        return self.p_conn

    def get_model_output(self, **kwargs):
        """Return (constant) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs['src_pos']
        tgt_pos = kwargs['tgt_pos']
        assert src_pos.shape[1] == tgt_pos.shape[1], 'ERROR: Dimension mismatch of source/target neuron positions!'
        return np.full((src_pos.shape[0], tgt_pos.shape[0]), self.get_conn_prob())

    def get_model_str(self):
        """Return model string describing the model."""
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  p_conn() = {self.p_conn:.3f} (constant)'
        return model_str


class ConnProb2ndOrderExpModel(AbstractModel):
    """ 2nd order connection probability model (exponential distance-dependent):
        -Returns (distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = ['scale', 'exponent']
    data_names = []
    input_names = ['src_pos', 'tgt_pos']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert 0.0 <= self.scale <= 1.0, 'ERROR: "Scale" must be between 0 and 1!'
        assert self.exponent >= 0.0, 'ERROR: "Exponent" must be non-negative!'

    def get_conn_prob(self, distance):
        """Return (distance-dependent) connection probability."""
        return self.scale * np.exp(-self.exponent * np.array(distance))

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections
        return dist_mat

    def get_model_output(self, **kwargs):
        """Return (distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs['src_pos']
        tgt_pos = kwargs['tgt_pos']
        assert src_pos.shape[1] == tgt_pos.shape[1], 'ERROR: Dimension mismatch of source/target neuron positions!'
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        return self.get_conn_prob(dist_mat)

    def get_model_str(self):
        """Return model string describing the model."""
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  p_conn(d) = {self.scale:.3f} * exp(-{self.exponent:.3f} * d)\n'
        model_str = model_str +  '  d...distance'
        return model_str


class ConnProb3rdOrderExpModel(AbstractModel):
    """ 3rd order connection probability model (bipolar exponential distance-dependent):
        -Returns (bipolar distance-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = ['scale_P', 'scale_N', 'exponent_P', 'exponent_N', 'bip_coord']
    data_names = []
    input_names = ['src_pos', 'tgt_pos']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert 0.0 <= self.scale_P <= 1.0 and 0.0 <= self.scale_N <= 1.0, 'ERROR: "Scale" must be between 0 and 1!'
        assert self.exponent_P >= 0.0 and self.exponent_N >= 0.0, 'ERROR: "Exponent" must not be negative!'
        assert isinstance(self.bip_coord, int) and self.bip_coord >= 0, 'ERROR: Bipolar coordinate "bip_coord" must be a non-negative integer!'

    def get_conn_prob(self, distance, bip):
        """Return (bipolar distance-dependent) connection probability."""
        p_conn_N = self.scale_N * np.exp(-self.exponent_N * np.array(distance))
        p_conn_P = self.scale_P * np.exp(-self.exponent_P * np.array(distance))
        p_conn = np.select([np.array(bip) < 0.0, np.array(bip) > 0.0], [p_conn_N, p_conn_P], default=0.5 * (p_conn_N + p_conn_P))
        return p_conn

    @staticmethod
    def compute_dist_matrix(src_pos, tgt_pos):
        """Compute distance matrix between pairs of neurons."""
        dist_mat = distance_matrix(src_pos, tgt_pos)
        dist_mat[dist_mat == 0.0] = np.nan # Exclude autaptic connections
        return dist_mat

    @staticmethod
    def compute_bip_matrix(src_pos, tgt_pos, bip_coord=2):
        """Computes bipolar matrix between pairs of neurons along specified coordinate axis (default: 2..z-axis),
           defined as sign of target (POST-synaptic) minus source (PRE-synaptic) coordinate value
           (i.e., POST-synaptic neuron below (delta < 0) or above (delta > 0) PRE-synaptic neuron assuming
            axis values increasing from lower to upper layers)"""
        bip_mat = np.sign(np.diff(np.meshgrid(src_pos[:, bip_coord], tgt_pos[:, bip_coord], indexing='ij'), axis=0)[0, :, :]) # Bipolar distinction based on difference in specified coordinate
        return bip_mat

    def get_model_output(self, **kwargs):
        """Return (bipolar distance-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs['src_pos']
        tgt_pos = kwargs['tgt_pos']
        assert src_pos.shape[1] == tgt_pos.shape[1], 'ERROR: Dimension mismatch of source/target neuron positions!'
        dist_mat = self.compute_dist_matrix(src_pos, tgt_pos)
        bip_mat = self.compute_bip_matrix(src_pos, tgt_pos, self.bip_coord)
        return self.get_conn_prob(dist_mat, bip_mat)

    def get_model_str(self):
        """Return model string describing the model."""
        coord_nr = self.bip_coord + 1
        coord_str = f'{coord_nr}{"st" if coord_nr == 1 else "nd" if coord_nr == 2 else "rd" if coord_nr == 3 else "th"}'
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  p_conn(d, delta) = {self.scale_N:.3f} * exp(-{self.exponent_N:.3f} * d) if delta < 0\n'
        model_str = model_str + f'                     {self.scale_P:.3f} * exp(-{self.exponent_P:.3f} * d) if delta > 0\n'
        model_str = model_str + f'                     AVERAGE OF BOTH MODELS  if delta == 0\n'
        model_str = model_str + f'  d...distance, delta...difference (tgt minus src) in {coord_str} coordinate'
        return model_str


class ConnProb4thOrderLinInterpnModel(AbstractModel):
    """ 4th order connection probability model (offset-dependent, linearly interpolated):
        -Returns (offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = []
    data_names = ['p_conn_table']
    input_names = ['src_pos', 'tgt_pos']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert len(self.p_conn_table.index.levels) == 3, 'ERROR: Data frame with 3 index levels (dx, dy, dz) required!'
        assert self.p_conn_table.shape[1] == 1, 'ERROR: Data frame with 1 column (conn. prob.) required!'

        self.data_points = [list(lev_pos) for lev_pos in self.p_conn_table.index.levels] # Extract data offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = np.array(self.p_data.shape) > 1 # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

    def data_points(self):
        """Return data offsets."""
        return self.data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, dx, dy, dz):
        """Return (offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self.data_points) if self.data_dim_sel[idx]]
        inp_sel = [val for idx, val in enumerate([np.array(dx), np.array(dy), np.array(dz)]) if self.data_dim_sel[idx]]
        p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        return p_conn

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dx/dy/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing='ij'), axis=0)[0, :, :] # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing='ij'), axis=0)[0, :, :] # Relative difference in y coordinate
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing='ij'), axis=0)[0, :, :] # Relative difference in z coordinate
        return dx_mat, dy_mat, dz_mat

    def get_model_output(self, **kwargs):
        """Return (offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs['src_pos']
        tgt_pos = kwargs['tgt_pos']
        assert src_pos.shape[1] == tgt_pos.shape[1], 'ERROR: Dimension mismatch of source/target neuron positions!'
        assert src_pos.shape[1] == 3, 'ERROR: Wrong number of input dimensions (3 required)!'
        dx_mat, dy_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
        return self.get_conn_prob(dx_mat, dy_mat, dz_mat)

    def get_model_str(self):
        """Return model string describing the model."""
        inp_names = np.array(['dx', 'dy', 'dz'])
        inp_str = ', '.join(inp_names[self.data_dim_sel])
        range_str = ', '.join([f'{inp_names[i]}: {np.min(self.data_points[i]):.2f}..{np.max(self.data_points[i]):.2f}' for i in range(len(self.data_points)) if self.data_dim_sel[i]])
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n'
        model_str = model_str +  '  dx/dy/dz...position offset (tgt minus src) in x/y/z dimension'
        return model_str


class ConnProb5thOrderLinInterpnModel(AbstractModel):
    """ 5th order connection probability model (position- & offset-dependent, linearly interpolated):
        -Returns (position- & offset-dependent) connection probabilities for given source/target neuron positions
    """

    # Names of model inputs, parameters and data frames which are part if this model
    param_names = []
    data_names = ['p_conn_table']
    input_names = ['src_pos', 'tgt_pos']

    def __init__(self, **kwargs):
        """Model initialization."""
        super().__init__(**kwargs)

        # Check parameters
        assert len(self.p_conn_table.index.levels) == 6, 'ERROR: Data frame with 6 index levels (x, y, z, dx, dy, dz) required!'
        assert self.p_conn_table.shape[1] == 1, 'ERROR: Data frame with 1 column (conn. prob.) required!'

        self.data_points = [list(lev_pos) for lev_pos in self.p_conn_table.index.levels] # Extract data positions & offsets from multi-index
        self.p_data = self.p_conn_table.to_numpy().reshape(self.p_conn_table.index.levshape)
        self.data_dim_sel = np.array(self.p_data.shape) > 1 # Select data dimensions to be interpolated (removing dimensions with only a single data point from interpolation)

    def data_points(self):
        """Return data offsets."""
        return self.data_points

    def get_prob_data(self):
        """Return connection probability data."""
        return self.p_data

    def get_conn_prob(self, x, y, z, dx, dy, dz):
        """Return (position- & offset-dependent) connection probability, linearly interpolating between data points (except dimensions with a single data point)."""
        data_sel = [val for idx, val in enumerate(self.data_points) if self.data_dim_sel[idx]]
        inp_sel = [val for idx, val in enumerate([np.array(x), np.array(y), np.array(z), np.array(dx), np.array(dy), np.array(dz)]) if self.data_dim_sel[idx]]
        p_conn = np.minimum(np.maximum(interpn(data_sel, np.squeeze(self.p_data), np.array(inp_sel).T, method="linear", bounds_error=False, fill_value=None), 0.0), 1.0).T
        return p_conn

    @staticmethod
    def compute_position_matrices(src_pos, tgt_pos):
        """Computes x/y/z position matrices of src/PRE neurons (src/PRE neuron positions repeated over tgt/POST neuron number)."""
        x_mat, y_mat, z_mat = [np.tile(src_pos[:, i:i + 1], [1, tgt_pos.shape[0]]) for i in range(src_pos.shape[1])]
        return x_mat, y_mat, z_mat

    @staticmethod
    def compute_offset_matrices(src_pos, tgt_pos):
        """Computes dx/dy/dz offset matrices between pairs of neurons (tgt/POST minus src/PRE position)."""
        dx_mat = np.diff(np.meshgrid(src_pos[:, 0], tgt_pos[:, 0], indexing='ij'), axis=0)[0, :, :] # Relative difference in x coordinate
        dy_mat = np.diff(np.meshgrid(src_pos[:, 1], tgt_pos[:, 1], indexing='ij'), axis=0)[0, :, :] # Relative difference in y coordinate
        dz_mat = np.diff(np.meshgrid(src_pos[:, 2], tgt_pos[:, 2], indexing='ij'), axis=0)[0, :, :] # Relative difference in z coordinate
        return dx_mat, dy_mat, dz_mat

    def get_model_output(self, **kwargs):
        """Return (position- & offset-dependent) connection probabilities <#src x #tgt> for all combinations of source/target neuron positions <#src/#tgt x #dim>."""
        src_pos = kwargs['src_pos']
        tgt_pos = kwargs['tgt_pos']
        assert src_pos.shape[1] == tgt_pos.shape[1], 'ERROR: Dimension mismatch of source/target neuron positions!'
        assert src_pos.shape[1] == 3, 'ERROR: Wrong number of input dimensions (3 required)!'
        x_mat, y_mat, z_mat = self.compute_position_matrices(src_pos, tgt_pos)
        dx_mat, dy_mat, dz_mat = self.compute_offset_matrices(src_pos, tgt_pos)
        return self.get_conn_prob(x_mat, y_mat, z_mat, dx_mat, dy_mat, dz_mat)

    def get_model_str(self):
        """Return model string describing the model."""
        inp_names = np.array(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        inp_str = ', '.join(inp_names[self.data_dim_sel])
        range_str = ', '.join([f'{inp_names[i]}: {np.min(self.data_points[i]):.2f}..{np.max(self.data_points[i]):.2f}' for i in range(len(self.data_points)) if self.data_dim_sel[i]])
        model_str = f'{self.__class__.__name__}\n'
        model_str = model_str + f'  p_conn({inp_str}) = LINEAR INTERPOLATION FROM DATA TABLE ({self.p_conn_table.shape[0]} entries; {range_str})\n'
        model_str = model_str +  '  x/y/z...src position, dx/dy/dz...position offset (tgt minus src) in x/y/z dimension'
        return model_str
