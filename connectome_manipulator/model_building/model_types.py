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

    @abstractmethod
    def apply(self, **kwargs):
        """Abstract method for returning model output given its model inputs."""
        pass
    #
    ###########################################################################

    def __init__(self, **kwargs):
        """Model initialization from file or kwargs."""
        if 'model_file' in kwargs: # Load model from file [must be of same type/class]
            model_file = kwargs.pop('model_file')
            model_dict, data_dict = self.load_model(model_file)
            assert 'model' in model_dict and model_dict['model'] == self.__class__.__name__, 'ERROR: Model type mismatch!'
            self.init_params(model_dict)
            self.init_data(data_dict)
        else: # Initialize directly from kwargs
            self.init_params(kwargs)
            self.init_data(kwargs)
        if len(kwargs) > 0:
            print(f'WARNING: Unused parameter(s): {set(kwargs.keys())}!')

    def init_params(self, model_dict):
        """Initialize model parameters from dict (removing used keys from dict)."""
        assert np.all([p in model_dict for p in self.param_names]), f'ERROR: Missing parameters for model initialization! Must contain initialization for {set(self.param_names)}.'
        for p in self.param_names:
            setattr(self, p, model_dict.pop(p))

    def init_data(self, data_dict):
        """Initialize data frames with supplementary model data from dict (removing used keys from dict)."""
        assert np.all([d in data_dict for d in self.data_names]), f'ERROR: Missing data for model initialization! Must contain initialization for {set(self.data_names)}.'
        assert np.all([isinstance(v, pd.DataFrame) for k, v in data_dict.items()]), 'ERROR: Model data must be Pandas dataframes!'
        for d in self.data_names:
            setattr(self, d, data_dict.pop(d))

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

        # Load supplementary model data (if any) from .h5 data file [same name and folder as .json file]
        data_file = os.path.splitext(model_file)[0] + '.h5'
        assert os.path.exists(data_file), f'ERROR: Data file "{data_file}" missing!'
        data_dict = {key: pd.read_hdf(data_file, key) for key in model_dict['data_keys']}

        return model_dict, data_dict


class LinDelayModel(AbstractModel):
    """ Linear distance-dependent delay model:
        -Delay mean: delay_mean_coefs[1] * dist + delay_mean_coefs[0] (linear)
        -Delay std: delay_std (constant)
        -Delay min: delay_min (constant)
    """

    # Names of model parameters and data frames which are part if this model
    param_names = ['delay_mean_coefs', 'delay_std', 'delay_min']
    data_names = []

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
        return self.delay_std

    def get_min(self, distance):
        """Get delay min for given distance (constant)."""
        return self.delay_min

    def apply(self, **kwargs):
        """Draw distance-dependent delay values from truncated normal distribution [seeded through numpy]."""
        d_mean = self.get_mean(kwargs['distance'])
        d_std = self.get_std(kwargs['distance'])
        d_min = self.get_min(kwargs['distance'])
        return truncnorm(a=(d_min - d_mean) / d_std, b=np.inf, loc=d_mean, scale=d_std).rvs()
