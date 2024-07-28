from transformers.utils import PushToHubMixin
from ..generation.ct2_utils import Ct2GenerationMixin




class Ct2PreTrainedModel(PushToHubMixin, Ct2GenerationMixin):
    """
    Base class for all models.

    [`Ct2PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    pass
