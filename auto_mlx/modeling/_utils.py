import os
import sys
from importlib import import_module

from transformers import AutoConfig
from ._const import MLX_SUPPORTED_MODEL_TYPES


def get_model_config(model_path, trust_remote_code=False):
    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def get_model_type_class(model_path, trust_remote_code=False):
    model_config = get_model_config(model_path, trust_remote_code=trust_remote_code)
    # Check supported models, otherwise raise error
    if model_config.model_type not in MLX_SUPPORTED_MODEL_TYPES:
        raise ValueError(f"AutoMLX: Model type {model_config.model_type} is not supported.")

    # Get CausalLM class from this model
    model_class = None
    if model_config.auto_map is not None and "AutoModelForCausalLM" in model_config.auto_map:
        model_class = model_config.auto_map["AutoModelForCausalLM"]

    return model_config.model_type, model_class


def get_model_eos_token(model_path, trust_remote_code=False):
    model_config = get_model_config(model_path, trust_remote_code=trust_remote_code)
    if "eos_token_id" not in model_config:
        return 0
    return model_config.eos_token_id


def import_model_class(model_path, model_class):
    if model_path not in sys.path:
        sys.path.append(model_path)
    # split model_class into class
    class_file, class_name = tuple(model_class.split("."))
    # import class
    module = import_module(class_file)
    return getattr(module, class_name)


def get_mlx_weight_file(model_path):
    # by default, mlx weight file is saved in model_path/mlx_models/weights.npz
    return os.path.join(model_path, "mlx_model", "weights.npz")


def get_mlx_config_file(model_path):
    # by default, mlx config file is saved in model_path/mlx_models/config.json
    return os.path.join(model_path, "mlx_model", "config.json")