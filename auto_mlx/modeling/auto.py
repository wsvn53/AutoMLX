import json
import types
import numpy as np
import torch

import mlx
from mlx import core as mx
from mlx.utils import tree_unflatten

from ._utils import *


def sample_logits(logits, temperature=1.0, top_p=1.0, top_k=-1):
    # TODO: support top_k and top_p
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature), axis=-1)


def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        streamer=None,
        **kwargs
):
    # convert torch tokens into mx.array
    input_ids = input_ids.cpu().numpy() if not isinstance(input_ids, np.ndarray) else input_ids
    input_ids = mx.array(input_ids, dtype=mx.int32)

    past_key_values = None
    output_tokens = []

    while True:
        input_ids = input_ids if past_key_values is None else input_ids[:, None]
        outputs = self(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        input_ids = sample_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        output_tokens.append(input_ids.item())
        if input_ids.item() == self.end_token_id or len(output_tokens) >= max_length:
            if streamer is not None:
                streamer.end()
            break

        if streamer is not None:
            streamer.put(torch.tensor(input_ids.tolist()))

    return (output_tokens,)


class AutoMLXForCausalLM:
    def __init__(self):
        raise EnvironmentError("AutoMLXForCausalLM can only be used as class and not as instance.")

    @classmethod
    def load_model(
            cls,
            model_name_or_path,
            model_class
    ):
        # load model with cache disabled
        mlx.core.metal.set_cache_enabled(False)

        config_file = get_mlx_config_file(model_name_or_path)
        weight_file = get_mlx_weight_file(model_name_or_path)
        with open(config_file, "r") as f:
            config = json.load(f)
        model = model_class(config)
        weights = mx.load(weight_file)
        model.update(tree_unflatten(list(weights.items())))
        model.eval()

        # Add additional attributes to model
        model.device = "mps"
        model.eos_token_id = config["eos_token_id"] if "eos_token_id" in config else 0
        model.end_token_id = model.eos_token_id
        model.bos_token_id = config["bos_token_id"] if "bos_token_id" in config else None
        model.pad_token_id = config["pad_token_id"] if "pad_token_id" in config else None

        # Add additional methods to model
        model.generate = types.MethodType(generate, model)

        return model

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            trust_remote_code=False,
    ):
        model_type, model_class = get_model_type_class(model_name_or_path, trust_remote_code=trust_remote_code)
        model_class = import_model_class(model_name_or_path, model_class)
        return cls.load_model(model_name_or_path, model_class)


__all__ = ["AutoMLXForCausalLM"]
