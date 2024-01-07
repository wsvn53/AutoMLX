import json
import time

from auto_mlx import AutoMLXForCausalLM
from transformers import AutoTokenizer
from mlx import core as mx


class TestAutoMLXForCausalLM:
    @classmethod
    def test_from_pretrained(cls):
        model_path = "models/Refact-1_6B-fim-mlx"
        model = AutoMLXForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # encode prompt
        prompt = "write a snake game in python"
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

        for i in range(10):
            start_time = time.time()
            outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.2)
            print(f"Generated output:\n", tokenizer.decode(outputs[0]), f", timecost: {time.time()-start_time}")
            print(f"Token count: {len(inputs['input_ids'][0])}, {len(outputs[0])}")


if __name__ == '__main__':
    # Test AutoMLXForCausalLM.test_from_pretrained()
    TestAutoMLXForCausalLM.test_from_pretrained()
