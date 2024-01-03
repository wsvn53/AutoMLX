import json

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
        prompt = "# print hello world\n"
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        # input_ids = mx.array(inputs["input_ids"])

        outputs = model.generate(inputs["input_ids"], max_length=10, temperature=0)
        print("Generated output:\n", tokenizer.decode(outputs[0]))


if __name__ == '__main__':
    # Test AutoMLXForCausalLM.test_from_pretrained()
    TestAutoMLXForCausalLM.test_from_pretrained()
