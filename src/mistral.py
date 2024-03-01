from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" # the device to load the model onto

from mlx_lm import load, generate

model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.2")

response = generate(model, tokenizer, prompt="hello", verbose=True)