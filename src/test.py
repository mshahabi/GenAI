from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    print("Attempting to use MPS...")
    mps_device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",use_fast=True)
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
      "mistralai/Mistral-7B-Instruct-v0.2",
      trust_remote_code=True,low_cpu_mem_usage=True
    )
    model.to(mps_device)
    
    inputs = tokenizer(
      "\n###Instruction\n\nGenerate a python function to find number of CPU cores\n\n###Response\n",
      return_tensors="pt",
      return_token_type_ids=False,
    ).to(mps_device)
    tokens = model.generate(
      **inputs,
      max_new_tokens=48,
      temperature=0.2,
      do_sample=True,
      pad_token_id=50256,
      streamer=streamer
    )

    print(tokenizer.decode(tokens[0], skip_special_tokens=True))