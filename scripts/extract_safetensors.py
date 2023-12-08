from safetensors import safe_open
import torch
import sys

model_filename = sys.argv[1]
embeddings_var = sys.argv[2]

with safe_open(model_filename, framework="pt", device="cpu") as model:
    if embeddings_var in model.keys():
        embeddings = model.get_tensor(embeddings_var)

        torch.save(embeddings, "embeddings.pt")

    else:
        print(f"not found: {embeddings_var}")
        print(model.keys())
