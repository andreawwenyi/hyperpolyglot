import torch
import numpy as np

embeddings = np.load("embeddings.npy", allow_pickle=True)
embeddings = torch.from_numpy(embeddings)
torch.save(embeddings, "embeddings.pt")
