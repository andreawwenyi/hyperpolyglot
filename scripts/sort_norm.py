import numpy as np
import torch
import sys
from tqdm import trange
from token_info import *


vocab_file = sys.argv[1]
map_file = sys.argv[2]
embeddings_file = sys.argv[3]

vocab = np.load(vocab_file, allow_pickle=True)
vocab_map = np.load(map_file, allow_pickle=True)
subset_vocab = vocab[vocab_map]

embeddings = torch.load(embeddings_file)
subset_embeddings = embeddings[vocab_map,:]
row_norms = torch.norm(subset_embeddings, dim=1)

sorted_words = sorted(list(zip(row_norms, vocab)))

for norm, word in sorted_words:
    category = string_info(word)
    print(f"{norm:.8f}\t{word}\t{category}")
