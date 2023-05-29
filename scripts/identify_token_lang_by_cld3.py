import csv
import numpy as np
import pandas as pd
import cld3
model_family = 'mt5'
vocab_file = f'vocab/{model_family}_vocab.npy'
vocab = np.load(vocab_file, allow_pickle=True)
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)
subset_vocab = vocab[vocab_map]
def preprocess(vocab):
    vocab = vocab.lower()
    vocab = vocab.strip("▁")
    return vocab
print("Token\tLangID\tConfidence\n")
for vocab in subset_vocab:
    r = cld3.get_language(preprocess(vocab))
    print(f"{vocab}\t{r.language}\t{r.probability:.3f}\n")