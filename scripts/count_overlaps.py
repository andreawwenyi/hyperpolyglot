import numpy as np

bloom_vocabulary = np.load("vocab/bloom_vocab.npy", allow_pickle=True)
reverse_bloom = {b: a for a, b in enumerate(bloom_vocabulary)}
xlm_vocabulary = np.load("vocab/xlm_vocab.npy", allow_pickle=True)
reverse_xlm = {b: a for a, b in enumerate(xlm_vocabulary)}
mt5_vocabulary = np.load("vocab/mt5_vocab.npy", allow_pickle=True)
reverse_mt5 = {b: a for a, b in enumerate(mt5_vocabulary)}

print("XLM", len(xlm_vocabulary))
print("mT5", len(mt5_vocabulary))
print("BLOOM", len(bloom_vocabulary))


print("XLM mT5", len(set(mt5_vocabulary) & set(xlm_vocabulary)))
print("XLM BLOOM", len(set(bloom_vocabulary) & set(xlm_vocabulary)))
print("BLOOM mT5", len(set(bloom_vocabulary) & set(mt5_vocabulary)))

intersection = set(bloom_vocabulary) & set(xlm_vocabulary)
intersection = list(intersection & set(mt5_vocabulary))

print("all", len(intersection))
