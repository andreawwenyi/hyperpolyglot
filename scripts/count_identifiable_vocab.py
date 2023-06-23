import pickle as pk
import numpy as np
from collections import Counter
d = pk.load(open("dictionaries/all_clusters_df.pk", "rb"))
identified_vocab = d['vocab'].unique()

mt5_vocab=np.load("vocab/mt5_vocab.npy", allow_pickle=True)
xlm_vocab=np.load("vocab/xlm_vocab.npy", allow_pickle=True)
def clean_vocab(v):
	v = v.lower()
	v = v.replace("▁", "")
	return v

mt5_vocab = [clean_vocab(v) for v in mt5_vocab]
xlm_vocab = [clean_vocab(v) for v in xlm_vocab]

mt5_counter = Counter(mt5_vocab)
mt5_identified_vocab = set(mt5_counter.keys()).intersection(identified_vocab)
mt5_c = [mt5_counter[v] for v in mt5_identified_vocab]

xlm_counter = Counter(xlm_vocab)
xlm_identified_vocab = set(xlm_counter.keys()).intersection(identified_vocab)
xlm_c = [xlm_counter[v] for v in xlm_identified_vocab]

print(f"mT5\t{sum(mt5_c)/ sum(mt5_counter.values())}")
print(f"xlm\t{sum(xlm_c)/ sum(xlm_counter.values())}")


