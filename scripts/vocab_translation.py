import re
import sys
import numpy as np
import pickle as pk
import duckdb

dictionary = pk.load(open("dictionaries/all_clusters_df.pk", "rb"))

def preprocess(vocab):
    vocab = vocab.lower()
    vocab = vocab.strip("▁")
    return vocab

def find_english_translation(vocab):
    vocab = preprocess(vocab)
    try:
        returned_df = duckdb.query(f"SELECT vocab FROM dictionary where lang = 'en' and cluster_id in (SELECT cluster_id from dictionary where vocab == '{vocab}')").to_df()
    except:
        print(vocab)
        return ""
    if len(returned_df) > 0:
        eng_words = "|".join(returned_df['vocab'].values)
        return eng_words
    else:
        return ""


vocab_file = sys.argv[1]
output_file = sys.argv[2]
vocab = np.load(vocab_file, allow_pickle=True)

translations = np.array([ find_english_translation(v) for v in vocab ])
np.save(output_file, translations)
