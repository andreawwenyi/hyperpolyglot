import numpy as np
from pathlib import Path
model_types = ["bloom", "mt5", "xlm", "llama"]

vocab_npy_dir = Path("./vocab/")
output_file = "combined_vocab.tsv"

with open(output_file, 'w') as f:
    f.write("model_type\ttoken_idx\ttoken\ttoken_category\n")
    for model_type in model_types:
        vocabulary = np.load(vocab_npy_dir / f"{model_type}_vocab.npy", allow_pickle=True)
        category = np.load(vocab_npy_dir / f"{model_type}_categories.npy", allow_pickle=True)
        zipped = list(zip(range(len(vocabulary)), vocabulary, category))
        for idx, vocab, cat in zipped:
            f.write(f"{model_type}\t{idx}\t{repr(vocab)[1:-1]}\t{cat}\n")