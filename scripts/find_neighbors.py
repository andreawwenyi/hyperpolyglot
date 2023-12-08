import numpy as np
import torch
import sys
from tqdm import trange

vocab_file = sys.argv[1]
map_file = sys.argv[2]
embeddings_file = sys.argv[3]
similarity_file = sys.argv[4]

vocab = np.load(vocab_file, allow_pickle=True)
vocab_map = np.load(map_file, allow_pickle=True)
subset_vocab = vocab[vocab_map]

embeddings = torch.load(embeddings_file)
subset_embeddings = embeddings[vocab_map, :].float()
row_norms = torch.norm(subset_embeddings, dim=1)

subset_embeddings = torch.div(subset_embeddings, row_norms.reshape(-1, 1))

nrows, ncols = subset_embeddings.shape

tranche_size = 50
tranches = int(nrows / tranche_size) + 1

with open(similarity_file, "wt") as writer:
    for tranche in trange(tranches):
        start = tranche * tranche_size
        end = start + tranche_size
        cosines = torch.matmul(subset_embeddings, subset_embeddings[start:end, :].t())

        # for row in trange(nrows):
        #    cosines = subset_embeddings.mv(subset_embeddings[row,:])
        for i in range(tranche_size):
            row = start + i
            if row >= nrows:
                break

            ranked_word_ids = torch.topk(cosines[:, i], 100).indices

            outputs = []
            for word_id in ranked_word_ids:
                cosine = cosines[word_id, i]
                writer.write(
                    f"{cosine:.3f}\t{row}\t{subset_vocab[row]}\t{word_id}\t{subset_vocab[word_id]}\n"
                )
