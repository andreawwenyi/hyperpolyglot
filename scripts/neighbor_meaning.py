import sys
import csv
import numpy as np

model_name = sys.argv[1]
model_family = model_name.split("-")[0]

vocab_file = f'vocab/{model_family}_vocab.npy'
vocab = np.load(vocab_file, allow_pickle=True)
vocab_categories_file = f'vocab/{model_family}_categories.npy'
vocab_categories = np.load(vocab_categories_file, allow_pickle=True)
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)
vocab_translations_file = f'vocab/{model_family}_translations.npy'
vocab_translations = np.load(vocab_translations_file, allow_pickle=True)

subset_vocab = vocab[vocab_map]
subset_vocab_categories = vocab_categories[vocab_map]
subset_vocab_translations = vocab_translations[vocab_map]

neighbor_file = f'{model_name}/m_x_u_neighbors.tsv'
outf = open(f"{model_name}/m_x_u_neighbors_meaning.tsv", "w")
print(outf)

with open(neighbor_file, "r") as f:
    tsv_file = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    data = list()
    for line in tsv_file:
        word_index = int(line[1])
        neighbor_index = int(line[3])
        word_translation = subset_vocab_translations[word_index]
        word_category = subset_vocab_categories[word_index]
        neighbor_translation = subset_vocab_translations[neighbor_index]
        
        if word_translation and neighbor_translation: # both are non-empty string
            has_intersection = int(bool(set(word_translation.split("|")).intersection(neighbor_translation.split("|"))))
        else:
            has_intersection = 0
        outf.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{word_category}\t{word_translation}\t{line[3]}\t{line[4]}\t{neighbor_translation}\t{has_intersection}\n")

outf.close()
