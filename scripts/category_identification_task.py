import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
import sys; sys.path.insert(0, "")
import torch

model_name = sys.argv[1]

print(model_name)
stash = list()
## read X,y
model_family = model_name.split("-")[0]
vocab_file = f'vocab/{model_family}_vocab.npy'
vocab = np.load(vocab_file, allow_pickle=True)
vocab_categories_file = f'vocab/{model_family}_categories.npy'
vocab_categories = np.load(vocab_categories_file, allow_pickle=True)
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)

subset_vocab = vocab[vocab_map]
subset_vocab_categories = vocab_categories[vocab_map]

embeddings_file = f"./{model_name}/embeddings.pt"
embeddings = torch.load(embeddings_file)
subset_embeddings = embeddings[vocab_map,:].float()
row_norms = torch.norm(subset_embeddings, dim=1)

subset_embeddings = torch.div(subset_embeddings, row_norms.reshape(-1, 1))

X = subset_embeddings
y = subset_vocab_categories

## define model
lr = LogisticRegression()
#for category in variables.CATEGORIES:
for category in np.unique(subset_vocab_categories):
    y = np.array([1 if s == category else 0 for s in subset_vocab_categories])
    rus = RandomUnderSampler(random_state=150)
    if sum(y)<500:
        stash.append({"model": model_name, "n_tokens": sum(y), "category": category})
        continue
    X_resampled, y_resampled = rus.fit_resample(X, y)
    scores = cross_val_score(lr, X_resampled, y_resampled, cv=10)
    stash.append({"model": model_name, "n_tokens": sum(y), "category": category, 'scores': scores})
    print(f"{category}\t{ sum(y)}\t{np.mean(scores):.3f}\t{np.std(scores):.3f}")
#     print(sorted(Counter(y_resampled).items()))

import pickle as pk
pk.dump(stash, open(f"{model_name}/category_identification_result.pk", "wb"))