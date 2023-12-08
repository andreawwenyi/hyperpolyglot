# hyperpolyglot
Code for "Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings"

```sh
# get vocab
python3 scripts/get_vocabs.py mt5 vocab/mt5_vocab.npy
# get categories of vocab
python3 scripts/vocab_categories.py vocab/mt5_vocab.npy vocab/mt5_categories.npy
```