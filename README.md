# hyperpolyglot
Code for [Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings](https://arxiv.org/pdf/2311.18034.pdf)

```sh
# get vocab
python3 scripts/get_vocabs.py mt5 vocab/mt5_vocab.npy
# get categories of vocab
python3 scripts/vocab_categories.py vocab/mt5_vocab.npy vocab/mt5_categories.npy
# make maps
python3 scripts/make_maps.py
```
