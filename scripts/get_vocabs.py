from transformers import AutoTokenizer, AutoModel
import sentencepiece
from collections import Counter
import numpy as np
import sys

model_family = sys.argv[1]
vocab_file = sys.argv[2]

if model_family == 'xlm':
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
elif model_family == 'gpt2':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
elif model_family =='bloom':
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
elif model_family == 'mt5':
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
elif model_family =='stable':
    tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-3b")
else: 
    raise Exception (f"Unknown model_family: {model_family}")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def to_unicode(s):
    try:
        return bytes([u_to_b[c] for c in s]).decode("utf-8").replace(" ", "▁")
    except:
        return "bad: " + s


def convert(vocab_map, unicode=False):
    vocab_array = np.empty(len(vocab_map), dtype=object)
    for s, i in vocab_map.items():
        vocab_array[i] = to_unicode(s) if unicode else s
    return vocab_array


b_to_u = bytes_to_unicode()
u_to_b = {c: i for i, c in b_to_u.items()}

if model_family == 'xlm':
    vocabulary = convert(tokenizer.get_vocab())   
elif model_family == 'gpt2':
    vocabulary = convert(tokenizer.get_vocab(), unicode=True)
elif model_family =='bloom':
    vocabulary = convert(tokenizer.get_vocab(), unicode=True)
elif model_family == 'mt5':
    vocabulary = convert(tokenizer.get_vocab())
elif model_family =='stable':
    vocabulary = convert(tokenizer.get_vocab(), unicode=True)
else: 
    raise Exception (f"Unknown model_family: {model_family}")

np.save(vocab_file, vocabulary)