from transformers import AutoTokenizer, AutoModel
import sentencepiece
from collections import Counter
import numpy as np

#xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
#mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
stable_tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-3b")


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
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
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
u_to_b = { c:i for i,c in b_to_u.items() }

"""
bloom_vocabulary = convert(bloom_tokenizer.get_vocab(), unicode=True)
np.save("bloom_vocab.npy", bloom_vocabulary)
xlm_vocabulary = convert(xlm_tokenizer.get_vocab())
np.save("xlm_vocab.npy", xlm_vocabulary)
gpt2_vocabulary = convert(gpt2_tokenizer.get_vocab(), unicode=True)
np.save("gpt2_vocab.npy", gpt2_vocabulary)
mt5_vocabulary = convert(mt5_tokenizer.get_vocab())
np.save("mt5_vocab.npy", mt5_vocabulary)
"""
stable_vocabulary = convert(stable_tokenizer.get_vocab(), unicode=True)
np.save("stable_vocab.npy", stable_vocabulary)

