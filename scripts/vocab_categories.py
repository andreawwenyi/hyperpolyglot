import unicodedata, sys
from collections import Counter
import numpy as np

category_map = {
    "C": "Control",
    "M": "Mark",
    "N": "Number",
    "P": "Punctuation",
    "S": "Symbol",
    "Z": "Separator",
}


def character_info(c):
    category = unicodedata.category(c)
    if category.startswith("L"):
        name = unicodedata.name(c)
        return name.split(" ")[0]
    else:
        return category[0]


def string_info(s):
    categories = Counter([character_info(c) for c in s])
    if len(categories) > 0:
        most_common, _ = categories.most_common(1)[0]
        return most_common
    else:
        return "Unknown"


vocab_file = sys.argv[1]
categories_file = sys.argv[2]
vocab = np.load(vocab_file, allow_pickle=True)

categories = np.array([string_info(s) for s in vocab])
np.save(categories_file, categories)

counts = Counter(categories)
print(" ".join([f"{s} ({c})" for s, c in counts.most_common()]))
