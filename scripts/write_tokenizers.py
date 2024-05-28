import tiktoken
import numpy as np

import unicodedata, sys
from collections import Counter

category_map = { "C": "Control",
                 "M": "Mark",
                 "N": "Number",
                 "P": "Punctuation",
                 "S": "Symbol",
                 "Z": "Separator" }

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


#print("model_name\ttoken_idx\ttoken\ttoken_category")
for model_name in ["gpt2", "gpt-4", "gpt-4o"]:
    tokenizer = tiktoken.encoding_for_model(model_name)
    vocab_size = tokenizer.n_vocab

    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            category = string_info(token)
            token = token.replace(" ", "▁").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        except:
            break

        print(f"{model_name}\t{i}\t{token}\t{category}")