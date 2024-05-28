from transformers import AutoTokenizer

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

models = [
    "FacebookAI/xlm-roberta-base",
    "google-t5/t5-base",
    "google/mt5-base",
    "EleutherAI/pythia-70m",
    "allenai/OLMo-7B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "microsoft/phi-2",
    "codellama/CodeLlama-7b-hf",
    "google/gemma-7b",
    "mistralai/Mistral-7B-v0.1"
]


print("model_name\ttoken_idx\ttoken\ttoken_category")
for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    for i in range(vocab_size):
        token = tokenizer.decode(i)
        category = string_info(token)
        token = token.replace(" ", "▁").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            

        print(f"{model_name}\t{i}\t{token}\t{category}")