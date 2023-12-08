import unicodedata
from collections import Counter

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
