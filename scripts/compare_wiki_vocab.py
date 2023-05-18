import sys
import numpy as np
import regex as re

title_pattern = re.compile(r"<title>(.*)</title>")

vocab_file = sys.argv[1]
vocab = np.load(vocab_file, allow_pickle=True)

wiki_set = set()

wiki_file = sys.argv[2]
with open(wiki_file) as reader:
    for line in reader:
        line = line.strip()
        title = line.replace("<title>", "").replace("</title>", "")
        title = title.lower()
        wiki_set.add(title)

print(len(wiki_set))


shared = set()
for word in vocab:
    word = word.lower()
    word = word.replace("▁", "")
    if word in wiki_set:
        shared.add(word)
        #print("!", word)
    #else:
    #    print("X", word)

print(len(shared), len(vocab))
