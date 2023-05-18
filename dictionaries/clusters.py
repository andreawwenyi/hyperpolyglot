import glob
import regex as re
import pickle as pk
import pandas as pd

line_pattern = re.compile("^(.+)\s(.+)$")
file_pattern = re.compile("en-(..).txt")
eng_trans = {}

all_files = glob.glob("en-*.txt")
#all_files = ["en-de.txt", "en-da.txt", "en-sv.txt", "en-uk.txt"]

for filename in all_files:
    with open(filename) as reader:
        match = file_pattern.search(filename)
        language = match.group(1)
        
        for line in reader:
            match = line_pattern.search(line.strip())
            if match:
                english = match.group(1)
                other = match.group(2)

                if not english in eng_trans:
                    eng_trans[english] = set()

                eng_trans[english].add(f"{other}/{language}")

#for english in eng_trans.keys():
#    print(english, "|".join(eng_trans[english]))

writer = open("all_clusters.txt", "w")
with open("/share/luxlab/mimno/models/dictionaries/all_clusters.txt", "r") as f:
    for line in f.readlines():
        english, syn = line.split()
        writer.write(f'|{english}/en|{syn}\n')

#dictionary = list()
#with open("all_clusters.txt", "r") as f:
#    for i, line in enumerate(f.readlines()):
#        eng_vocab, synonymous = line.split()
#        dictionary.append({"lang": 'en', 'vocab': eng_vocab, 'cluster_id': i, "anchor": True})
#        for syn in synonymous.split("|"):
#            vocab, lang = syn.rsplit("/", 1)
#            dictionary.append({"lang": lang, 'vocab': vocab, 'cluster_id': i})

#pk.dump(pd.DataFrame(dictionary), open("all_clusters_df.pk", "wb"))

