import sys
import numpy as np
from collections import Counter

def load_neighbors(filename):

    neighbors = {}
    
    with open(filename) as reader:
        for line in reader:
            fields = line.rstrip().split("\t")
            sim = float(fields[0])
            source_id = int(fields[1])
            source = fields[2]
            target_id = int(fields[3])
            if len(fields) == 5:
                target = fields[4]
            else:
                target = ""

            if source not in neighbors:
                neighbors[source] = {}
            neighbors[source][target] = sim

    return neighbors

def jaccard(a, b):
    a_set = set(a.keys())
    b_set = set(b.keys())

    return (len(a_set & b_set) / len(a_set | b_set), len(a_set & b_set) / len(a_set))

left_neighbors = load_neighbors(sys.argv[1])
right_neighbors = load_neighbors(sys.argv[2])

for key in left_neighbors.keys():
    jaccard_score, percent_overlap = jaccard(left_neighbors[key], right_neighbors[key])
    percent_overlap *= 100
    print(f"{jaccard_score:.3f}\t{percent_overlap}\t{key}")
    
