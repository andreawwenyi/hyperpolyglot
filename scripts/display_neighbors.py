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


left_neighbors = load_neighbors(sys.argv[1])
if len(sys.argv) == 3:
    right_neighbors = load_neighbors(sys.argv[2])
else:
    right_neighbors = None

for key in left_neighbors.keys():
    if right_neighbors != None:
        overlap = len(
            set(left_neighbors[key].keys()) & set(right_neighbors[key].keys())
        )
        total = len(left_neighbors[key])

        print(key, overlap / total)
        sorted_words = sorted(
            [(sim, val) for val, sim in left_neighbors[key].items()], reverse=True
        )
        top_100 = " ".join([val for sim, val in sorted_words])
        print(top_100)
        sorted_words = sorted(
            [(sim, val) for val, sim in right_neighbors[key].items()], reverse=True
        )
        top_100 = " ".join([val for sim, val in sorted_words])
        print(top_100)

    else:
        sorted_words = sorted(
            [(sim, val) for val, sim in left_neighbors[key].items()], reverse=True
        )
        top_100 = " ".join([val for sim, val in sorted_words])
        print(top_100)
