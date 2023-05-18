import sys
from collections import Counter

## certain pieces seem to keep popping up, are they more frequent?

sim_counter = Counter()

with open(sys.argv[1]) as reader:
    for line in reader:
        fields = line.rstrip().split("\t")
        if len(fields) == 5:
            sim_counter[fields[4]] += 1
        else:
            print("wrong # of tokens: " + line.rstrip())

for token, count in sim_counter.most_common(30):
    print(f"{count}\t{token}")
