from pathlib import Path
import pandas as pd
import sys
import csv
import scipy.stats

model_families = sys.argv[1].split(",")
model_dir = Path("./")
model_names = list()
for model_family in model_families:
    model_names += [str(p) for p in model_dir.glob(f"{model_family}-*")]

with open("./all-category-category.tsv", "w") as f:
    f.write("VOCAB_CATEGORY\tNEIGHBOR_CATEGORY\tMODEL_NAME\tMEAN\tCI\n")
    for model_name in model_names:
        print(model_name)
        df = pd.read_csv(f"{model_name}/m_x_u_neighbors_categories.tsv", sep='\t', quoting=csv.QUOTE_NONE, header=None)
        df.columns = ['cos', 'vocab_index','vocab', 'vocab_category', 'neighbor_index', 'neighbor_vocab', 'neighbor_category']
        stats = df.groupby(['vocab_index', 'vocab_category','neighbor_category']).size().reset_index(name='n')

        stats = stats.groupby(['vocab_category', 'neighbor_category'])['n'].agg(['mean', 'sem']).reset_index()
        stats['ci'] = stats['sem'] * 1.96
        stats = stats.drop("sem", axis=1)
        stats = stats.fillna(0)
        stats['model_name'] = model_name
        for line in stats[['vocab_category', 'neighbor_category', 'model_name', 'mean', 'ci']].values:
            f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]:.3f}\t{line[4]:.3f}\n")
