import glob
import numpy as np
import pandas as pd

import token_info

with open("all_comparisons.tsv", "w") as writer:

    writer.write("Model1\tModel2\tJaccard\tOverlap\tCategory\n")
    
    for filename in glob.glob("comparisons/*.tsv"):
        print(filename)
        #df = pd.read_csv(filename, header=None, names=["Jaccard", "Overlap", "Token"], sep="\t")
        
        model_a, model_b = filename.replace("comparisons/", "").replace(".tsv", "").split("_")

        with open(filename) as reader:
            for line in reader:
                jaccard, overlap, token = line.rstrip().split("\t")

                category = token_info.string_info(token)
                
                writer.write(f"{model_a}\t{model_b}\t{jaccard}\t{overlap}\t{category}\n")
