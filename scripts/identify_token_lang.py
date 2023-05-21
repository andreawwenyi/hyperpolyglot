import csv
import numpy as np
import pandas as pd

data = list()
with open("/share/magpie/datasets/mc4/merged_counts.tsv", "r") as f:
    lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

    for i, line in enumerate(lines):
        data.append(line)

model_family = 'mt5'
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)

lang = pd.DataFrame(data[1:], columns=data[0])
lang['Count'] = lang['Count'].astype(int)
lang['TokenID'] = lang['TokenID'].astype(int)
lang = lang[lang['TokenID'].isin(vocab_map)].reset_index()

lang = lang.iloc[lang.reset_index().groupby(['TokenID', 'Token'])['Count'].idxmax()]

with open("/share/luxlab/andrea/models/vocab/m_x_u_mt5_vocab_lang.tsv", "w") as f:
    f.write("TokenID\tToken\tLangID\n")
    for r in lang.set_index("TokenID")[['Token', 'LangID']].reset_index().to_dict(orient='records'):
        f.write(f"{r['TokenID']}\t{r['Token']}\t{r['LangID']}\n")