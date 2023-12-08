import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import variables

model_name = 'mt5-xl'
model_family = model_name.split("-")[0]
vocab_file = f'vocab/{model_family}_vocab.npy'
vocab = np.load(vocab_file, allow_pickle=True)
vocab_categories_file = f'vocab/{model_family}_categories.npy'
vocab_categories = np.load(vocab_categories_file, allow_pickle=True)
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)

subset_vocab = vocab[vocab_map]
subset_vocab_categories = vocab_categories[vocab_map]

embeddings_file = f"./{model_name}/embeddings.pt"
embeddings = torch.load(embeddings_file)
subset_embeddings = embeddings[vocab_map,:].float()
row_norms = torch.norm(subset_embeddings, dim=1)

subset_embeddings = torch.div(subset_embeddings, row_norms.reshape(-1, 1))

n_neighbors = 15
min_dist = 0.2
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    metric='euclidean'
)
projection = reducer.fit_transform(subset_embeddings)


df = pd.DataFrame(projection, columns=["dim_1", "dim_2"])
df['category'] = subset_vocab_categories
df['model_name'] = model_name

model_name = 'xlm-roberta-xl'
model_family = model_name.split("-")[0]
vocab_file = f'vocab/{model_family}_vocab.npy'
vocab = np.load(vocab_file, allow_pickle=True)
vocab_categories_file = f'vocab/{model_family}_categories.npy'
vocab_categories = np.load(vocab_categories_file, allow_pickle=True)
map_file = f'vocab/m_x_u_{model_family}_map.npy'
vocab_map = np.load(map_file, allow_pickle=True)

subset_vocab = vocab[vocab_map]
subset_vocab_categories = vocab_categories[vocab_map]

embeddings_file = f"./{model_name}/embeddings.pt"
embeddings = torch.load(embeddings_file)
subset_embeddings = embeddings[vocab_map,:].float()
row_norms = torch.norm(subset_embeddings, dim=1)

subset_embeddings = torch.div(subset_embeddings, row_norms.reshape(-1, 1))

n_neighbors = 15
min_dist = 0.2
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    metric='euclidean'
)
projection = reducer.fit_transform(subset_embeddings)

df_xlm = pd.DataFrame(projection, columns=["dim_1", "dim_2"])
df_xlm['category'] = subset_vocab_categories
df_xlm['model_name'] = model_name
df_xlm = pd.DataFrame(projection, columns=["dim_1", "dim_2"])
df_xlm['category'] = subset_vocab_categories
df_xlm['model_name'] = model_name
proj_df = pd.concat((df_xlm, df))
proj_df['model_name'] = proj_df['model_name'].apply(lambda m: "mT5-XL" if m == "mt5-xl" else "XLM-RoBERTa-XL")

plt.figure(figsize=(30, 15))
sns.set_palette("Paired")
sns.set_style("white")
with sns.plotting_context(rc={"legend.fontsize":12, 'legend.title_fontsize': 12.0}):
    g = sns.relplot(
        data=proj_df[proj_df['category'].isin(variables.CATEGORIES + ["S", "N"])], 
                    x="dim_1", 
                    y="dim_2", 
        col="model_name", hue="category", style="category",
        kind="scatter",
        facet_kws={'sharey': False, "sharex": False},
#         edgecolor=None,
        alpha=0.3

    )
    for t in g._legend.texts:
        if t.get_text() == "S":
            t.set_text("S (Symbols)")
        elif t.get_text() == "N":
            t.set_text("N (Numbers)")
    g.set(xticklabels=[])
    g.set(yticklabels=[])
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set_titles(col_template="{col_name}", size=15)
    
plt.savefig("./figures/UMAP_mt5-xl_xlm-r-xl.png", bbox_inches='tight', dpi=400)

