import numpy as np

bloom_vocabulary = np.load("vocab/bloom_vocab.npy", allow_pickle=True)
reverse_bloom = {b: a for a, b in enumerate(bloom_vocabulary)}
xlm_vocabulary = np.load("vocab/xlm_vocab.npy", allow_pickle=True)
reverse_xlm = {b: a for a, b in enumerate(xlm_vocabulary)}
mt5_vocabulary = np.load("vocab/mt5_vocab.npy", allow_pickle=True)
reverse_mt5 = {b: a for a, b in enumerate(mt5_vocabulary)}
umt5_vocabulary = np.load("vocab/umt5_vocab.npy", allow_pickle=True)
reverse_umt5 = {b: a for a, b in enumerate(umt5_vocabulary)}

intersection = set(bloom_vocabulary) & set(xlm_vocabulary)
intersection = list(intersection & set(mt5_vocabulary))

bloom_map = np.array([reverse_bloom[w] for w in intersection])
xlm_map = np.array([reverse_xlm[w] for w in intersection])
mt5_map = np.array([reverse_mt5[w] for w in intersection])

np.save("vocab/bloom_map.npy", bloom_map)
np.save("vocab/xlm_map.npy", xlm_map)
np.save("vocab/mt5_map.npy", mt5_map)

xlm_mt5_intersection = list(set(xlm_vocabulary) & set(mt5_vocabulary))
print("xlm mt5", len(xlm_mt5_intersection))
xlm_map = np.array([reverse_xlm[w] for w in xlm_mt5_intersection])
np.save("vocab/m_x_xlm_map.npy", xlm_map)
mt5_map = np.array([reverse_mt5[w] for w in xlm_mt5_intersection])
np.save("vocab/m_x_mt5_map.npy", mt5_map)

mt5_umt5_intersection = set(umt5_vocabulary) & set(mt5_vocabulary)
print("mt5 umt5", len(mt5_umt5_intersection))
mt5_map = np.array([reverse_mt5[w] for w in mt5_umt5_intersection])
np.save("vocab/m_u_mt5_map.npy", mt5_map)
umt5_map = np.array([reverse_umt5[w] for w in mt5_umt5_intersection])
np.save("vocab/m_u_umt5_map.npy", mt5_map)


xlm_mt5_umt5_intersection = list(mt5_umt5_intersection & set(xlm_vocabulary))
print("xlm mt5 umt5", len(xlm_mt5_umt5_intersection))

xlm_map = np.array([reverse_xlm[w] for w in xlm_mt5_umt5_intersection])
np.save("vocab/m_x_u_xlm_map.npy", xlm_map)
mt5_map = np.array([reverse_mt5[w] for w in xlm_mt5_umt5_intersection])
np.save("vocab/m_x_u_mt5_map.npy", mt5_map)
umt5_map = np.array([reverse_umt5[w] for w in xlm_mt5_umt5_intersection])
np.save("vocab/m_x_u_umt5_map.npy", umt5_map)


np.save("vocab/null_xlm_map.npy", np.array(list(range(len(xlm_vocabulary)))))
np.save("vocab/null_mt5_map.npy", np.array(list(range(len(mt5_vocabulary)))))
np.save("vocab/null_umt5_map.npy", np.array(list(range(len(umt5_vocabulary)))))


vocab_file = "./vocab/mt5_vocab.npy"
vocab = np.load(vocab_file, allow_pickle=True)
sampled_token_map = np.random.choice(
    range(len(vocab)), size=int(len(vocab) * 0.01), replace=False
)
np.save("vocab/sampled_mt5_map.npy", sampled_token_map)

vocab_file = "./vocab/xlm_vocab.npy"
vocab = np.load(vocab_file, allow_pickle=True)
sampled_token_map = np.random.choice(
    range(len(vocab)), size=int(len(vocab) * 0.01), replace=False
)
np.save("vocab/sampled_xlm_map.npy", sampled_token_map)
