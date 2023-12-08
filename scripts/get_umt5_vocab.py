import numpy as np
import sentencepiece

tokenizer = sentencepiece.SentencePieceProcessor(
    model_file="umt5-base/vocabs_umt5.256000_sentencepiece.model"
)

vocab_array = np.empty(tokenizer.vocab_size(), dtype=object)
for i in range(tokenizer.vocab_size()):
    vocab_array[i] = tokenizer.id_to_piece(i)

np.save("vocab/umt5_vocab.npy", vocab_array)
