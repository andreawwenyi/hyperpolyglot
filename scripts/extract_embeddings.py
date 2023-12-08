import torch

state_dict = torch.load("pytorch_model.bin")
if "roberta.embeddings.word_embeddings.weight" in state_dict:
    print("found roberta style")
    torch.save(state_dict["roberta.embeddings.word_embeddings.weight"], "embeddings.pt")
elif "encoder.embed_tokens.weight" in state_dict:
    print("found mt5 style")
    torch.save(state_dict["encoder.embed_tokens.weight"], "embeddings.pt")
elif "word_embeddings.weight" in state_dict:
    print("found bloom style")
    torch.save(state_dict["word_embeddings.weight"], "embeddings.pt")
else:
    print("no embeddings found")
    print(state_dict.keys())
