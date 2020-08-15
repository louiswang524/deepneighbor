import pandas as pd
import embed
from embed import Embed

if __name__ == "__main__":
    data = pd.read_csv('datasets/processed_data.csv')
    emb = Embed(data)
    emb.train()
    print("Trained")
    #print(emb.sentences)
    #emb.get_embeddings().shape
    print(emb.search('Alex',k=10))
    print("Found!")
    #return embed.get_embeddings()
