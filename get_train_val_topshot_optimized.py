import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import pickle
import torch

# Setup environment for better memory management
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

def load_data(file_path):
    data = pd.read_json(file_path, lines=True)
    data['abstract'] = data['article'].apply(lambda x: x.split("\n")[0])
    return data

elife_train = load_data("/data/dachengma/dachengma/biolaysumm2024_data/eLife_train.jsonl")
elife_val = load_data("/data/dachengma/dachengma/biolaysumm2024_data/eLife_val.jsonl")
plos_train = load_data("/data/dachengma/dachengma/biolaysumm2024_data/PLOS_train.jsonl")
plos_val = load_data("/data/dachengma/dachengma/biolaysumm2024_data/PLOS_val.jsonl")

model = BGEM3FlagModel('/data/dachengma/models/bge-m3', use_fp16=True)
model = torch.nn.DataParallel(model)
model.to('cuda')

def compute_second_max(df1, df2):
    with torch.no_grad():
        embeddings_1 = model.module.encode(df1['abstract'].tolist(), batch_size=1, max_length=8192)['dense_vecs']
        embeddings_2 = model.module.encode(df2['abstract'].tolist(), batch_size=1, max_length=8192)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    np.fill_diagonal(similarity, -np.inf)
    second_max_indices = np.argsort(similarity, axis=1)[:, -2]
    return second_max_indices

try:
    elife_indices_val = compute_second_max(elife_val, elife_train)
    elife_indices_train = compute_second_max(elife_train, elife_train)
    plos_indices_val = compute_second_max(plos_val, plos_train)
    plos_indices_train = compute_second_max(plos_train, plos_train)
except RuntimeError as e:
    print("Caught an out-of-memory error:", e)
    torch.cuda.empty_cache()
    # Optionally adjust batch_size or handle error specifically

def add_top_shot_data(df, indices, reference_df):
    df['topshot_abstract'] = reference_df.iloc[indices]['abstract'].values
    df['topshot_laysumm'] = reference_df.iloc[indices]['lay_summary'].values
    return df

elife_val = add_top_shot_data(elife_val, elife_indices_val, elife_train)
elife_train = add_top_shot_data(elife_train, elife_indices_train, elife_train)
plos_val = add_top_shot_data(plos_val, plos_indices_val, plos_train)
plos_train = add_top_shot_data(plos_train, plos_indices_train, plos_train)

elife_val.to_csv("elife_val_with_top_shot.csv")
elife_train.to_csv("elife_train_with_top_shot.csv")
plos_val.to_csv("PLOS_val_with_top_shot.csv")
plos_train.to_csv("PLOS_train_with_top_shot.csv")
