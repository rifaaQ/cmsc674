"""
This file generates vector embeddings for a dataset of documents words.
---
I have no idea if this works, I running the actual one on collab bc I don't have a GPU.
"""
import torch
import pandas as pd
import argparse
import os.path
import numpy as np

from utils import *
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Generate Vector Embeddings')
parser.add_argument('--data-file', type=str, default="./test_data/wiki_hard.pkl",
                    help='input file for pkl dataframe (default: wiki_hard.pkl')
parser.add_argument('--model-name', type=str, default='facebook/contriever',
                    help='embedding model name (default: contriever)')

def mean_pooling(token_embeddings, mask):
    """
    Performs mean pooling on token embeddings.
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embedd(text, tokenizer, model, device):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

    return embeddings.cpu().detach().numpy()

def main():
    global args
    args = parser.parse_args()
    df = pd.read_pickle(args.data_file)

    model_name = args.model_name
    chunk_sizes = [100, 200, 300, 400]
    overlaps = [20, 50, 100]
    save_path = '.\embeddings'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # use GPU to generate embeddings, CPU is too slow
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_text = list(df['text'])

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            emb_name = f"\{model_name}_chnk-{chunk_size}_ovlp-{overlap}_emb"
            # check if embedding already exists
            if os.path.exists(save_path + emb_name):
                continue

            # split text and get all text from data
            splits = [split_text(x, chunk_size, overlap) for x in all_text]
            all_text_splits = explode_list(splits)

            # embedd text
            # I have a RAM issue so we gotta do this one by one
            embeddings = embedd(all_text_splits[0])

            for txt in tqdm(all_text[1:]):
                emb = embedd(txt)
                embeddings = np.concatenate((embeddings, emb), axis=0)
            
            # save embeddings
            np_save_file = emb_name + '.npy'
            np.save(np_save_file, embeddings)

if __name__ == '__main__':
    main()
