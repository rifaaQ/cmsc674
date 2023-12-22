"""
This file generates vector embeddings for a dataset of documents words.
"""
import torch
import pandas as pd
import argparse
import os.path
import numpy as np

from utils import *
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Generate Vector Embeddings')
parser.add_argument('--data-path', type=str, default="./test_data",
                    help='path where data is stored (default: ./test_data')
parser.add_argument('--model-name', type=str, default='facebook/contriever',
                    help='embedding model name (default: contriever)')
parser.add_argument('--save-dir', type=str, default='./emb_examples',
                    help='path where embeddings are saved')

def mean_pooling(token_embeddings, mask):
    """
    Performs mean pooling on token embeddings.
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embedd(text, tokenizer, model):
    """
    Create vector embeddings for some text.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

    return embeddings.cpu().detach().numpy()

def generate_query_embs(query_ranges, model, tokenizer, model_name, data_path, emb_path):
    """
    Generate embeddings for the queries.
    """
    for query_size in query_ranges:
        # check existence
        emb_name = f"{emb_path}_test-{query_size[0]}-{query_size[1]}_emb.npy"
        emb_file = f"{emb_path}/{emb_name}"
        if os.path.isfile(emb_file):
            print(f"{emb_name} already exists")
            continue

        # get text
        df = pd.read_pickle(f"{data_path}/wiki_hard_test_{query_size[0]}-{query_size[1]}.pkl")
        test_text = list(df['text'])

        # embedd text
        test_emb = embedd([test_text[0]])
        for txt in tqdm(test_text[1:]):
            emb = embedd([txt])
            test_emb = np.concatenate((test_emb, emb), axis=0)

        # save
        np.save(emb_file, test_emb)
        print(f"saved: {emb_name} {test_emb.shape}\n")

def generate_doc_embs(chunk_sizes, overlaps, model, tokenizer, model_name, data_path, emb_path):
    """
    Generate embeddings for the documents.
    """
    # prepare documents
    df = pd.read_pickle(f"{data_path}/wiki_hard.pkl")
    all_text = list(df['text'])

    # start embeddings
    iter = 1
    total_iter = len(chunk_sizes)*len(overlaps)

    for chunk in chunk_sizes:
        for overlap in overlaps:
            # emb names
            ovlp_val = int(chunk*overlap)
            emb_name = f"{model_name}_chnk-{chunk}_ovlp-{ovlp_val}_emb"
            emb_file = f"{emb_path}/{emb_name}.npy"

            print(f"[{iter}/{total_iter}]: {model_name} chnk={chunk} ovlp={ovlp_val}")
            iter += 1

            # skip if embeddings already exists
            if os.path.isfile(emb_file):
                print(f"embedding exists \n")
                continue

            # get text
            splits = [split_text(x, chunk, ovlp_val) for x in all_text]
            all_text_splits = explode_list(splits)

            # generate embeddings
            embeddings = embedd([all_text_splits[0]])

            for txt in tqdm(all_text_splits[1:]):
                emb = embedd([txt])
                embeddings = np.concatenate((embeddings, emb), axis=0)

            # save embeddings
            print(embeddings.shape)
            np.save(emb_file, embeddings)
            print()

def main():
    global args
    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    emb_path = args.save_dir

    chunk_sizes = [100 + 50*x for x in range(9)]
    overlaps = [0, .05, .1, .15, .2, .25, .3, .4, .5]
    query_ranges = [(25,50),(50,100)]

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    # queries
    generate_query_embs(query_ranges, model, tokenizer, model_name, data_path, emb_path)

    # documents 
    generate_doc_embs(chunk_sizes, overlaps, model, tokenizer, model_name, data_path, emb_path)
    
if __name__ == '__main__':
    main()
