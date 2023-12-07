"""
This file uses PCA to reduce the dimension of some embedding model and tests the accuracy.
#TODO: Update this to actual one
"""
import pandas as pd
import argparse
import os.path
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Dimensionality Reduction Test')
parser.add_argument('--data-file', type=str, default="./test_data/wiki_hard.pkl",
                    help='input file for pkl dataframe (default: wiki_hard.pkl')
parser.add_argument('--emb-path', type=str, default='./emb_examples/',
                    help='path where embeddings are stored')
parser.add_argument('--model-name', type=str, default='facebook/contriever',
                    help='embedding model name (default: contriever)')
parser.add_argument('--min-qsize', type=int, default=50,
                    help='minimum query test size (default: 50)')
parser.add_argument('--max-qsize', type=int, default=100,
                    help='maximum query test size (default: 100)')
parser.add_argument('--save-file', type=str, default='results.pkl',
                    help='results save directory')

def main():
    global args
    args = parser.parse_args()
    df = pd.read_pickle(args.data_file)

    # get query embeddings
    min_qlen = args.min_qsize
    max_qlen = args.max_qsize
    query_file = f"/contriever_test-{query_size[0]}-{query_size[1]}_emb.npy"
    # embs_query = np.load(DATA_PATH + query_file)

    dims = [int(0.005*x*768) for x in range(1,34)] + [int(0.05*(20-x)*768) for x in range(17)][::-1]
    df = pd.read_pickle(args.data_file)

    model_name = args.model_name
    save_dir = args.save_dir

    chunk_sizes = [100 + 50*x for x in range(9)]
    overlaps = [.05, .1, .15, .2, .25, .3, .4, .5]

    all_text = list(df['text'])

if __name__ == '__main__':
    main()
