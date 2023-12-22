"""
This file uses PCA to reduce the dimension of some embedding model and tests the accuracy.
"""
import pandas as pd
import argparse
import os.path
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from utils import *
from tqdm import tqdm
from itertools import product


parser = argparse.ArgumentParser(description='Dimensionality Reduction Test')
parser.add_argument('--data-path', type=str, default='./test_data/',
                    help='path were documents/queries are stored (default: ./test_data)')
parser.add_argument('--emb-path', type=str, default='./emb_examples/',
                    help='path where embeddings are stored (default: ./emb_examples)')
parser.add_argument('--save-file', type=str, default='results',
                    help='results save name')

def define_params(params):
    """
    Defines the parameters to test.
    """
    if params == None:
        params = {
            'model_name': ['contriever', 'minilm', 'bert'],
            'query_size': [(25,50), (50,100)],
            'chunk_size': [100 + 50*x for x in range(9)],
            'overlap': [.05, .1, .15, .2, .25, .3]
        }

    # return all parameter values to test
    all_test = list(product(*params.values()))

    return all_test

def dim_reduce(dimensions, params, data_path, emb_path):
    """
    Dimensionality reduction for different parameters. Returns a list of data points.
    """
    data_pts = []
    iter = 1

    param_lst = define_params(params)

    # loop through each combination
    for param in param_lst:
        # param values to test
        P = dict(zip(params.keys(), param))
        print(f"{iter}/{len(param_lst)}: {P}")
        iter += 1

        # get query embs
        query_file = f"{emb_path}/{P['model_name']}_test-{P['query_size'][0]}-{P['query_size'][1]}_emb.npy"

        if not os.path.isfile(query_file):
            print(f"{query_file} not found")
            continue

        query_embs = np.load(query_file)

        # get document embs
        ovlp_val = int(P['chunk_size']*P['overlap'])
        doc_file = f"{emb_path}/{P['model_name']}_chnk-{P['chunk_size']}_ovlp-{ovlp_val}_emb.npy"

        if not os.path.isfile(doc_file):
            print(f"{doc_file} not found")
            continue

        doc_embs = np.load(doc_file)

        # get document/query texts
        df = pd.read_pickle(f"{data_path}/wiki_hard.pkl")
        df['text_split'] = df['text'].apply(lambda x: split_text(x, chunk_size=P['chunk_size'], overlap=ovlp_val))
        df = df.explode('text_split', ignore_index=True)
        doc_ids = df['id']

        df_q = pd.read_pickle(f"{data_path}/wiki_hard_test_{P['query_size'][0]}-{P['query_size'][1]}.pkl")
        query_ids = df_q['id']

        ####### dim_reduce #######
        for dim in tqdm(dimensions):
            embs_reduced = None
            embs_query_reduced = None

            # dim reduce
            if dim >= query_embs.shape[1]:
                # no dim reduce
                embs_reduced = doc_embs
                embs_query_reduced = query_embs
            else:
                pca = PCA(n_components=dim)
                pca.fit(doc_embs)

                embs_reduced = pca.transform(doc_embs)
                embs_query_reduced = pca.transform(query_embs)

            # cosine sim and find index of best match
            cos_sim = cosine_similarity(embs_query_reduced, embs_reduced)
            close_idx = np.argmax(cos_sim, axis=1)

            # match index with document ids
            matched_ids = doc_ids[close_idx]

            # get document match accuracy
            acc = (matched_ids.values == query_ids.values).sum() / len(matched_ids) * 100

            # saved space
            # space = embs_reduced.size / (embs_reduced.shape[0]*max_dval)

            # add data point
            data_pts.append((P['model_name'], P['chunk_size'], P['overlap'], dim, acc, P['query_size']))
        print()

    return data_pts 

def main():
    global args
    args = parser.parse_args()

    data_path = args.data_path
    emb_path = args.emb_path
    save_file = args.save_file

    params = {
        'model_name': ['contriever', 'minilm', 'bert'],
        'query_size': [(25,50), (50,100)],
        'chunk_size': [100 + 50*x for x in range(9)],
        'overlap': [.05, .1, .15, .2, .25, .3]
    }

    # dimensions to test
    max_dval = 384
    dims = [int(0.005*x*max_dval) for x in range(1,34)] + [int(0.05*(20-x)*max_dval) for x in range(17)][::-1]
    data_pts = dim_reduce(dims, params, data_path, emb_path)

    # create and save dataframe
    results = pd.DataFrame(data_pts, columns=['model', 'chunk_size', 'overlap', 'embedding_dim', 'accuracy', 'query_size'])
    results.to_pickle(f"{save_file}.pkl")

if __name__ == '__main__':
    main()
