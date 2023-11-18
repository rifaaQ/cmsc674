"""
This file creates data for testing by generating a bunch of random queries of a dataset.
"""

import sys
sys.path.append('..')

from utils import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Testing Data Generation')
parser.add_argument('--data-file', type=str, default="wiki_hard.pkl",
					help='input file for pkl dataframe (default: wiki_hard.pkl')
parser.add_argument('--out-file', type=str, default="wiki_hard_test.pkl",
					help='output file for data (default: wiki_hard_pkl.txt)')
parser.add_argument('--min-qsize', type=int, default=50,
					help='Minimum query size (default: 50)')
parser.add_argument('--max-qsize', type=int, default=100,
					help='Maximum query size (default: 100)')
parser.add_argument('--num-samples', type=int, default=25,
					help='number of samples per document (default: 25)')

def main():
	global args
	args = parser.parse_args()
	df = pd.read_pickle(args.data_file)
	
	# don't need these cols
	df = df.drop(columns=['url', 'title'])
	
	min_qlen = args.min_qsize
	max_qlen = args.max_qsize
	num_samples = args.num_samples
	
	# get queries 
	df['text'] = df['text'].apply(lambda x: get_random_query(x, min_len=min_qlen, max_len=max_qlen, n_samples=num_samples))
	df = df.explode('text', ignore_index=True)
	
    # save df
	df.to_pickle(args.out_file)
    
if __name__ == '__main__':
	SEED = 47
	random.seed(SEED)
	main()
