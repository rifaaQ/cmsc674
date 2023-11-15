"""
Some utility functions for text processing.
"""
import random

def split_text(text, chunk_size, overlap):
    """
    Splits text into a list of chunks of some size with some overlap.
    """
    words = text.split()
    chunks = []

    if overlap > chunk_size:
        overlap = chunk_size
    
    for i in range(0, len(words)-chunk_size+1, chunk_size-overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def get_random_query(text, size, n_samples):
    """
    Given some text, generate a number of random substring of given size.
    """
    words = text.split()

    if size > len(words):
        raise ValueError("Query size is too large.")

    # generate random indicies of text to look at
    rand_idx = random.sample(range(len(words)-size+1), n_samples)
    queries =  [' '.join(words[i:i+size]) for i in rand_idx]

    return queries

