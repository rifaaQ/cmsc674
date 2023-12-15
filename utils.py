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

def get_random_query(text, min_len, max_len, n_samples):
    """
    Given some text, generate a number of random substring of size ranging [min_len, max_len].
    """
    words = text.split()

    if min_len > max_len or min_len < 1:
        raise ValueError("Invalid size parameters.")
    if max_len > len(words):
        raise ValueError("Max size length too large.")

    queries = []
    # generate random samples
    for i in range(n_samples):
        # get random index to look at
        query_len = random.randint(min_len, max_len)
        rand_idx = random.randint(0, len(words) - query_len)

        query = ' '.join(words[rand_idx : rand_idx+query_len])
        queries.append(query)

    return queries

def explode_list(some_list):
    """
    Explode a list of lists to one list.
    """
    return [x for sublist in some_list for x in sublist]
